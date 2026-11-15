"""Evaluate detections only against explicitly reviewed Hayabusa frame labels."""
import argparse
import json
import math
import os
from pathlib import Path
import time


CLASSES = ('fire', 'blackfire', 'shuriken', 'slash', 'flash')


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    return intersection/max(area_a+area_b-intersection, 1e-12)


def validate_boxes(boxes, *, width=1280, height=410):
    for box in boxes:
        coordinates = box.get('xyxy', ())
        if (box.get('class') not in CLASSES or len(coordinates) != 4 or
                not all(isinstance(v, (int, float)) and math.isfinite(v) for v in coordinates)):
            raise ValueError('Invalid reviewed box class or coordinates')
        x1, y1, x2, y2 = coordinates
        if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
            raise ValueError('Reviewed box falls outside the cropped image')


def clip_predictions(boxes, width=1280, height=410):
    clipped = []
    for box in boxes:
        x1, y1, x2, y2 = box['xyxy']
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            continue
        coordinates = [max(0., min(width, x1)), max(0., min(height, y1)),
                       max(0., min(width, x2)), max(0., min(height, y2))]
        if coordinates[2] > coordinates[0] and coordinates[3] > coordinates[1]:
            clipped.append({**box, 'xyxy': coordinates})
    return clipped


def load_reviewed(manifest, split):
    payload = json.loads(Path(manifest).read_text(encoding='utf-8'))
    if payload.get('classes') != list(CLASSES) or payload.get('crop_xyxy') != [0, 90, 1280, 500]:
        raise ValueError('Dataset classes or crop do not match the detector contract')
    frames = [frame for frame in payload['frames'] if frame['split'] == split]
    if not frames:
        raise ValueError(f'No {split} frames in manifest')
    if len({frame['file'] for frame in frames}) != len(frames):
        raise ValueError('Duplicate evaluation frame paths')
    for frame in frames:
        if frame.get('reviewed') is not True:
            raise ValueError(f"Refusing unreviewed evaluation labels: {frame['file']}")
        validate_boxes(frame['boxes'])
    return payload, frames


def class_curve(frames, predictions, kind, threshold):
    truths = {frame['file']: [box for box in frame['boxes'] if box['class'] == kind] for frame in frames}
    remaining = {key: set(range(len(boxes))) for key, boxes in truths.items()}
    total = sum(map(len, truths.values()))
    ordered = sorted(((box.get('confidence', 1.), frame['file'], box) for frame in frames
                      for box in predictions.get(frame['file'], []) if box['class'] == kind),
                     key=lambda item: item[0], reverse=True)
    curve, tp, fp = [], 0, 0
    for confidence, key, box in ordered:
        overlap, index = max(((iou(box['xyxy'], truths[key][i]['xyxy']), i) for i in remaining[key]),
                             default=(0., None))
        matched = overlap >= threshold and index is not None
        if matched:
            remaining[key].remove(index)
            tp += 1
        else:
            fp += 1
        curve.append((confidence, tp, fp))
    return total, curve


def average_precision(total, curve):
    """101-point interpolated AP; classes without ground truth are excluded."""
    if not total:
        return None
    points = [(tp/total, tp/(tp+fp)) for _, tp, fp in curve]
    return sum(max((precision for recall, precision in points if recall+1e-12 >= step/100), default=0.)
               for step in range(101))/101


def evaluate(frames, predictions, confidence=.25, threshold=.5):
    allowed = {frame['file'] for frame in frames}
    if set(predictions)-allowed:
        raise ValueError('Predictions contain frames outside the reviewed evaluation split')
    for boxes in predictions.values():
        validate_boxes(boxes)
        if any(not math.isfinite(box.get('confidence', 1.)) or not 0 <= box.get('confidence', 1.) <= 1 for box in boxes):
            raise ValueError('Invalid prediction confidence')
    result = {}
    aps = []
    for kind in CLASSES:
        total, curve = class_curve(frames, predictions, kind, threshold)
        filtered = [point for point in curve if point[0] >= confidence]
        _, tp, fp = filtered[-1] if filtered else (0, 0, 0)
        by_threshold = [average_precision(*class_curve(frames, predictions, kind, .5+.05*i)) for i in range(10)]
        ap50 = by_threshold[0]
        mean_ap = sum(by_threshold)/10 if total else None
        result[kind] = {'truth': total, 'tp': tp, 'fp': fp, 'fn': total-tp,
                        'precision': tp/(tp+fp) if tp+fp else None,
                        'recall': tp/total if total else None, 'ap50': ap50, 'ap50_95': mean_ap}
        if total:
            aps.append((ap50, mean_ap))
    totals = {key: sum(row[key] for row in result.values()) for key in ('truth', 'tp', 'fp', 'fn')}
    totals.update({'precision': totals['tp']/(totals['tp']+totals['fp']) if totals['tp']+totals['fp'] else None,
                   'recall': totals['tp']/totals['truth'] if totals['truth'] else None,
                   'map50': sum(row[0] for row in aps)/len(aps) if aps else None,
                   'map50_95': sum(row[1] for row in aps)/len(aps) if aps else None})
    return {'per_class': result, 'total': totals}


def predict(weights, root, frames, device):
    # Never let Ultralytics install dependencies implicitly in a user's environment.
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    import cv2
    import numpy as np

    onnx = weights.suffix.lower() == '.onnx'
    if onnx:
        from ninja3_assistant.combat.hayabusa.detector import HayabusaDetector
        model = HayabusaDetector(weights, use_gpu=device.lower() != 'cpu')
        model.confidence, model.iou = .001, .7
    else:
        from ultralytics import YOLO
        model = YOLO(str(weights), task='detect')
    predictions, timings = {}, []
    cv2.setNumThreads(1)
    for frame in frames:
        path = (root/frame['file']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Evaluation image path escapes the dataset')
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != (410, 1280):
            raise ValueError(f'Expected a 1280x410 cropped image: {path}')
        started = time.perf_counter()
        if onnx:
            full = np.zeros((720, 1280, 3), np.uint8)
            full[90:500] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictions[frame['file']] = [{'class': box.kind,
                'xyxy': [box.xyxy[0], box.xyxy[1]-90, box.xyxy[2], box.xyxy[3]-90],
                'confidence': box.confidence} for box in model.predict(full)]
        else:
            result = model.predict(image, imgsz=640, conf=.001, iou=.7, device=device, rect=False, verbose=False)[0]
            boxes = result.boxes
            names = result.names
            predictions[frame['file']] = [{'class': names[int(kind)], 'xyxy': coordinates, 'confidence': float(conf)}
                for coordinates, kind, conf in zip(boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist())]
        predictions[frame['file']] = clip_predictions(predictions[frame['file']])
        timings.append((time.perf_counter()-started)*1000)
    return predictions, timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--weights', type=Path)
    source.add_argument('--predictions', type=Path, help='JSON object mapping frame file to predicted boxes')
    parser.add_argument('--device', default='0')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    payload, frames = load_reviewed(args.manifest, args.split)
    if args.output.resolve() in {p.resolve() for p in (args.manifest, args.weights, args.predictions) if p is not None}:
        parser.error('Output would overwrite an input artifact')
    if args.predictions:
        predictions = json.loads(args.predictions.read_text(encoding='utf-8'))
        timings = []
    else:
        predictions, timings = predict(args.weights, args.manifest.parent, frames, args.device)
    baseline = {frame['file']: frame.get('weak_boxes', []) for frame in frames}
    report = {'scope': payload.get('evaluation_scope', 'Reviewed cropped frames from one edited source video; detection evaluation only.'),
              'split': args.split, 'reviewed_frames': len(frames), 'source': payload.get('sources', payload.get('source')),
              'thresholds': {'confidence': .25, 'iou': .5, 'ap_iou': [.5+.05*i for i in range(10)]},
              'ap_protocol': '101-point precision interpolation, all sizes; classes without truth excluded from mAP.',
              'model': evaluate(frames, predictions), 'rules': evaluate(frames, baseline),
              'predictions': predictions,
              'limits': payload.get('evaluation_limits', ['Weak training labels are not ground truth.',
                         'Held-out clips share one edited recording and do not establish independent-run generalization.',
                         'Ambiguous overlapping frames were excluded; these samples favor clearer scenes.',
                         'This test split is reused for diagnostic comparisons, not a blind final benchmark.',
                         'Detection precision/recall do not measure dodge success, jump control or battle clear rate.'])}
    if timings:
        ordered = sorted(timings)
        report['in_memory_ms'] = {'mean': sum(timings)/len(timings), 'p95': ordered[math.ceil(.95*len(ordered))-1],
                                   'samples': len(timings), 'includes': 'preprocess, inference, postprocess, device output read',
                                   'excludes': 'image loading, emulator capture, policy, touch and input response; first frame includes warmup'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('split', 'reviewed_frames', 'model', 'rules')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
