"""Train and export the scoped Hayabusa detector without deploying it."""

import argparse
import hashlib
import json
from pathlib import Path


CLASSES = ['fire', 'blackfire', 'shuriken', 'slash', 'flash']
CROP = [0, 90, 1280, 500]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--export-only', type=Path, metavar='BEST_PT')
    args = parser.parse_args()
    from ultralytics import YOLO
    import torch
    import ultralytics

    if not args.data.is_file() or not args.weights.is_file():
        parser.error('data and local pretrained weights must exist')
    output = args.output.resolve()
    if args.export_only:
        weights = args.export_only.resolve()
        if not weights.is_file():
            parser.error('export weights do not exist')
        output.mkdir(parents=True, exist_ok=True)
    else:
        if args.epochs < 1 or args.batch < 1:
            parser.error('epochs and batch must be positive')
        if output.exists():
            parser.error('output already exists; preserve previous experiments')
        if not torch.cuda.is_available():
            parser.error('this training run requires the configured CUDA environment')
        model = YOLO(str(args.weights.resolve()))
        model.train(data=str(args.data.resolve()), project=str(output.parent), name=output.name,
                    device=0, epochs=args.epochs, patience=12, imgsz=640, batch=args.batch,
                    workers=0, amp=False, seed=13, deterministic=True, plots=True,
                    fliplr=.5, flipud=0, degrees=0, scale=.3, mosaic=.5, close_mosaic=5)
        weights = Path(model.trainer.best)
    model = YOLO(str(weights))
    if list(model.names.values()) != CLASSES:
        raise ValueError(f'Unexpected detector class order: {model.names}')
    exported = Path(model.export(format='onnx', imgsz=640, opset=17, simplify=False,
                                 dynamic=False, nms=False, device='cpu', batch=1))
    manifest_path = args.data.resolve().parent/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    training = [row for row in manifest['frames'] if row['split'] == 'train']
    reviewed_count = sum(row.get('reviewed') is True for row in training)
    metadata = {
        'schema_version': 1, 'model': 'YOLO11n', 'classes': CLASSES,
        'crop': CROP, 'imgsz': 640, 'conf': .25, 'iou': .5,
        'input': 'RGB float32 NCHW / 255; centered letterbox with value 114',
        'output': '[1, 4 + classes, 8400]; xywh pixels, class scores; no objectness or NMS',
        'weights_sha256': sha256(weights), 'onnx_sha256': sha256(exported),
        'pretrained_sha256': sha256(args.weights), 'dataset_yaml_sha256': sha256(args.data),
        'ultralytics': ultralytics.__version__, 'torch': torch.__version__,
        'dataset_counts': manifest['counts'], 'training_reviewed_frames': reviewed_count,
        'training_unique_reviewed_frames': len({row.get('review_origin_id', row['id']) for row in training if row.get('reviewed') is True}),
        'training_sample_exposures': len(training),
        'training_sampling_policy': manifest.get('sampling_policy', 'one exposure per materialized image'),
        'dataset_manifest_sha256': sha256(manifest_path),
        'training_weak_frames': len(training)-reviewed_count,
        'training_label_source': manifest['label_source'],
        'scope': manifest.get('model_scope', 'Experimental attack detector from one edited Hayabusa teaching video.'),
        'sources': manifest.get('sources', [manifest.get('source')]),
        'limits': manifest.get('model_limits', ['Visual annotations are assistant-reviewed approximate boxes, not independently human-reviewed.',
                   'Only the fixed crop is detected; attacks outside it are not covered.',
                   'One edited video does not establish generalization or a live clear rate.',
                   'Ambiguous overlapping frames were excluded; evaluation is biased toward clearer frames.',
                   'Detection boxes are visible extents, not independently verified damage hitboxes.']),
    }
    meta_path = exported.with_suffix('.json')
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'weights': str(weights), 'onnx': str(exported),
                      'metadata': str(meta_path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
