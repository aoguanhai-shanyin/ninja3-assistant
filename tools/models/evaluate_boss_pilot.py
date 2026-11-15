"""Compare a pilot model with stored weak proposals on the selected held-out frames."""
from ninja3_assistant.paths import project_root

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


ROOT = project_root() / "data" / "boss-zen-heart"


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1)


def score(truth, predictions, counts):
    unused = set(range(len(truth)))
    for prediction in sorted(predictions, key=lambda item: item.get("confidence", 1), reverse=True):
        candidates = [(iou(prediction["xyxy"], truth[i]["xyxy"]), i) for i in unused
                      if prediction["class"] == truth[i]["class"]]
        overlap, best = max(candidates, default=(0, None))
        if overlap >= 0.5:
            unused.remove(best)
            counts[prediction["class"]]["tp"] += 1
        else:
            counts[prediction["class"]]["fp"] += 1
    for i in unused:
        counts[truth[i]["class"]]["fn"] += 1


def annotate(frame, boxes, color, prefix):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box["xyxy"])
        label = prefix + box["class"]
        if "confidence" in box:
            label += f" {box['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()
    curated = json.loads((ROOT / "pilot-reviewed.json").read_text(encoding="utf-8"))
    proposals = json.loads((ROOT / "proposals-unreviewed.json").read_text(encoding="utf-8"))
    baseline = {row["file"]: row["boxes"] for row in proposals["frames"]}
    model = YOLO(args.weights)
    counts = {name: defaultdict(lambda: defaultdict(int)) for name in ("model", "rule")}
    timings = []
    examples = []
    samples = []
    for path, truth in curated["test"].items():
        frame = cv2.imread(str(ROOT / path))
        if frame is None:
            raise FileNotFoundError(path)
        samples.append(frame)
        result = model.predict(frame, imgsz=640, conf=0.25, device=0, quantize=16, verbose=False)[0]
        predictions = [{"class": model.names[int(cls)], "xyxy": [round(float(x), 1) for x in box],
                        "confidence": round(float(conf), 4)}
                       for box, cls, conf in zip(result.boxes.xyxy.cpu().tolist(),
                                                 result.boxes.cls.cpu().tolist(),
                                                 result.boxes.conf.cpu().tolist())]
        score(truth, predictions, counts["model"])
        score(truth, baseline[path], counts["rule"])
        overlay = frame.copy()
        annotate(overlay, truth, (0, 220, 0), "GT ")
        annotate(overlay, predictions, (0, 190, 255), "AI ")
        cv2.putText(overlay, Path(path).stem, (10, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        examples.append(cv2.resize(overlay, (640, 360)))

    # Frames are already in memory; this includes preprocessing and postprocessing but not capture.
    for frame in samples:
        model.predict(frame, imgsz=640, conf=0.25, device=0, quantize=16, verbose=False)
    for _ in range(5):
        for frame in samples:
            torch.cuda.synchronize()
            started = time.perf_counter()
            model.predict(frame, imgsz=640, conf=0.25, device=0, quantize=16, verbose=False)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - started) * 1000)

    result = {
        "scope": f"{len(curated['test'])} selected, visually reviewed frames from one held-out recording; approximate boxes",
        "thresholds": {"confidence": 0.25, "iou": 0.5},
        "counts": counts,
        "latency_ms_in_memory": {"mean": float(np.mean(timings)), "p95": float(np.percentile(timings, 95)),
                                  "samples": len(timings), "excludes": "screen capture and control"},
    }
    output = Path(args.weights).resolve().parents[1]
    (output / "heldout-pilot.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if len(examples) % 2:
        examples.append(np.zeros_like(examples[0]))
    grid = np.vstack([np.hstack(examples[i:i + 2]) for i in range(0, len(examples), 2)])
    cv2.imwrite(str(output / "heldout-pilot.jpg"), grid)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
