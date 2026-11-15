"""Measure capture plus pilot inference on MuMu instance 0 without sending input."""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from mumu import MuMu


def main():
    run = PROJECT / "data" / "boss-zen-heart" / "experiments" / "weak-v1"
    model = YOLO(str(run / "weights" / "best.pt"))
    device = MuMu(r"D:\Program Files\Netease\MuMu", 0)
    capture_times, inference_times, total_times = [], [], []
    try:
        for index in range(110):
            torch.cuda.synchronize()
            start = time.perf_counter()
            frame = cv2.cvtColor(device.capture(), cv2.COLOR_RGB2BGR)
            captured = time.perf_counter()
            model.predict(frame, imgsz=640, conf=0.25, device=0, quantize=16, verbose=False)
            torch.cuda.synchronize()
            end = time.perf_counter()
            if index >= 10:
                capture_times.append((captured - start) * 1000)
                inference_times.append((end - captured) * 1000)
                total_times.append((end - start) * 1000)
    finally:
        device.close()

    def summary(values):
        return {"mean": float(np.mean(values)), "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}

    result = {"instance": 0, "screen": "current screen, not automatically classified", "samples": len(total_times),
              "capture_and_color_ms": summary(capture_times), "inference_ms": summary(inference_times),
              "total_ms": summary(total_times), "control": "not measured; no input sent"}
    (run / "live-benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
