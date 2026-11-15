"""Train an experimental detector; never attaches it to the combat controller."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Local pretrained YOLO11n weights")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1] / "data" / "boss-zen-heart"
    model = YOLO(args.weights)
    model.train(
        data=str(root / "pilot-dataset-v1" / "dataset.yaml"),
        project=str(root / "experiments"), name="weak-v1",
        device=0, epochs=40, patience=15, imgsz=640, batch=8, workers=0,
        amp=False, seed=13, plots=False,
    )
    print(f"PILOT_WEIGHTS={model.trainer.save_dir / 'weights' / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
