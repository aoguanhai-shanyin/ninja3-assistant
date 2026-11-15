"""Prepare a small YOLO pilot, retaining the distinction between weak and reviewed labels."""
from ninja3_assistant.paths import project_root

import json
import shutil
from collections import Counter


ROOT = project_root() / "data" / "boss-zen-heart"
OUTPUT = ROOT / "pilot-dataset-v1"
WIDTH, HEIGHT = 1280, 720


def main():
    curated = json.loads((ROOT / "pilot-reviewed.json").read_text(encoding="utf-8"))
    proposals = json.loads((ROOT / "proposals-unreviewed.json").read_text(encoding="utf-8"))
    classes = curated["classes"]
    if OUTPUT.exists():
        raise SystemExit(f"Output already exists; preserve it: {OUTPUT}")

    splits = {"train": {}, "validation": curated["validation"], "test": curated["test"]}
    for frame in proposals["frames"]:
        path = frame["file"]
        if path.startswith("train/") and frame["boxes"]:
            splits["train"][path] = curated["train_overrides"].get(path, frame["boxes"])

    counts = {}
    for split, frames in splits.items():
        images = OUTPUT / split / "images"
        labels = OUTPUT / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        totals = Counter()
        for relative, boxes in frames.items():
            source = ROOT / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, images / source.name)
            lines = []
            for box in boxes:
                x1, y1, x2, y2 = box["xyxy"]
                if not (0 <= x1 < x2 <= WIDTH and 0 <= y1 < y2 <= HEIGHT):
                    raise ValueError(f"Invalid box: {relative} {box}")
                class_id = classes.index(box["class"])
                totals[box["class"]] += 1
                coords = ((x1 + x2) / (2 * WIDTH), (y1 + y2) / (2 * HEIGHT),
                          (x2 - x1) / WIDTH, (y2 - y1) / HEIGHT)
                lines.append(str(class_id) + " " + " ".join(f"{value:.6f}" for value in coords))
            (labels / f"{source.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[split] = {"images": len(frames), "boxes": dict(totals)}

    yaml = (f"path: {OUTPUT.as_posix()}\ntrain: train/images\nval: validation/images\n"
            f"test: test/images\nnames:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(classes)))
    (OUTPUT / "dataset.yaml").write_text(yaml, encoding="utf-8")
    (OUTPUT / "dataset-summary.json").write_text(json.dumps({
        "training_labels": f"weak proposals except {len(curated['train_overrides'])} visually corrected frames; empty proposals omitted",
        "validation_and_test": "small selected sets visually checked by assistant, approximate boxes",
        "counts": counts,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
