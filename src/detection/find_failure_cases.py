"""Failure-case gallery selection (Phase 6 Task 9): run the current YOLOv8n-seg
multi-source checkpoint on every real test image, compare predicted box count to
labeled box count, and flag the images where they disagree most, plus the lowest-
confidence detections and the darkest images (a low-light proxy, since this dataset
has no explicitly tagged night images) -- all genuine, not hand-picked to make a point.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_MANIFEST = REPO_ROOT / "data" / "annotations" / "splits" / "test.txt"
LABELS_DIR = REPO_ROOT / "data" / "merged" / "labels"
MODEL_PATH = REPO_ROOT / "models" / "yolov8n_seg_multisource.pt"
OUT_DIR = REPO_ROOT / "evaluation"


def _label_box_count(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    return sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip() and line.split()[0] == "0")


def main() -> None:
    model = YOLO(str(MODEL_PATH))
    image_paths = [Path(p) for p in TEST_MANIFEST.read_text(encoding="utf-8").splitlines() if p.strip()]

    rows = []
    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        brightness = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
        label_count = _label_box_count(LABELS_DIR / f"{img_path.stem}.txt")

        result = model.predict(image, verbose=False)[0]
        confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])
        pred_count = len(confs)
        min_conf = float(confs.min()) if len(confs) else float("nan")
        mean_conf = float(confs.mean()) if len(confs) else float("nan")

        rows.append(
            {
                "image": img_path.name,
                "brightness": round(brightness, 1),
                "label_count": label_count,
                "pred_count": pred_count,
                "count_diff": pred_count - label_count,
                "min_conf": round(min_conf, 3) if confs.size else "",
                "mean_conf": round(mean_conf, 3) if confs.size else "",
            }
        )

    out_path = OUT_DIR / "failure_case_candidates.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[find_failure_cases] wrote {out_path} ({len(rows)} images)")

    missed = sorted([r for r in rows if r["label_count"] > 0], key=lambda r: r["count_diff"])[:8]
    print("\nMost under-detected (pred_count << label_count):")
    for r in missed:
        print(f"  {r['image']}: label={r['label_count']} pred={r['pred_count']} diff={r['count_diff']}")

    over = sorted([r for r in rows if r["label_count"] > 0], key=lambda r: -r["count_diff"])[:8]
    print("\nMost over-detected (pred_count >> label_count, e.g. duplicate boxes):")
    for r in over:
        print(f"  {r['image']}: label={r['label_count']} pred={r['pred_count']} diff={r['count_diff']}")

    darkest = sorted([r for r in rows if r["label_count"] > 0], key=lambda r: r["brightness"])[:8]
    print("\nDarkest labeled test images:")
    for r in darkest:
        conf = r["mean_conf"] if r["mean_conf"] != "" else "no detections"
        print(f"  {r['image']}: brightness={r['brightness']} label={r['label_count']} pred={r['pred_count']} mean_conf={conf}")


if __name__ == "__main__":
    main()
