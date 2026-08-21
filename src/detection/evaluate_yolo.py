"""Compute detection/segmentation metrics for a trained YOLOv8-seg checkpoint on the
held-out test split (Phase 3 Task 6 / Phase 6 Task 1): mAP@0.5, mAP@0.5:0.95,
precision/recall per class, and mask IoU (via Ultralytics' built-in seg validator).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = REPO_ROOT / "data" / "annotations" / "data.yaml"
EVAL_DIR = REPO_ROOT / "evaluation"


def evaluate(weights_path: str, split: str = "test", out_csv: str | None = None) -> dict:
    model = YOLO(weights_path)
    metrics = model.val(data=str(DATA_YAML), split=split)

    box = metrics.box
    seg = metrics.seg

    rows = []
    names = metrics.names
    for i, class_name in names.items():
        rows.append(
            {
                "class": class_name,
                "box_precision": float(box.p[i]) if i < len(box.p) else float("nan"),
                "box_recall": float(box.r[i]) if i < len(box.r) else float("nan"),
                "box_map50": float(box.ap50[i]) if i < len(box.ap50) else float("nan"),
                "box_map50_95": float(box.ap[i]) if i < len(box.ap) else float("nan"),
                "mask_map50": float(seg.ap50[i]) if i < len(seg.ap50) else float("nan"),
                "mask_map50_95": float(seg.ap[i]) if i < len(seg.ap) else float("nan"),
            }
        )

    summary = {
        "box_map50": float(box.map50),
        "box_map50_95": float(box.map),
        "box_precision_mean": float(box.mp),
        "box_recall_mean": float(box.mr),
        "mask_map50": float(seg.map50),
        "mask_map50_95": float(seg.map),
    }

    out_csv_path = Path(out_csv) if out_csv else EVAL_DIR / f"detection_metrics_{split}.csv"
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary_path = out_csv_path.with_name(out_csv_path.stem + "_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(summary.keys()))
        writer.writerow(list(summary.values()))

    print(f"[evaluate_yolo] per-class metrics -> {out_csv_path}")
    print(f"[evaluate_yolo] summary -> {summary_path}: {summary}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", help="path to a trained .pt checkpoint")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    evaluate(args.weights, split=args.split, out_csv=args.out_csv)
