"""Assemble docs/phase3/04_detection_results.md from the CSVs evaluate_yolo.py and
evaluate_maskrcnn.py write to evaluation/ (Phase 3 Task 8). Re-run any time either
evaluation is re-run — this is a pure formatting step, no numbers are computed here.
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "evaluation"
OUT_PATH = REPO_ROOT / "docs" / "phase3" / "04_detection_results.md"


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fmt(value: str) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def build_report() -> str:
    yolo_summary_rows = _read_csv_rows(EVAL_DIR / "detection_metrics_test_summary.csv")
    yolo_per_class_rows = _read_csv_rows(EVAL_DIR / "detection_metrics_test.csv")
    maskrcnn_summary_rows = _read_csv_rows(EVAL_DIR / "maskrcnn_metrics_test_summary.csv")

    lines = ["# Phase 3 / Task 8 — Detection Results: YOLOv8-seg vs. Mask R-CNN", ""]
    lines.append(
        "Numbers below are read directly from `evaluation/*.csv`, produced by "
        "`evaluate_yolo.py` and `evaluate_maskrcnn.py` on the held-out test split. "
        "See [docs/phase2/01_dataset_download_status.md](../phase2/01_dataset_download_status.md) "
        "for why this test set is Pothole-600 only (582 images, single class) rather than the "
        "full multi-source merge the blueprint scoped — RDD2022/Roboflow/Kaggle downloads are "
        "blocked in this environment pending manual/account-based access."
    )
    lines.append("")

    lines.append("## Summary comparison")
    lines.append("")
    lines.append("| Model | box mAP@0.5 | box mAP@0.5:0.95 | mask mAP@0.5 | mask mAP@0.5:0.95 |")
    lines.append("|---|---|---|---|---|")
    if yolo_summary_rows:
        r = yolo_summary_rows[0]
        lines.append(
            f"| YOLOv8n-seg | {_fmt(r['box_map50'])} | {_fmt(r['box_map50_95'])} "
            f"| {_fmt(r['mask_map50'])} | {_fmt(r['mask_map50_95'])} |"
        )
    else:
        lines.append("| YOLOv8n-seg | _not yet evaluated_ | | | |")

    if maskrcnn_summary_rows:
        r = maskrcnn_summary_rows[0]
        lines.append(
            f"| Mask R-CNN (torchvision, ResNet-50-FPN) | {_fmt(r['box_map50'])} | {_fmt(r['box_map50_95'])} "
            f"| {_fmt(r['mask_map50'])} | {_fmt(r['mask_map50_95'])} |"
        )
    else:
        lines.append("| Mask R-CNN (torchvision, ResNet-50-FPN) | _not yet evaluated_ | | | |")

    lines.append("")

    if yolo_per_class_rows:
        lines.append("## YOLOv8n-seg — per-class breakdown")
        lines.append("")
        lines.append("| Class | Precision | Recall | box mAP@0.5 | mask mAP@0.5 |")
        lines.append("|---|---|---|---|---|")
        for r in yolo_per_class_rows:
            lines.append(
                f"| {r['class']} | {_fmt(r['box_precision'])} | {_fmt(r['box_recall'])} "
                f"| {_fmt(r['box_map50'])} | {_fmt(r['mask_map50'])} |"
            )
        lines.append("")

    if maskrcnn_summary_rows:
        r = maskrcnn_summary_rows[0]
        lines.append("## Mask R-CNN inference speed")
        lines.append("")
        lines.append(f"Mean inference time: {_fmt(r.get('mean_inference_time_s', 'n/a'))} s/image "
                     f"over {r.get('num_images', 'n/a')} test images.")
        lines.append("")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(build_report(), encoding="utf-8")
    print(f"[write_results_doc] wrote {OUT_PATH}")
