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

    # classes absent from the current ground truth (e.g. crack_* / road_surface / footpath --
    # no source has labeled those yet, see class_map.py) report NaN for every metric;
    # listing them adds noise without information, so they're left out of the table.
    present_rows = [r for r in yolo_per_class_rows if r.get("box_precision", "nan") != "nan"]
    if present_rows:
        lines.append("## YOLOv8n-seg — per-class breakdown")
        lines.append("")
        lines.append(
            "Classes with no labeled instances in the current dataset (crack_longitudinal, "
            "crack_transverse, crack_alligator, road_surface, footpath — see "
            "[class_map.py](../../src/preprocessing/class_map.py)) are omitted below; they'll "
            "appear once a source that labels them is added."
        )
        lines.append("")
        lines.append("| Class | Precision | Recall | box mAP@0.5 | mask mAP@0.5 |")
        lines.append("|---|---|---|---|---|")
        for r in present_rows:
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

    if yolo_summary_rows and maskrcnn_summary_rows:
        lines.append("## Training configuration & fairness caveats")
        lines.append("")
        lines.append(
            "- Both models trained on the same 478-image augmented split "
            "(`train_augmented.txt` — 239 original + 239 rain/glare/blur augmented, see "
            "[01_data_splits_and_augmentation.md](01_data_splits_and_augmentation.md)) and evaluated "
            "on the same 171-image held-out test split."
        )
        lines.append(
            "- **Not an even fight yet**: YOLOv8n-seg trained 60 epochs with Ultralytics' tuned "
            "default schedule (cosine LR, warmup, mosaic). Mask R-CNN trained 15 epochs with a "
            "plain constant-LR SGD loop (see [train_maskrcnn.py](../../src/detection/train_maskrcnn.py)) "
            "-- no LR decay schedule, which Mask R-CNN recipes typically rely on to reach their "
            "reported COCO numbers. Mask R-CNN's true accuracy ceiling on this dataset is likely "
            "higher than shown here; treat this as a first real baseline reading, not Mask R-CNN's "
            "best possible result. The **inference speed** gap (YOLOv8n-seg ~3ms/image vs. Mask "
            "R-CNN ~67ms/image, both on the same RTX 5050 GPU) is architectural and expected to hold "
            "regardless of further Mask R-CNN tuning -- a nano single-stage model vs. a ResNet-50-FPN "
            "two-stage model."
        )
        lines.append(
            "- Detectron2 (the blueprint's original choice) isn't available on Windows; "
            "torchvision's Mask R-CNN is used instead, see "
            "[03_maskrcnn_baseline_note.md](03_maskrcnn_baseline_note.md)."
        )
        lines.append("")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(build_report(), encoding="utf-8")
    print(f"[write_results_doc] wrote {OUT_PATH}")
