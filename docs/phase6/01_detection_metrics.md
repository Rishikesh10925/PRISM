# Phase 6 / Task 1 — Final Detection Metric Suite

Numbers below are read directly from `evaluation/detection_metrics_test_summary.csv` and
`evaluation/detection_metrics_test.csv`, produced by
[evaluate_yolo.py](../../src/detection/evaluate_yolo.py) running `yolov8n_seg_multisource.pt`
against the current held-out test split (`data/annotations/splits/test.txt`, 360 images, the
multi-source set). Confirmed fresh by file timestamp (metrics CSV newer than both the model
checkpoint and the test split) — this is the same run already reported in
[docs/phase3/04_detection_results.md](../phase3/04_detection_results.md), re-verified here as
Phase 6's official final number rather than recomputed, since nothing about the model or test
set changed since that run.

| Metric | Value |
|---|---|
| Box precision (mean) | 0.786 |
| Box recall (mean) | 0.629 |
| Box mAP@0.5 | 0.707 |
| Box mAP@0.5:0.95 | 0.418 |
| Mask mAP@0.5 | 0.685 |
| Mask mAP@0.5:0.95 | 0.384 |

## Per-class breakdown

Only `pothole` has labeled instances in the current merged dataset — none of the four sources
merged so far label `crack_longitudinal`, `crack_transverse`, `crack_alligator`, `road_surface`,
or `footpath`, so those rows are all-NaN and omitted (see
[class_map.py](../../src/preprocessing/class_map.py)).

| Class | Precision | Recall | Box mAP@0.5 | Mask mAP@0.5 |
|---|---|---|---|---|
| pothole | 0.786 | 0.629 | 0.707 | 0.685 |

## What "mask IoU" means here

Ultralytics' segmentation validator reports mask mAP (mask mAP@0.5 = 0.685) rather than a bare
mean IoU number — mAP is computed by matching predicted masks to ground truth by IoU threshold
and integrating precision/recall across confidence thresholds, which is a stricter and more
standard metric than a simple mean-IoU-over-matches would be. No separate raw mean-IoU number is
reported alongside it, consistent with how Phase 3 reported these same metrics.
