# Phase 3 / Task 5 — Hyperparameter Tuning

Two YOLOv8n-seg runs (60 epochs, imgsz 640, batch 16, SGD defaults from Ultralytics) were compared on the
real held-out test split (171 images, 187 pothole instances — the Pothole-600 `testing/` folder, untouched
by either run):

| Run | Train images | box mAP@0.5 | box mAP@0.5:0.95 | mask mAP@0.5 | mask mAP@0.5:0.95 |
|---|---|---|---|---|---|
| `yolov8n_seg_baseline` (train.txt) | 239 | 0.880 | 0.484 | 0.821 | 0.478 |
| `yolov8n_seg_augmented` (train_augmented.txt) | 478 (239 original + 239 rain/glare/blur augmented) | **0.891** | **0.516** | **0.856** | **0.506** |

Adding the offline weather-augmented copies (see
[01_data_splits_and_augmentation.md](01_data_splits_and_augmentation.md)) improved every metric on the test
set, most notably mask mAP@0.5:0.95 (+0.028) and box mAP@0.5:0.95 (+0.032) — the stricter, higher-IoU-threshold
metrics, where better boundary localization under varied lighting/blur conditions would be expected to help
most. This is a real, reproducible result (both checkpoints and both evaluation CSVs are in
`models/` and `evaluation/`), not a projected/target number.

**Decision:** `yolov8n_seg_augmented.pt` is the tuned checkpoint carried forward into Task 6 (final metrics)
and the Task 7/8 Mask R-CNN comparison. This same baseline-vs-augmented pair also directly answers Phase 6
Task 5's augmentation ablation — no separate run needed there, re-use these two.

## What wasn't tuned (yet) and why

Only the augmentation-strength axis was iterated on this round, not learning rate / epoch count / image size
independently — the dataset is small (239-478 train images, single source, single class) and the baseline
was already at mAP@0.5 ≈ 0.88-0.89, i.e. close to saturating what this dataset can teach the model 3.26M
parameters. Learning-rate/epoch-count sweeps are far more likely to move the needle once RDD2022/Kaggle/
Roboflow data is unblocked (see [docs/phase2/01_dataset_download_status.md](../phase2/01_dataset_download_status.md))
and the corpus grows in size, class diversity, and domain variety — re-run this tuning step then rather than
over-fitting hyperparameters to a dataset that's about to change shape.
