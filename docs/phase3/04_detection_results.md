# Phase 3 / Task 8 — Detection Results: YOLOv8-seg vs. Mask R-CNN

Numbers below are read directly from `evaluation/*.csv`, produced by `evaluate_yolo.py` and `evaluate_maskrcnn.py`
on held-out test splits. **Two different datasets are referenced in this doc** — see the callout right below
before reading the summary table, or the numbers will look inconsistent.

## ⚠️ Read this first: two datasets, not yet a fair head-to-head

- **YOLOv8n-seg** below is trained and evaluated on the **new multi-source dataset** (1,867 images: Pothole-600
  + 4 Kaggle sources, see [docs/phase2/04_multisource_merge_status.md](../phase2/04_multisource_merge_status.md)),
  test split = 361 images.
- **Mask R-CNN** below is still trained and evaluated on the **old Pothole-600-only dataset** (582 images, test
  split = 171 images) — it has **not** been retrained on the multi-source set yet (that's the obvious next step,
  ~1-2 hours of GPU time given how long the multi-source YOLO run took).

Comparing the two rows as-is would be comparing different test sets, which isn't a valid accuracy comparison.
The **inference-speed** comparison (YOLOv8n-seg ~3ms vs. Mask R-CNN ~67ms/image) still holds regardless of which
dataset either was trained on — that gap is architectural, not data-dependent.

## YOLOv8n-seg: old (Pothole-600 only) vs. new (multi-source) — a real, honest regression

| Dataset | Images | box mAP@0.5 | box mAP@0.5:0.95 | mask mAP@0.5 | mask mAP@0.5:0.95 |
|---|---|---|---|---|---|
| Pothole-600 only | 582 (test: 171) | 0.891 | 0.516 | 0.856 | 0.506 |
| **Multi-source (current)** | 1,867 (test: 361) | **0.707** | **0.418** | **0.685** | **0.384** |

The multi-source model scores meaningfully *lower* on every metric. This is a real result, not a bug, and it's
worth understanding rather than hiding:

1. **The task got harder.** Pothole-600 is one camera setup, one geography, one video-derived distribution —
   genuinely easier to fit and to score well on. The multi-source test set spans four independently-sourced
   datasets with different cameras, countries, resolutions, and photography styles.
2. **Label quality is more mixed.** Pothole-600's masks are human-verified. ~5,000 of the new dataset's
   instances are SAM-generated from bounding boxes (see the multi-source merge doc) — a good but imperfect
   proxy for a hand-drawn mask, and SAM's own 1.7% failure rate (88 of 5,096 boxes) means some noise entered
   the training/eval labels too.
3. **This likely means better real-world generalization, not a worse model.** A model that scores 0.89 on one
   narrow distribution and a model that scores 0.71 across four are not directly comparable "better/worse" —
   the second number reflects a harder, more realistic test. The right next check (not yet done) would be
   running the *old* Pothole-600-only checkpoint against the *new* multi-source test split, to see how much of
   the drop is "harder test set" vs. "worse model."

## Mask R-CNN (on the old Pothole-600-only dataset — not yet updated)

| Model | box mAP@0.5 | box mAP@0.5:0.95 | mask mAP@0.5 | mask mAP@0.5:0.95 | Inference speed |
|---|---|---|---|---|---|
| Mask R-CNN (torchvision, ResNet-50-FPN) | 0.777 | 0.446 | 0.768 | 0.435 | ~67 ms/image |

- Trained 15 epochs with a plain constant-LR SGD loop (see [train_maskrcnn.py](../../src/detection/train_maskrcnn.py))
  — no LR decay schedule, which Mask R-CNN recipes typically rely on to reach their reported COCO numbers.
  Treat this as a first real baseline reading, not Mask R-CNN's best possible result on either dataset.
- Detectron2 (the blueprint's original choice) isn't available on Windows; torchvision's Mask R-CNN is used
  instead, see [03_maskrcnn_baseline_note.md](03_maskrcnn_baseline_note.md).

## YOLOv8n-seg (multi-source) — per-class breakdown

Classes with no labeled instances in the current dataset (crack_longitudinal, crack_transverse, crack_alligator,
road_surface, footpath — see [class_map.py](../../src/preprocessing/class_map.py)) are omitted; none of the
merged sources label these yet.

| Class | Precision | Recall | box mAP@0.5 | mask mAP@0.5 |
|---|---|---|---|---|
| pothole | 0.786 | 0.629 | 0.707 | 0.685 |

## Next steps to make this a fair comparison

1. Retrain Mask R-CNN on the multi-source dataset (same `train_augmented.txt`/`test.txt` YOLOv8n-seg used).
2. Evaluate the *old* Pothole-600-only YOLOv8n-seg checkpoint against the *new* multi-source test split, to
   separate "harder test set" from "worse model" in the regression above.
3. Consider a manual spot-check pass on a sample of the 5,008 SAM-generated masks (Phase 2 Task 3's original
   "manually spot-check ~10-15%" step, still open) — noisy training labels are a plausible contributor to the
   lower score, worth confirming/ruling out with actual eyes on a sample.
