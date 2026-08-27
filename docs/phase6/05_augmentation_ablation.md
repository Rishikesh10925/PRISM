# Phase 6 / Task 5 — Augmentation Ablation

Already answered by Phase 3's hyperparameter-tuning run — see
[docs/phase3/02_hyperparameter_tuning.md](../phase3/02_hyperparameter_tuning.md) for the full
writeup. Reproduced here for Phase 6 completeness, no new run needed (same checkpoints, same
171-image Pothole-600 test set, both still present in `models/` and `evaluation/`):

| Run | Train images | Box mAP@0.5 | Box mAP@0.5:0.95 | Mask mAP@0.5 | Mask mAP@0.5:0.95 |
|---|---|---|---|---|---|
| `yolov8n_seg_baseline` (no augmentation) | 239 | 0.880 | 0.484 | 0.821 | 0.478 |
| `yolov8n_seg_augmented` (+ rain/glare/blur) | 478 (239 original + 239 augmented) | **0.891** | **0.516** | **0.856** | **0.506** |

Offline rain/glare/motion-blur augmentation (photometric only, so labels are reused unchanged —
see [augmentations.py](../../src/detection/augmentations.py)) improved every metric, most
notably the stricter mAP@0.5:0.95 numbers (+0.032 box, +0.028 mask) — consistent with better
boundary localization under varied lighting/blur conditions being exactly what weather
augmentation should help with.

Note this augmentation pipeline was carried forward into the current multi-source training set
too — `train_augmented.txt` (2,285 images: the full multi-source train split plus one augmented
copy each) is what `yolov8n_seg_multisource.pt` (Phase 6 Task 1's model) was actually trained on,
so its benefit is already baked into every other Phase 6 result, not just this isolated pair.
