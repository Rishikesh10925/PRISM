# Phase 3 / Task 8 — Detection Results: YOLOv8-seg vs. Mask R-CNN

Numbers below are read directly from `evaluation/*.csv`, produced by `evaluate_yolo.py` and `evaluate_maskrcnn.py` on the held-out test split. See [docs/phase2/01_dataset_download_status.md](../phase2/01_dataset_download_status.md) for why this test set is Pothole-600 only (582 images, single class) rather than the full multi-source merge the blueprint scoped — RDD2022/Roboflow/Kaggle downloads are blocked in this environment pending manual/account-based access.

## Summary comparison

| Model | box mAP@0.5 | box mAP@0.5:0.95 | mask mAP@0.5 | mask mAP@0.5:0.95 |
|---|---|---|---|---|
| YOLOv8n-seg | 0.891 | 0.516 | 0.856 | 0.506 |
| Mask R-CNN (torchvision, ResNet-50-FPN) | 0.777 | 0.446 | 0.768 | 0.435 |

## YOLOv8n-seg — per-class breakdown

Classes with no labeled instances in the current dataset (crack_longitudinal, crack_transverse, crack_alligator, road_surface, footpath — see [class_map.py](../../src/preprocessing/class_map.py)) are omitted below; they'll appear once a source that labels them is added.

| Class | Precision | Recall | box mAP@0.5 | mask mAP@0.5 |
|---|---|---|---|---|
| pothole | 0.901 | 0.825 | 0.891 | 0.856 |

## Mask R-CNN inference speed

Mean inference time: 0.067 s/image over 171 test images.

## Training configuration & fairness caveats

- Both models trained on the same 478-image augmented split (`train_augmented.txt` — 239 original + 239 rain/glare/blur augmented, see [01_data_splits_and_augmentation.md](01_data_splits_and_augmentation.md)) and evaluated on the same 171-image held-out test split.
- **Not an even fight yet**: YOLOv8n-seg trained 60 epochs with Ultralytics' tuned default schedule (cosine LR, warmup, mosaic). Mask R-CNN trained 15 epochs with a plain constant-LR SGD loop (see [train_maskrcnn.py](../../src/detection/train_maskrcnn.py)) -- no LR decay schedule, which Mask R-CNN recipes typically rely on to reach their reported COCO numbers. Mask R-CNN's true accuracy ceiling on this dataset is likely higher than shown here; treat this as a first real baseline reading, not Mask R-CNN's best possible result. The **inference speed** gap (YOLOv8n-seg ~3ms/image vs. Mask R-CNN ~67ms/image, both on the same RTX 5050 GPU) is architectural and expected to hold regardless of further Mask R-CNN tuning -- a nano single-stage model vs. a ResNet-50-FPN two-stage model.
- Detectron2 (the blueprint's original choice) isn't available on Windows; torchvision's Mask R-CNN is used instead, see [03_maskrcnn_baseline_note.md](03_maskrcnn_baseline_note.md).

