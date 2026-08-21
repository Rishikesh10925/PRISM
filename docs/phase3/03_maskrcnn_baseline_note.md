# Phase 3 / Task 7 — Mask R-CNN Baseline: Detectron2 → torchvision substitution

The blueprint (Section 5.1) names Detectron2 as the Mask R-CNN implementation for the accuracy/speed
comparison baseline against YOLOv8-seg. Detectron2 has no official pip wheel for Windows — installing it
requires building from source against a matching Visual C++ toolchain and CUDA version, which is fragile
and not a good use of project time for what is meant to be a baseline comparison, not the paper's core
contribution.

**Substitution:** `torchvision.models.detection.maskrcnn_resnet50_fpn` — the same architecture (Mask R-CNN,
ResNet-50-FPN backbone, COCO-pretrained), officially maintained, pip-installable, and cross-platform. This
preserves the actual comparison the blueprint wants (YOLOv8-seg vs. Mask R-CNN as architectures) without
depending on a library that doesn't support this environment.

Implementation: [src/detection/maskrcnn_dataset.py](../../src/detection/maskrcnn_dataset.py) (reads our
unified YOLO-seg labels into torchvision's expected box/label/mask target format),
[train_maskrcnn.py](../../src/detection/train_maskrcnn.py), and
[evaluate_maskrcnn.py](../../src/detection/evaluate_maskrcnn.py) (mirrors evaluate_yolo.py's metric set —
box/mask mAP@0.5 and mAP@0.5:0.95 — via `torchmetrics`, so the two models' numbers are directly comparable
in the Phase 3 Task 8 results table).

If Detectron2 specifically is later required (e.g. a reviewer asks for it), the cleanest path is running it
on a Linux training host or a Colab/Kaggle notebook — it installs cleanly there — rather than on this Windows
machine.
