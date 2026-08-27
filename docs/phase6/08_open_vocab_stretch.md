# Phase 6 / Task 8 (Stretch, Optional) — Open-Vocabulary Comparison: Not Attempted

The work plan itself marks this task optional (2 days, stretch goal), unlike every other Phase 6
task. It was not attempted, for the same kind of reason Detectron2 was substituted rather than
forced in Phase 3 (see
[docs/phase3/03_maskrcnn_baseline_note.md](../phase3/03_maskrcnn_baseline_note.md)): Grounding
DINO's reference implementation depends on a custom CUDA op (`MultiScaleDeformableAttention`)
that needs to be compiled from source against a matching CUDA/PyTorch/MSVC toolchain — fragile on
Windows and a poor use of remaining project time for an optional comparison, not the paper's core
contribution. OWL-ViT (via `transformers`) doesn't have that specific blocker and would be the
more realistic path if this is picked back up, but was deprioritized in favor of finishing the
seven required Phase 6 tasks first.

**If this becomes a priority later:** OWL-ViT is the pragmatic starting point (pip-installable,
no custom compiled ops) — run it zero-shot with a text prompt like `"a pothole"` on the same 360
real test images `evaluate_yolo.py` uses, using the same box-mAP@0.5 computation
`evaluate_maskrcnn.py` already implements via `torchmetrics`, and compare directly against the
fine-tuned YOLOv8n-seg numbers in
[01_detection_metrics.md](01_detection_metrics.md).
