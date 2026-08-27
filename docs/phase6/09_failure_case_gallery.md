# Phase 6 / Task 9 — Failure-Case Gallery

Selected programmatically, not hand-picked to make a point:
[find_failure_cases.py](../../src/detection/find_failure_cases.py) ran `yolov8n_seg_multisource.pt`
on all 360 real test images and compared predicted box count to labeled box count, plus flagged
the darkest labeled test images (this dataset has no explicitly-tagged night images, so image
brightness is used as an honest proxy for "night/low-light," rather than claiming images are
night shots when they're just dim daytime frames — full ranked list in
`evaluation/failure_case_candidates.csv`). Annotated renders for the four cases below are in
`evaluation/failure_gallery/`.

## 1. Severe under-detection in extreme clutter

**`kaggle_potholes_yolov8_pothole_103.jpg`** — 55 labeled pothole instances (a badly deteriorated
bridge/highway surface, essentially wall-to-wall damage), only 25 predicted (recall ≈ 45% on this
one image). This is the single worst under-detection case in the entire test set by a wide margin
(next-worst is 16 labels/5 predictions). The model clearly still finds the largest, clearest
potholes (several boxes at 0.8-0.94 confidence) but loses track of smaller/overlapping ones
packed close together — exactly the "occlusion"-adjacent failure mode the plan calls out, just
caused by pothole-on-pothole crowding rather than an external occluder.

## 2. Duplicate/fragmented boxes on textured surfaces

**`kaggle_annotated_potholes_img-428.jpg`** — 6 labeled instances, 15 predicted. Several ground-
truth potholes each produce 2-3 overlapping predicted boxes at different confidences rather than
one clean box, visible directly in the rendered image (stacked "pothole 0.5x/0.7x" labels on the
same region). Non-max suppression isn't fully collapsing near-duplicate proposals for some
textures/shapes — a real precision cost even where recall looks fine.

## 3. Near-total darkness: low-confidence, mislocated box

**`kaggle_indian_roads_PotHoles_205.jpg`** — the single darkest labeled test image (mean pixel
brightness 15.3/255, essentially a night dashcam frame lit only by a headlight cone). The model
still fires confidently (0.85) but the box sits in the upper-left, off the illuminated area
entirely — it isn't clearly keyed to any visible road damage in the frame. This reads as a
genuine false-positive-shaped failure under near-zero light, not just reduced confidence.

## 4. Low light + wet/reflective surface: degraded confidence

**`kaggle_indian_roads_PotHoles_699.jpg`** — brightness 42.6 (dim but not pitch black), wet road
at night. Two overlapping boxes over a large reflective area, both at low confidence (0.33, 0.36)
— the lowest mean confidence of any labeled dark-test image. Consistent with the general pattern:
across the 8 darkest labeled test images, mean detection confidence ranges 0.34-0.72 (see
`evaluation/failure_case_candidates.csv`) — well below the 0.7-0.9+ confidences typical of clear
detections on well-lit images in the same gallery (case 1 and 2 above) — low light measurably
degrades confidence even when it doesn't cause an outright miss.

## What this doesn't cover

No genuinely waterlogged-but-well-lit example stood out as a clean *failure* in this pass — the
handful of water-filled potholes inspected in the Ultralytics validation grids
(`models/runs/eval_test-4/val_batch*.jpg`) were detected correctly, if sometimes at slightly
lower confidence than dry potholes. Worth another look once more flooded-road images are in the
dataset; the current merged sources don't have many.
