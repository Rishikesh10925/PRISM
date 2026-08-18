# Phase 1 / Task 2 — Research Gap Statement

Based on: [01_annotated_bibliography.csv](01_annotated_bibliography.csv)

## Statement (Introduction-ready, one paragraph)

Existing work on automated pothole and road-damage assessment overwhelmingly stops at *detection*: recent
YOLOv8-seg and Mask R-CNN variants now report bounding-box/mask accuracy in the 85-99% mAP@0.5 range on
public pothole imagery (P01, P04, P07), and multiple 2024-2025 surveys (P11, P12) confirm this is where
the field has concentrated its effort. Almost none of this work produces a quantitative, reproducible
measure of *how bad* a given pothole actually is — severity is either omitted entirely or collapsed into
a coarse small/medium/large label derived from bounding-box size alone, which conflates apparent size with
actual repair urgency and ignores depth or structural degradation. The one paper found that estimates
geometric properties beyond area (P03, YOLOv8 + point-cloud fusion) requires stereo/point-cloud hardware
rather than a single dashcam-grade image, and civil-engineering severity literature such as the Pavement
Condition Index (P09, P10) operates on manual or tabular inspection data, entirely disconnected from
image-based detection pipelines. On the decision-making side, a separate body of civil-engineering
literature (P15, P16) addresses multi-criteria road-maintenance prioritization using methods like AHP or
Fuzzy BWM+VIKOR, but these frameworks consume manually-collected condition survey data as input, not
outputs of a vision pipeline — no work found here couples an image-derived severity score directly to a
multi-factor repair-priority ranking. This leaves a clear, two-part gap that is the core claim of this
project: (1) a monocular, multi-cue severity-quantification method that fuses geometric, depth-proxy, and
structural-irregularity cues into a single calibrated 0-100 score, validated against human severity
ratings rather than assumed correct; and (2) a prioritization layer that combines that score with
contextual road-risk factors (road type, traffic exposure, recurrence) to produce an actionable,
explainable repair-priority ranking — turning a detector into a decision-support system rather than
leaving prioritization as a manual step performed after detection ends.

## Traceability

| Gap claim | Supporting evidence |
|---|---|
| Detection is largely solved / saturated | P01 (93.8% mAP), P04 (99.2% mAP), P07 (0.86 acc.), P11, P12 (both surveys confirm detection dominance) |
| Severity reduced to size label or requires special hardware | P03 (needs point cloud), P08 (detection-only, no severity output at all) |
| PCI/severity literature is disconnected from image pipelines | P09, P10 (tabular/inspection-driven, not vision-pipeline-integrated) |
| Prioritization literature exists but isn't vision-coupled | P15, P16 (AHP / Fuzzy BWM+VIKOR on manually-collected survey data) |
| No open-vocabulary work found in road-damage domain | P14 (Grounding DINO applied elsewhere, not to road damage) — supports Contribution 3 as a secondary, stretch-level novelty claim, not the core gap |
