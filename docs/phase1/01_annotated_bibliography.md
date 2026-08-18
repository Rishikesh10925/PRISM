# Phase 1 / Task 1 — Annotated Bibliography

Full spreadsheet: [01_annotated_bibliography.csv](01_annotated_bibliography.csv) (16 papers/datasets: paper, method, dataset, reported metrics, notes)

## Search methodology

Literature was located via targeted web searches across the three areas called for by the work plan:

1. Pothole/road-damage **detection** (YOLO and Mask R-CNN family): `YOLOv8 pothole detection segmentation`, `Mask R-CNN pothole detection segmentation accuracy comparison`, `pothole severity estimation depth area classification deep learning`, `pothole detection survey review paper 2023 2024`.
2. **Road-damage datasets**: `RDD2022 road damage detection dataset paper crowdsensing`, `Pothole-600 CNRDD stereo depth pothole dataset`.
3. **Pavement Condition Index (PCI) / civil-engineering severity literature**: `pavement condition index PCI automated assessment machine learning literature review`.
4. Two adjacent areas needed to support this project's specific novelty claims: **open-vocabulary detection** (`Grounding DINO OWL-ViT open vocabulary road damage crack detection zero-shot`) and **multi-criteria maintenance prioritization** (`road maintenance repair prioritization multi-criteria decision making traffic severity ranking`).

16 sources were kept (slightly above the 10-15 target) because the prioritization and PCI angles turned out to be thin and needed extra coverage to ground Contribution 2 properly.

## Breakdown by category

| Category | Count | IDs |
|---|---|---|
| Detection (YOLO/Mask R-CNN based) | 8 | P01-P05, P07, P08 |
| Datasets | 2 | P06, P13 |
| PCI / civil-engineering severity | 2 | P09, P10 |
| Surveys | 2 | P11, P12 |
| Open-vocabulary detection | 1 | P14 |
| Prioritization / MCDM | 2 | P15, P16 |

## Key takeaways going into Task 2 (research gap) and Task 3 (related work draft)

- **Detection is a saturated problem.** Reported mAP@0.5 for YOLOv8-seg variants on pothole data clusters in the 90-99% range (P01, P04). Several papers (P07, P08, P11, P12) confirm the field has largely converged on YOLO-family detectors as the practical default, with Mask R-CNN as a slower but sometimes more accurate baseline (P07: 0.86 vs 0.81 accuracy, but with a real-time speed penalty).
- **Severity is almost always reduced to a discrete label or a single geometric proxy**, not a calibrated, multi-cue continuous score. P03 is the closest prior work (YOLOv8 + point-cloud fusion for area/depth/perimeter) but requires point-cloud/stereo hardware rather than a monocular camera. P13 (Pothole-600) is the only shortlisted dataset with real depth-proxy (disparity) ground truth and is a candidate for validating our MiDaS-based depth proxy on a subset.
- **PCI literature (P09, P10) treats severity assessment as a separate, tabular-inspection-driven ML problem**, disconnected from image-based detection pipelines — this is exactly the disconnect our severity-fusion formula is meant to bridge, and gives us civil-engineering grounding to cite when justifying the area/depth/irregularity formula.
- **Prioritization/MCDM literature (P15, P16) is civil-engineering-native (AHP, Fuzzy BWM+VIKOR) and not image-derived.** P16's empirical AHP weights (safety 0.269, traffic 0.186, PCI 0.172) support treating severity, road type, and traffic as the dominant terms in our own weighted formula.
- **Open-vocabulary detection (P14 — Grounding DINO) has not been applied to road damage** in anything found in this search — supporting the claim in the blueprint that Contribution 3 (stretch) is genuinely novel for this domain.
- No paper found combines detection + a continuous multi-cue severity score + multi-criteria prioritization into one system — this is the gap Task 2 formalizes.

## Caveats

- Several full texts (P02, P05, P10) are only accessible via abstract/preview; metrics marked "not extracted from abstract" need to be filled in once full PDFs are obtained (library/Sci-Hub-free access, or requesting via NMIMS library).
- P04's 99.2% mAP is flagged for a closer read — that number is high enough to suspect a small or easy test set, and it should not be cited uncritically in the paper's Related Work.
