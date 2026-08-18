# Related Work — Draft v0

Source material: [01_annotated_bibliography.csv](01_annotated_bibliography.csv). This is a rough structural
draft for the paper's Related Work section — grouped by sub-topic as the work plan specifies. Prose will be
tightened and expanded in Phase 7 (Task 7.2) once experimental results are in and citation formatting is
finalized; the goal here is to lock the section's skeleton and argument flow.

## 3.1 Detection-only pothole/road-damage methods

The dominant line of work treats pothole/road-damage assessment purely as an object-detection or
instance-segmentation problem. YOLO-family detectors are the de facto default: Asad et al. [P08] compare
YOLOv4 and Tiny-YOLOv4 for edge deployment, reporting 90% detection accuracy at 31.76 FPS for the
real-time variant. More recent work fine-tunes YOLOv8 variants specifically for potholes: an
attention-augmented YOLOv8n-seg (Dynamic Snake Convolution + SimAM + GELU) reaches 93.8% mAP@0.5 versus
91.9% for vanilla YOLOv8n-seg [P01]; POT-YOLO adds an edge-segmentation branch to sharpen boundary
localization [P02]; and a multi-scale variant targets the small-vs-large pothole size range in a single
frame [P05]. One paper reports 99.2% mAP for a YOLOv8 instance-segmentation approach [P04] — a number high
enough to warrant a close read of the underlying test-set size before citing it as representative.
Mask R-CNN remains the standard two-stage comparison baseline: a benchmark across adverse real-world
conditions found Mask R-CNN more accurate but slower than YOLACT (0.86 vs 0.81 accuracy) [P07]. Two recent
surveys covering 2023-2025 [P11, P12] confirm this pattern across the wider literature: the field has
converged on detection accuracy as the primary metric, almost always reported as mAP/precision/recall,
rarely accompanied by any severity or urgency output.

**Positioning:** our project uses this line of work as an input stage, not a contribution. YOLOv8-seg is
adopted precisely because it is the well-validated default in this literature (P01, P04, P05), and
Mask R-CNN is retained only as the accuracy/speed comparison baseline that P07 shows is standard practice.

## 3.2 Road-damage and pothole datasets

RDD2022 [P06] is the largest and most widely used multi-national dataset (47,420 images, six countries,
four damage classes including pothole), released through the CRDDC'2022 challenge, and is the primary
source for multi-class, cross-country training data and for the open-vocabulary stretch goal's non-pothole
classes (cracks). Pothole-600 [P13] is the only shortlisted dataset with genuine depth information: 600
RGB/disparity/label collections from a vehicle-mounted stereo camera, with disparity computed via the
PT-SRP transformation. Because no other shortlisted source provides real depth ground truth, Pothole-600 is
earmarked as a validation subset for our monocular MiDaS-based depth-proxy module, not just training data.

**Positioning:** these two are complemented by Kaggle/Roboflow community pothole sets and an Indian Roads
segmentation set (see [04_dataset_shortlist.md](04_dataset_shortlist.md)) to build a merged, deduplicated
training pool spanning multiple geographies and capture conditions.

## 3.3 Severity and pavement-condition (PCI) literature

Civil engineering has a mature, decades-old severity framework in the Pavement Condition Index, but current
PCI automation work operates on a different input modality than pothole detection papers: deep-learning PCI
pipelines detect and measure cracks (width, via a skeletonization algorithm) directly toward a PCI score,
reporting 95% crack-detection and 90% crack-width-estimation accuracy [P09], while separate work predicts
PCI from tabular inspection records using classical ML (Linear Regression, Decision Tree, Random Forest,
ANN, SVM) [P10]. Neither couples to an object-detection/segmentation pipeline over dashcam-style images,
and neither produces a per-instance severity score for an individual pothole — PCI is a road-segment-level
aggregate, not a per-defect score.

**Positioning:** our severity fusion formula (area + depth-proxy + irregularity → S ∈ [0,100]) is framed as
a per-instance analogue to PCI's severity philosophy — geometry-driven, human-validated — but computed
directly from a single detection mask instead of manual/tabular inspection input, closing the modality gap
this subsection identifies.

## 3.4 Open-vocabulary detection (stretch-goal context)

Grounding DINO [P14] fuses a DINO-style transformer detector with grounded vision-language pre-training,
reaching 52.5 AP zero-shot on COCO minival and 28.7 mAP on LVIS, and can localize arbitrary text-prompted
phrases without task-specific fine-tuning. No application of Grounding DINO (or comparable open-vocabulary
detectors such as OWL-ViT) to road damage was found in this search.

**Positioning:** cited only to support Contribution 3 (stretch) — prompting with "pothole," "crack,"
"waterlogged depression," etc., compared zero-shot against our fine-tuned closed-set YOLOv8-seg model.

## 3.5 Multi-criteria maintenance prioritization

A separate civil-engineering thread addresses *which* road segments to repair first, independent of how
damage was measured: an AHP-based weighting study [P16] finds road safety (0.269), traffic volume (0.186),
and PCI (0.172) as the three highest-weighted criteria among a larger factor set, and a Fuzzy Best-Worst
Method + VIKOR framework [P15] ranks maintenance alternatives using similarly-structured multi-criteria
weighted scoring. Both operate on manually-collected agency survey data as input.

**Positioning:** these papers justify the *structure* of our prioritization formula
(P = α·S + β·RoadTypeWeight + γ·TrafficProxy + δ·RecurrenceFactor) and the relative importance of its
terms, but neither line of work couples this weighting to an automated, image-derived severity score — this
is the second half of the gap this project addresses (see [02_research_gap_statement.md](02_research_gap_statement.md)).

## Open items before this becomes submission-ready prose

- Convert inline `[Pxx]` markers to the paper's target citation format (BibTeX keys) once a venue/template is chosen.
- Fill in author names/full metadata for entries currently marked "Anonymous" in the bibliography CSV by pulling full citation data from each paper's landing page.
- Read P02, P05, P10 in full (currently abstract-only) and update reported metrics.
- Verify P04's 99.2% mAP claim against its test-set description before repeating it uncritically.
