# Kaggle "Indian Roads Dataset" (`mitangshu11/indian-roads-dataset`) — Class ID Mapping

This Kaggle mirror is **not** the same dataset as the Supervisely-format "Indian Roads (semantic segmentation)"
originally shortlisted (D7 in [docs/phase1/04_dataset_shortlist.md](../phase1/04_dataset_shortlist.md), which
cites road/footpath/shallow/pothole pixel masks under GPL 2.0). This is a different, YOLO-box-labeled dataset
described on its Kaggle page as:

> "YOLO format labelled dataset of speed-breakers, potholes and unpaved roads." — 4000+ images, 3 classes.
> License: **unknown** (stated as such by the dataset owner)

The archive ships no `data.yaml`/`classes.txt`, so which integer class ID (0/1/2) corresponds to which of the
three named classes isn't documented anywhere. Filenames (`PotHoles_*.txt`, `AN_unpaved_*.txt`) don't reliably
disambiguate either — checking class-ID frequency per filename prefix showed both prefixes contain a mix of
IDs (each photo commonly has more than one type of road feature in frame).

**Resolved by direct visual inspection**: rendered the bounding boxes from several single-class-only label
files onto their images and looked at them.

| Class ID | Visual content | Conclusion |
|---|---|---|
| 0 | A wide box spanning the full road width, over a raised/lighter patch with road-stud reflectors on it (two independent examples) | **Speed-breakers** |
| 1 | A small, irregular dark patch on otherwise-uniform paved asphalt (two independent examples) | **Potholes** |
| 2 | A box covering visibly loose, unpaved dirt/gravel road surface | **Unpaved roads** |

This also matches the order the classes are listed in the dataset's own description text, which is a useful
secondary confirmation but was not relied on alone — the visual check is what actually settles it.

**Only class 1 (pothole) is mapped into the unified schema** ([class_map.py](../../src/preprocessing/class_map.py)),
since `speed_breaker` and `unpaved_road` have no equivalent class in the current 6-class unified set. `pothole`
instances from this source: **2,122** (of 4,462 total labeled boxes across all three classes).

**License caveat**: since the dataset owner marked the license "unknown" (not the GPL 2.0 the original D7
shortlist entry assumed — that referred to a different dataset), this source is used for internal
training/experimentation only, the same treatment given to Pothole-600's ambiguous license — **not** included
in any dataset export intended for public redistribution.
