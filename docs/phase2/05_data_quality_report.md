# Phase 2 (hardening) — Data Quality Report

Every number below comes from a script run on 2026-10-26 against the current
`data/merged/` and `data/annotations/splits/`. Re-run with:

```
cd src/preprocessing
..\..\.venv\Scripts\python.exe validate_dataset.py          # verify against frozen manifest
..\..\.venv\Scripts\python.exe validate_dataset.py --freeze # only when a split change is intended
```

Outputs: [evaluation/data_quality/](../../evaluation/data_quality/) — `validation_report.json`,
`split_manifest.csv` (SHA-256 of every split image and label), `near_duplicates_cross_split.csv`.

## 1. Provenance and licensing

| Source | Original labels | Masks used for training | train / val / test images | License |
|---|---|---|---|---|
| Pothole-600 | Pixel masks | Original masks (human-annotated) | 239 / 172 / 171 (source's own split) | No license stated — internal use, no redistribution |
| `kaggle_annotated_potholes` | VOC boxes | **SAM pseudo-labels** | 303 / 65 / 65 | Not confirmed (Phase 1 Task 5 open item) |
| `kaggle_potholes_yolov8` | YOLO boxes | **SAM pseudo-labels** | 211 / 45 / 46 | Not confirmed |
| `kaggle_indian_roads` | YOLO boxes, 3 classes | **SAM pseudo-labels** (class 1 only) | 330 / 68 / 66 | "Unknown" per owner — internal use only |
| `kaggle_severity_levels` | VOC boxes, 3 severity classes | **SAM pseudo-labels** | 60 / 13 / 13 | Not confirmed |
| **Total** | | | **1,143 / 363 / 361** | |

- Each Kaggle source was randomly subsampled to 500 images before merging (`build_merged_dataset.py`).
- The merge-time dedup ([data/merged/dedup_report.csv](../../data/merged/dedup_report.csv)) dropped
  719 cross-source duplicates and 13 unreadable files. Most duplicates were Kaggle re-exports of the
  `kaggle_annotated_potholes` photos.
- SAM box→mask conversion: 5,008 of 5,096 boxes converted, 88 dropped (see
  [04_multisource_merge_status.md](04_multisource_merge_status.md)).
- **Pseudo-label rule:** SAM masks are model output, not ground truth. Any metric measured on a split
  that contains them is "agreement with SAM-derived labels", and should be described that way.
- `kaggle_severity_levels`' severity classes are mostly script-generated and are **not** used as
  severity ground truth anywhere.

## 2. Automated validation — passes

| Check | Result |
|---|---|
| Missing or undecodable images | 0 |
| Missing label files | 0 |
| Invalid label rows (unknown class, < 3 points, coordinates outside [0, 1], zero area) | 0 |
| Images listed in more than one split | 0 |
| Instances (train / val / test) | 2,496 / 711 / 719 — all class 0 `pothole` |
| Tiny instances (< 0.05% of image area; flagged, kept) | 89 / 25 / 56 |
| Images with an empty label file | 2 (see §4) |

Classes 1–5 in `data.yaml` (cracks, road surface, footpath) have **zero** instances in any split.

## 3. Split reproducibility and freezing

- Re-running `prepare_splits.py` (seed 0) reproduces `train.txt`, `val.txt` and `test.txt`
  **byte-for-byte**.
- **Fixed:** `prepare_splits.py` used to scan `data/merged/images/`, which also holds the 1,143
  `*_aug0` offline-augmented copies. A re-run would have assigned augmented copies of training images
  to val and test. Those copies are now excluded, and a test covers it.
- The current splits are frozen in `evaluation/data_quality/split_manifest.csv`. From now on the test
  set is fixed. `validate_dataset.py` fails if any split image or label changes, is added or is
  removed.

## 4. Leakage and near-duplicates

Perceptual hash (pHash), Hamming distance ≤ 6, across all 1,867 split images: **0 exact duplicates**
and **17 cross-split near-duplicate pairs**. Every pair was inspected visually
([figures/cross_split_near_duplicates.jpg](../../evaluation/data_quality/figures/cross_split_near_duplicates.jpg)).

| Pairs | Verdict |
|---|---|
| 7 Pothole-600 pairs (including all 3 train↔test pairs) | **False positives.** Low-texture grey asphalt gives similar hashes, but the pairs clearly show different potholes. |
| 8 Kaggle train↔val pairs | **Real near-duplicates.** Consecutive video frames or re-exports. Validation scores are slightly optimistic. |
| 2 Kaggle val↔test pairs | **Real near-duplicates.** One is the same photo in `kaggle_potholes_yolov8` and `kaggle_severity_levels`. |

**Train↔test leakage: none found.** The splits are left unchanged to keep earlier metrics comparable;
the leakage above is documented here instead.

Other findings from the same inspection:
- The two "empty label" images (`val/kaggle_indian_roads_PotHoles_39.jpg`,
  `test/kaggle_indian_roads_PotHoles_38.jpg`) **do contain a visible pothole** — they are missing
  labels, not true negatives. A correct detection on them is currently counted as a false positive.
- Some `kaggle_indian_roads` frames contain **burned-in detection overlays** from another model, e.g.
  a box labelled "UnPaved-road 0.41".

## 5. Main finding: inconsistent "pothole" definition in `kaggle_indian_roads`

Fraction of the image covered by each instance:

| Source / sub-collection | Instances | Median | 90th percentile | Share > 25% of image |
|---|---|---|---|---|
| `kaggle_potholes_yolov8` | 1,291 | 0.004 | 0.093 | 3.4% |
| `kaggle_annotated_potholes` | 1,088 | 0.012 | 0.099 | 2.9% |
| `kaggle_severity_levels` | 183 | 0.008 | 0.077 | 2.7% |
| Pothole-600 | 602 | 0.034 | 0.151 | 4.2% |
| `kaggle_indian_roads` / `PotHoles` | 366 | 0.046 | 0.204 | 6.8% |
| `kaggle_indian_roads` / `PotHoles_plus_unpaved` | 65 | 0.098 | 0.371 | 18.5% |
| `kaggle_indian_roads` / `UngradedRoad` | 40 | **0.160** | 0.324 | **30.0%** |

94 of the 464 `kaggle_indian_roads` images have an instance covering more than 20% of the frame.
Visual inspection
([figures/indian_roads_oversized_instances.jpg](../../evaluation/data_quality/figures/indian_roads_oversized_instances.jpg))
shows many of these are **unpaved or gravel road surface**, and at least one is a **speed-breaker**,
labelled as `pothole`. Some are borderline (badly broken surface that could be called a large pothole).
This source therefore uses a broader "pothole" definition than the other four. Its class-ID mapping
([03_kaggle_indian_roads_class_mapping.md](03_kaggle_indian_roads_class_mapping.md)) was set from two
visual examples per class, and the `UngradedRoad` / `*_unpaved` sub-collections may not follow it.

**Status: measured, not yet acted on.** The manual mask review (§6) includes 56 images from this source
and will put a number on the problem. What to do about it — drop the source, drop the inconsistent
sub-collections, or relabel — is decided in Phase 3. That decision must be made on train and val only;
the frozen test set is still reported as-is, alongside any cleaned subset.

## 6. SAM pseudo-label quality — awaiting human review

The `review_checklist.csv` sample created at merge time had **0 of 225** rows reviewed. A blind review
page has been built from it:

```
cd src/preprocessing
..\..\.venv\Scripts\python.exe build_mask_review.py        # -> data/merged/review_sample/mask_review.html
..\..\.venv\Scripts\python.exe summarize_mask_review.py    # after exporting mask_review_results.csv
```

- **174 images:** all 154 SAM-masked images in the sample, plus 20 Pothole-600 images with human masks
  as hidden controls. The control ratings show how strict the reviewer is.
- Ratings are good / acceptable / bad, plus flags for missed potholes, masks on non-potholes, and
  burned-in artifacts.
- The summary reports SAM and control results separately, with Wilson 95% confidence intervals.

**Results: pending** — they need a person to do the review. This section will be updated with the
measured numbers.

## 7. Open items

1. Human SAM-mask review (§6) — blocks the class-definition decision (§5).
2. Kaggle license confirmation (Phase 1 Task 5) — needs a logged-in Kaggle browser session.
3. Two missing labels (§4) — fix in Phase 3 through a documented label-correction file. Test images
   are corrected only together with re-freezing the manifest and reporting both versions.
4. About 7,800 extra box-labelled Kaggle images were left out by the 500-per-source cap — a Phase 3
   data-volume experiment.
