# Phase 2 — Multi-Source Merge Status (update)

Following the Kaggle API token being provided, three more sources were downloaded and merged alongside
Pothole-600, plus a bonus source not on the original shortlist. All are box-only formats (VOC XML or YOLO
txt), so this also completed Phase 2 Task 3 (SAM box→mask conversion), which had been an open gap.

## Sources merged

| Source | Real size | Subsampled to* | Format | Notes |
|---|---|---|---|---|
| Pothole-600 | 600 images | (all) | Real segmentation masks | Unchanged from the earlier single-source run |
| `kaggle_annotated_potholes` (D2, `chitholian/annotated-potholes-dataset`) | 665 images | 500 | VOC XML boxes | |
| `kaggle_potholes_yolov8` (D3, `anggadwisunarto/potholes-detection-yolov8`) | 1,977 images | 500 | YOLO txt boxes | own train/valid split, merged into one pool |
| `kaggle_indian_roads` (`mitangshu11/indian-roads-dataset`) | 4,462 images (2,122 pothole-labeled) | 500 | YOLO txt boxes, 3 classes | class-ID mapping resolved by visual inspection, see [03_kaggle_indian_roads_class_mapping.md](03_kaggle_indian_roads_class_mapping.md); license "unknown" — internal use only |
| `kaggle_severity_levels` (`idanbaru/annotated-potholes-with-severity-levels`, bonus) | 717 images | 500 | VOC XML boxes, 3 severity classes | Not on the original shortlist — found while browsing Kaggle for D2/D3. Severity labels are mostly script-generated (665/717), not human-rated — see [docs/phase4/01_severity_modules.md](../phase4/01_severity_modules.md) for why it can't substitute for the Phase 2 human-rater task |

*Subsampled: box-only sources need a SAM pass before they're usable (~1-2s/image on this GPU with
MobileSAM), so each new source was capped at a random 500-image subsample this round rather than processing
all ~7,800 available box-only images (~4-7 hours). Re-run `build_merged_dataset.py` with
`max_per_new_source=None` for the full volume when there's time to let it run longer.

## Real merge result

- **SAM box→mask conversion**: 5,096 box-only instances submitted, 5,008 converted successfully (98.3%),
  88 failed/dropped (empty or degenerate SAM output) — see `data/merged/sam_conversion_manifest.csv`.
- **Deduplication**: 2,600 images in → 1,868 survivors, 732 dropped as near-duplicates. The high duplicate
  count is expected: `kaggle_severity_levels` is explicitly an enhanced re-annotation of
  `kaggle_annotated_potholes`'s underlying 665 photos (same source, described in its own README), so most of
  the two sources' overlap gets correctly caught and merged down to one copy.
- **Final merged dataset**: **1,867 images, 3,927 pothole instances** — up from 582 images (Pothole-600 only).
- **Splits**: 1,143 train / 363 val / 361 test (61/19/19%, closer to the plan's 70/15/15 target now that most
  of the pool uses the stratified-random fallback split rather than Pothole-600's native 40/30/30 split).

## A real bug found and fixed along the way

`kaggle_severity_levels` has 11 XML annotation files whose `<filename>` field doesn't match the actual image
filename on disk (e.g. the XML says `0b360769-img-338.jpg`, the real file is `img-338.jpg`) — a data-quality
issue in that third-party dataset, not something to silently work around by guessing. Worse, `dedup.py` had
a real bug of its own: an image that fails to open (missing/corrupt) was never added to a duplicate cluster,
so it also never appeared in the "dropped" set the survivor list is computed from — it silently passed
through as if it were a normal unique image. The merge script then crashed with `FileNotFoundError` trying
to copy that nonexistent path into `data/merged/`. Fixed in `dedup.py`: unreadable paths are now explicitly
excluded from survivors and logged in the dedup report as `UNREADABLE`, with a regression test added
(`test_deduplicate_excludes_unreadable_paths_from_survivors`).

## Next: retraining

YOLOv8n-seg is being retrained on this larger, multi-source, multi-camera/geography dataset (still
single-class `pothole` — none of the new sources contribute crack/road-surface classes). Results will
replace the Pothole-600-only numbers in [docs/phase3/04_detection_results.md](../phase3/04_detection_results.md).
