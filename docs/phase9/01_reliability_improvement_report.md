# Reliability Improvement Report (October 2026)

A structured audit-and-fix pass over the whole PRISM pipeline, from dataset to dashboard.
Every number here was **measured** by a committed script on the stated split, unless it is
explicitly marked *fitted*, *heuristic* or *pending*. Nothing was tuned on the test split. The
test split was evaluated **once**, for the final model, at a threshold chosen on validation.

## 1. Summary

| Area | Before | After |
|---|---|---|
| Split reproducibility | Re-running the split script would have leaked augmented train copies into val/test | Leak fixed. Splits rebuild byte-for-byte and are frozen with SHA-256 checksums |
| Train→test leakage | Unknown | **None found** (all 17 cross-split near-duplicates inspected by eye) |
| SAM pseudo-label quality | Unmeasured (0 of 225 reviewed) | **98.7% usable** (95% CI 95.4–99.6%), about the same as human-drawn controls |
| Missing labels | Unknown | **6.5%** of SAM images have an unlabelled pothole |
| Mask alignment (backend/demo) | Letterbox padding ignored | `retina_masks`: val matched-instance IoU 0.788 → **0.809**, +17 GT instances matched |
| Deployment confidence threshold | 0.25, never tested | **0.35, chosen on val** (F1 0.691 → 0.710 on val) |
| Mask R-CNN evaluation | mAP computed after a 0.5 score cut | All detections scored (0.486 → 0.544 test mask mAP@0.5) |
| Severity irregularity cue | Resolution-dependent (circle 1.12 vs 1.51 after 4× upscale) | Scale-invariant (1.004 vs 1.004) |
| Severity depth cue | Raw MiDaS units against a fixed bound | Per-image scale-normalized, relative only |
| Severity vs human ratings | Never measured | **ρ = 0.31** (95% CI 0.04–0.53), single rater, n = 56 |
| Recurrence term in priority | Always 0 (never wired) | PostGIS count within 25 m of active reports |
| Road-type API failure | Silent fallback to 0.4 | Recorded per report (`road_type_source`) |
| Upload safety | No size or type checks, CORS `*`, dev secrets accepted anywhere | 15 MB cap, content-type check, configurable CORS, production refuses dev secrets |
| Backend test isolation | Tests silently ran against dev data | Isolated `test` schema |
| Automated tests | 94 | **203** (preprocessing 25, detection 17, severity 95, prioritization 29 + 2 network-skipped, backend 37) |

## 2. Data (details: [docs/phase2/05_data_quality_report.md](../phase2/05_data_quality_report.md))

- **Splits:** 1,143 / 363 / 361 images from 5 sources, all class `pothole`. Validation passes with 0 errors.
- **Leakage:** there is none between train and test. 8 train↔val and 2 val↔test near-duplicate
  pairs make validation slightly optimistic.
- **Pseudo-labels:** a blind review of 154 SAM masks, plus 20 hidden human-drawn controls, by one
  reviewer. SAM masks were rated 81% good vs 80% for the controls, so SAM is not the bottleneck.
- **Real label noise:** 6.5% of images (95% CI 3.6–11.5%) contain a pothole the original annotators
  never labelled. This pushes recall down during training and counts some correct detections as
  false positives in evaluation.
- **Class definition:** `kaggle_indian_roads` labels large broken or unpaved patches as potholes
  (median instance 16% of the frame in its `UngradedRoad` subset, vs 0.4–1% elsewhere). The reviewer
  accepted these as potholes. Experiment E1 (§3.3) tests whether removing them helps.

## 3. Detection

### 3.1 Baseline reproduction (validation split)
| Model | Val mask mAP@0.5 | mAP@0.5:0.95 | Mask recall |
|---|---|---|---|
| D0: deployed `yolov8n_seg_multisource.pt` (resumed run) | **0.728** | 0.456 | 0.644 |
| E0: clean retrain, identical config, seed 0 | 0.722 | 0.448 | 0.657 |

The baseline reproduces to within 0.6 points. D0 stays the deployed model.
Configs, seeds, environment and checkpoint hashes are in [evaluation/experiments/](../../evaluation/experiments/).

### 3.2 Final test evaluation (D0, 361 test images, run once)
| Metric | Value |
|---|---|
| Mask mAP@0.5 / mAP@0.5:0.95 | **0.685 / 0.384** |
| Box mAP@0.5 / mAP@0.5:0.95 | 0.707 / 0.418 |
| At deployed conf 0.35: instance precision / recall / F1 (mask IoU ≥ 0.5) | 0.714 / 0.627 / 0.668 |
| Mean IoU of matched instances | 0.797 |
| Images with no prediction | 25 / 361 |
| Latency, end-to-end per image (RTX 5050 Laptop, batch 1) | 18.5 ms median, 79 ms p95 |
| Model size | 6.8 MB |

For comparison, the older Pothole-600-only checkpoint scores **0.243** mask mAP@0.5 on the same test
set. The multi-source model is far more general; its lower headline number reflects a harder test set.

**Per source (recall / precision at conf 0.35):**
- Pothole-600: 0.81 / 0.87
- `kaggle_annotated_potholes`: 0.61 / 0.67
- `kaggle_potholes_yolov8`: 0.57 / 0.73
- `kaggle_severity_levels`: 0.50 / 0.55
- `kaggle_indian_roads`: 0.49 / 0.53

### 3.3 Pothole-definition experiment (E1)
E1 = E0's exact config, trained without the 142 oversized `kaggle_indian_roads` training images
(and their augmented copies). It was resumed at epoch 7 with 2 data-loading workers after a laptop
out-of-memory crash; the worker count does not affect the result.

| Model | Train images | Val mask mAP@0.5 | `val_compact` mask mAP@0.5 | Val mask P / R |
|---|---|---|---|---|
| E0 (broad definition) | 2,286 | **0.722** | 0.724 | 0.767 / **0.657** |
| E1 (compact definition) | 2,144 | 0.720 | **0.726** | **0.808** / 0.626 |

**Result: no meaningful difference** (±0.2 points, within run-to-run noise). Removing the large
patches trades about 3 points of recall for about 4 points of precision. For pothole reporting,
recall matters more, and the reviewer accepted the large patches as potholes. **Decision: keep the broad
definition.** The deployed model (D0) is unchanged, and the test split was not touched for this comparison.

### 3.4 Failure analysis ([gallery](../../evaluation/failure_gallery/test_worst_deployed_multisource.jpg))
1. **Small, distant potholes in dense scenes** are the dominant miss. One image alone has 38 of the
   268 test false negatives. A higher input resolution (960 px) is the most promising next experiment.
2. **Some false positives are unlabelled real potholes** (consistent with the 6.5% missing-label rate),
   so true precision is higher than measured.
3. **Indian Roads** has the lowest recall: mixed pothole definition, low light, and burned-in overlays.
4. Occasional over-segmentation in cluttered scenes (parked cars, puddles).

Mask R-CNN (multi-source checkpoint, corrected evaluation) reaches 0.544 test mask mAP@0.5 at
about 240 ms/image, against 0.685 and about 5 ms for YOLOv8n-seg. Its training log was never
saved, so this is a reference point, not a tuned comparison.

## 4. Severity

**Changes:**
- Robust, scale-invariant irregularity.
- Per-image normalized MiDaS depth, which is **relative evidence, not centimetres**.
- Shadow fallback measured as a contrast ratio.
- Validation for empty, tiny and malformed masks.
- One `SeverityConfig` holding every bound and threshold.
- Cue bounds fitted on the **train** split: 95th percentile over 2,496 instances
  ([cue_bounds.json](../../evaluation/severity/cue_bounds.json)).

**Validation:** 58 test-split potholes rated 1–5 by one rater (the author); 2 marked "can't judge".
Ground-truth masks are used, so detection errors don't enter.

| Configuration | Spearman ρ (95% bootstrap CI) | Kendall τ-b | Weighted κ (5 levels) | Predicted levels 1/2/3/4/5 |
|---|---|---|---|---|
| Legacy score | 0.26 (0.00–0.50) | 0.19 | 0.25 | 28/21/3/3/1 |
| Revised cues, default bounds | 0.31 (0.04–0.54) | 0.23 | 0.08 | 43/8/4/1/0 |
| **Revised cues, train-fitted bounds (deployed)** | **0.31 (0.04–0.53)** | 0.23 | **0.31** | 21/22/8/4/1 |
| Human ratings | — | — | — | 23/13/8/10/2 |

**Per-cue ρ with the ratings:**
- irregularity 0.36 (raw 0.29)
- area ratio 0.14
- depth 0.09 (raw 0.12)

**Interpretation:** the score is **weakly but positively** associated with human judgement, and the
confidence intervals are wide (n = 56, one rater). Irregularity carries most of the signal. The
depth proxy is close to uninformative here: the median normalized depth over training potholes
is about 0, meaning MiDaS_small on a single image rarely resolves pothole depth. The area cue is
limited by the bottom-60% road-area heuristic. **The severity score must not be presented as an
engineering-grade damage measure.** Fusion weights were deliberately *not* tuned on these ratings,
because they are the only held-out validation set.

## 5. Prioritization

- `P = α·S + β·RoadType + γ·Traffic + δ·Recurrence`. The Python and JS implementations are checked to
  agree by a test that runs the real `priorityFormula.js`.
- **Recurrence:** earlier reports within 25 m (PostGIS `ST_DWithin` on geography, so the distance is
  in metres) that still have a `reported` or `in_progress` detection. The current report is excluded.
  Repaired ones don't count.
- **Road type:** OpenStreetMap highway tag. Every report records whether the value came from OSM, from
  "no road found", or from an API error.
- **Traffic:** a road-type × time-of-day **heuristic**, not measured traffic. It uses the photo's EXIF
  capture hour when it's plausible, otherwise the server time (recorded per report).
- The earlier ranking evaluation ([docs/phase6/03](../phase6/03_prioritization_ranking_evaluation.md))
  uses simulated locations and times. It shows the formula's sensitivity, not real-world effectiveness.

## 6. Application hardening

- **Uploads:** 15 MB cap (413), JPEG/PNG/WebP detected from content (415), dimension limits (422) and
  sanitized filenames. A failed analysis leaves no database row and no orphaned file.
- **Configuration:** CORS origins come from `CORS_ORIGINS`. With `APP_ENV=production` the app refuses
  to start using dev secrets. Model threshold, severity bounds and recurrence radius are all set
  through environment variables.
- **Database:** migration `a3aed99e5b7e` adds `road_type_source` and `time_source`.

## 7. Limitations and next steps

1. **One severity rater.** Add 2–3 more raters and report inter-rater agreement. The rating tool is
   `src/severity/start_severity_rating.bat`.
2. **Detection recall (about 0.63)** is limited by small distant potholes and missing labels. Next:
   try 960 px input, re-label the about 6.5% of images with missed potholes, and use the about 7,800
   unused Kaggle images.
3. **Depth:** single-image relative depth is weak. A stereo, LiDAR or reference-object measurement
   would be needed for physical depth.
4. **Licenses:** the licenses of the Kaggle sources are still unconfirmed. Use them internally only.
5. **No field test** on Hyderabad roads yet.
6. The upload size cap is enforced after the request body is buffered. Production should also limit
   the body size at the reverse proxy.
