# Phase 4 / Tasks 1-6 — Severity Modules

All four cue modules are real, tested, and validated end-to-end against actual Pothole-600 images/masks
(`src/severity/tests/test_pipeline.py::test_compute_severity_end_to_end_on_real_image_with_midas`), not just
synthetic fixtures:

| Task | Module | What it computes |
|---|---|---|
| 1 | [geometric.py](../../src/severity/geometric.py) | `a` = pothole mask area / estimated road-surface area. No source labels `road_surface` yet (see [class_map.py](../../src/preprocessing/class_map.py)), so `road_surface_area_heuristic()` falls back to "bottom 60% of frame" — a coarse proxy, documented as a limitation, matching the blueprint's "simple road-plane heuristic" fallback option |
| 2 | [depth_proxy.py](../../src/severity/depth_proxy.py) | `d` = MiDaS (MiDaS_small, via `torch.hub`) inverse-depth differential between the pothole region and a dilated ring of surrounding road pixels |
| 3 | [shadow_heuristic.py](../../src/severity/shadow_heuristic.py) | Lightweight grayscale-darkness-ratio fallback/cross-check for `d` when MiDaS/GPU isn't available, plus an `agreement()` helper to flag when the two disagree on direction |
| 4 | [irregularity.py](../../src/severity/irregularity.py) | `i` = contour perimeter² / (4π·area), via OpenCV contour extraction |
| 5 | [fusion.py](../../src/severity/fusion.py) | Per-cue normalization (`norm_area`/`norm_depth`/`norm_irregularity`) + `S = 100·(w1·norm(a) + w2·norm(d) + w3·norm(i))`, plus the Low/Medium/High/Critical category thresholds |

[pipeline.py](../../src/severity/pipeline.py) wires all four together (`compute_severity(image, mask)` →
score + category + raw cues), with automatic MiDaS→shadow-heuristic fallback if MiDaS/GPU isn't available.

## Task 6 — Calibration: blocked, not skipped

[calibrate.py](../../src/severity/calibrate.py) implements the full grid-search procedure (weights +
normalization bounds, optimizing Spearman's ρ against human ratings) and is verified correct against
synthetic ground truth — on noise-free synthetic data it recovers ρ ≥ 0.95, proving the search and
correlation math are right.

**It has not been run against real human ratings, because the severity validation subset doesn't exist
yet.** That's Phase 2 Tasks 7-9 (select 150-250 images, recruit 2-3 raters, collect independent 1-5 ratings)
— explicitly deferred earlier since recruiting real human raters and collecting real subjective judgments
isn't something that can be done or faked from here (see the "human raters" decision in the Phase 2
conversation). Fabricating placeholder ratings to produce a rho number would misrepresent the paper's central
severity-validation claim, so `DEFAULT_DEPTH_MAX = 30.0` and `DEFAULT_IRREGULARITY_MAX = 5.0` in `fusion.py`,
and the even default weights (w1 = w2 = w3 = ⅓), are **uncalibrated placeholders** — reasonable starting
guesses, not a result.

**To finish Task 6 for real:** build `data/severity_val_subset/` (Phase 2 Task 7), collect ratings into a
CSV (Phase 2 Tasks 8-9), then run `pipeline.compute_severity_cues()` over that subset and feed the
`(a, d, i)` cues + ratings into `calibrate.calibrate()`. No further code changes should be needed — this is
purely a "the data doesn't exist yet" blocker, not a "the code doesn't work" one.

## Revision — robust cues, scale-normalized depth, configurable bounds (2026-10-26)

An audit found that several cues measured resolution or exposure rather than the pothole, and that the
placeholder bounds compressed scores (100 real test images: median S 20.5, max 53, none High/Critical).
What changed, all in `src/severity/` (callers' signatures unchanged; new parameters are optional):

| Cue | Function used by `compute_severity` now | Range | Normalization in `fusion.py` (default bound) |
|---|---|---|---|
| `a` | `geometric.area_ratio` (unchanged maths) | [0, 1] | `clip(a / area_max)`, `area_max = 1.0` (unchanged) |
| `d` (MiDaS) | `depth_proxy.normalized_depth_differential` — (ring mean − pothole mean inverse depth) / (P95 − P5 of the road region's inverse depth, pothole excluded) | dimensionless, ~±0.3 on real images | `clip(d / depth_max)`, `depth_max = 0.10` (**new**) |
| `d` (fallback) | `shadow_heuristic.shadow_contrast` — Weber contrast (ring − pothole) / ring brightness | (−∞, 1] | `clip(d / shadow_depth_max)`, `shadow_depth_max = 0.25` (**new**) |
| `i` | `irregularity.contour_roughness` — largest component, holes filled, rescaled to a canonical radius of 64 px, Gaussian-smoothed (σ = 5 % of radius), perimeter/area on an `approxPolyDP` polygon | 0 (empty), 1.0 (neutral: < `MIN_IRREGULARITY_AREA_PX` = 200 px), else ≥ 1 | `clip((i − 1)/(irregularity_max − 1))`, `irregularity_max = 5.0` (unchanged) |

* **Irregularity** is now resolution-independent: a filled circle scores 1.00–1.10 for r = 10–200 px, native or
  2×/4× nearest-upscaled (previously 1.12 → 1.51 under 4× upscaling); a 12-point star still scores ~3.4; a
  2-px spur no longer inflates the score. Edge detail finer than ~5 % of the pothole radius is deliberately
  smoothed away (treated as mask noise). Elongation also raises `i` (a 4:1 ellipse ≈ 1.87).
  The old measure is kept as `raw_contour_roughness` for comparison.
* **Depth** is RELATIVE evidence, not centimetres. MiDaS outputs inverse depth (larger = closer) with an
  unknown per-image scale and shift; dividing the ring-vs-pothole differential by the road's inverse-depth
  spread cancels both, so the old fixed bound of 30 raw units is no longer compared against numbers whose scale
  varies per image. Sign convention unchanged: deeper pothole ⇒ larger positive `d`. The raw variant remains as
  `depth_differential` / `depth_proxy` (and `compute_severity(normalize_depth=False)` reproduces the legacy cue
  with the legacy bound 30). The 0.10 default was chosen as old-bound ÷ typical MiDaS_small road spread
  (~300–600 units, sanity-checked on 6 images); it is a placeholder until fitted.
* **Shadow fallback** is normalized by ring brightness, so a uniformly brighter/darker exposure gives the same
  value. Darkness is not always depth: wet patches, puddles (dark, or bright from reflections), cast shadows of
  vehicles/poles/trees, fresh dark asphalt patches and oil stains all confound it.
* **Input handling**: all cue functions and `compute_severity` accept 0/1, 0/255, bool, float (soft masks
  thresholded at 0.5) and H×W×1 masks; empty masks give S = 0 (`depth_source = "none"`), single-pixel masks give
  finite values, holes are ignored for `i`, and a mask whose H×W differs from the image/depth map raises
  `ValueError` (it is not silently swallowed by the MiDaS fallback).
* **Configuration**: `fusion.SeverityConfig` holds every bound, the weights and the category thresholds
  (20/40/60/80) in one place; `compute_severity(config=SeverityConfig(...) | "path/to/cue_bounds.json")`.
* **Fitting bounds from data**: `fusion.calibrate_bounds(cues, percentile=95)` sets each cue's upper bound to
  the chosen percentile of its values on a reference set; `src/severity/fit_cue_bounds.py --manifest
  data/annotations/splits/train.txt` computes cues on ground-truth YOLO-seg instance masks and writes
  `evaluation/severity/cue_bounds.json`. **Not yet run on real data** — to be run on the TRAIN split only.
  This fixes the normalization range; fusion *weights* still require the human-rated subset (Task 6).

Remaining limitations: road area (for `a`) and the depth scale both use the bottom-60 % road heuristic when no
road mask exists; depth is relative and single-image (no metric depth, no multi-view); MiDaS_small is
unreliable on blur, water-filled potholes and night images; bounds and weights are uncalibrated against human
severity ratings.
