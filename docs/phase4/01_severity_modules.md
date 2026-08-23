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
