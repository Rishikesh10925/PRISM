"""Fusion-weight calibration (Phase 4 Task 6): grid-search w1/w2/w3 (and the depth_max/
irregularity_max normalization bounds) against a human-rated severity validation subset,
optimizing for Spearman's rank correlation between predicted S and human ratings.

**This cannot be run for real yet.** It needs the severity validation subset (150-250
images, independently rated 1-5 by 2-3 raters, per Phase 2 Tasks 7-9) as ground truth,
and that subset doesn't exist -- Phase 2's rater recruitment/rating collection is still
pending real human raters (see docs/phase2/01_dataset_download_status.md). Fabricating
placeholder "human ratings" to produce a rho number here would misrepresent the paper's
central severity-validation claim, so this module is deliberately just the calibration
*procedure*, tested against synthetic data to prove it's correct -- not against a real
rho result. Run `calibrate()` for real the moment
data/severity_val_subset/ratings.csv exists (see docs/phase4/01_severity_modules.md).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from fusion import FusionWeights, severity_score


@dataclass
class CalibrationResult:
    weights: FusionWeights
    depth_max: float
    irregularity_max: float
    spearman_rho: float
    mae: float  # mean absolute error, S (0-100 scale) vs human rating rescaled to 0-100


def _weight_grid(step: float = 0.1) -> list[FusionWeights]:
    """Every (w1, w2, w3) on a `step`-spaced simplex grid that sums to 1.0."""
    values = np.arange(0.0, 1.0 + step / 2, step)
    grid = []
    for w1, w2 in itertools.product(values, values):
        w3 = 1.0 - w1 - w2
        if -1e-9 <= w3 <= 1.0 + 1e-9:
            grid.append(FusionWeights(round(float(w1), 6), round(float(w2), 6), round(float(w3), 6)))
    return grid


def calibrate(
    cues: list[tuple[float, float, float]],  # (a, d, i) per image, from pipeline.compute_severity_cues
    human_ratings_1_to_5: list[float],
    weight_step: float = 0.1,
    depth_max_candidates: tuple[float, ...] = (10.0, 20.0, 30.0, 50.0),
    irregularity_max_candidates: tuple[float, ...] = (3.0, 5.0, 8.0),
) -> CalibrationResult:
    """Grid-searches weights + normalization bounds, returns the config with the best
    Spearman's rho against human_ratings_1_to_5 (rescaled to 0-100 for MAE only; rho is
    rank-based and scale-invariant so the raw 1-5 ratings are used directly for it)."""
    if len(cues) != len(human_ratings_1_to_5):
        raise ValueError("cues and human_ratings_1_to_5 must be the same length")
    if len(cues) < 3:
        raise ValueError("need at least 3 rated images to compute a meaningful correlation")

    human_ratings_0_100 = [(r - 1) / 4 * 100 for r in human_ratings_1_to_5]

    best: CalibrationResult | None = None
    for weights in _weight_grid(weight_step):
        for depth_max in depth_max_candidates:
            for irregularity_max in irregularity_max_candidates:
                predicted = [
                    severity_score(a, d, i, weights, depth_max, irregularity_max) for a, d, i in cues
                ]
                rho, _ = spearmanr(predicted, human_ratings_1_to_5)
                if np.isnan(rho):
                    continue
                mae = float(np.mean(np.abs(np.array(predicted) - np.array(human_ratings_0_100))))

                if best is None or rho > best.spearman_rho:
                    best = CalibrationResult(weights, depth_max, irregularity_max, float(rho), mae)

    if best is None:
        raise RuntimeError("calibration found no valid weight configuration (all rho were NaN)")
    return best
