"""Severity Score fusion (Phase 4 Task 5): S = 100 * (w1*norm(a) + w2*norm(d) + w3*norm(i)).

Each raw cue lives on a different natural scale (a is already a ratio in [0, 1]; d is
MiDaS inverse-depth units, unbounded and per-image scale-ambiguous; i is a contour-
roughness multiple of a perfect circle's 1.0), so each gets its own normalization to
[0, 1] before the weighted sum, per blueprint Section 5.2.

The clip bounds below (DEFAULT_DEPTH_MAX, DEFAULT_IRREGULARITY_MAX) and the default
even weights (w1 = w2 = w3 = 1/3) are *uncalibrated placeholders*. Real calibration
(Phase 4 Task 6 -- grid-searching w1/w2/w3 against the severity validation subset for
best Spearman's rho with human ratings) is blocked on that subset existing, which needs
real human raters (see docs/phase2/01_dataset_download_status.md and
docs/phase4/01_severity_modules.md for the current status). calibrate.py implements and
tests the calibration procedure itself against synthetic data so it's ready to run the
moment real ratings exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_DEPTH_MAX = 30.0  # MiDaS_small inverse-depth units; see docs/phase4/01_severity_modules.md
DEFAULT_IRREGULARITY_MAX = 5.0  # contour roughness multiple of a perfect circle (i=1.0)


@dataclass
class FusionWeights:
    w_area: float = 1 / 3
    w_depth: float = 1 / 3
    w_irregularity: float = 1 / 3

    def __post_init__(self):
        total = self.w_area + self.w_depth + self.w_irregularity
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"weights must sum to 1.0, got {total}")


def norm_area(a: float) -> float:
    """a is already a ratio in [0, 1] (geometric.area_ratio clips it there); this just
    guards against any out-of-range input reaching the fusion sum."""
    return float(np.clip(a, 0.0, 1.0))


def norm_depth(d: float, depth_max: float = DEFAULT_DEPTH_MAX) -> float:
    """Clipped-linear: 0 at d <= 0 (pothole reads no deeper than its surroundings, or
    depth estimation was unreliable), 1 at d >= depth_max."""
    if depth_max <= 0:
        raise ValueError("depth_max must be positive")
    return float(np.clip(d / depth_max, 0.0, 1.0))


def norm_irregularity(i: float, irregularity_max: float = DEFAULT_IRREGULARITY_MAX) -> float:
    """Clipped-linear over [1, irregularity_max] -- i=1 (perfect circle) maps to 0,
    i>=irregularity_max maps to 1."""
    if irregularity_max <= 1.0:
        raise ValueError("irregularity_max must be > 1.0")
    return float(np.clip((i - 1.0) / (irregularity_max - 1.0), 0.0, 1.0))


def severity_score(
    a: float,
    d: float,
    i: float,
    weights: FusionWeights | None = None,
    depth_max: float = DEFAULT_DEPTH_MAX,
    irregularity_max: float = DEFAULT_IRREGULARITY_MAX,
) -> float:
    """S in [0, 100]."""
    weights = weights or FusionWeights()
    fused = (
        weights.w_area * norm_area(a)
        + weights.w_depth * norm_depth(d, depth_max)
        + weights.w_irregularity * norm_irregularity(i, irregularity_max)
    )
    return float(100.0 * fused)


SEVERITY_CATEGORIES = ["Very Low", "Low", "Medium", "High", "Critical"]


def severity_category(score: float) -> str:
    """Maps S to the Very Low/Low/Medium/High/Critical labels the dashboard/worklist
    show (blueprint Section 3C extended with a fifth tier for the demo UI) -- even
    20-point-wide bands, uncalibrated for the same reason as the fusion weights above.
    This is a display-layer mapping only; it does not change how S itself is computed."""
    if score < 20:
        return "Very Low"
    if score < 40:
        return "Low"
    if score < 60:
        return "Medium"
    if score < 80:
        return "High"
    return "Critical"
