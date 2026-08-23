"""Glue: run all three cue modules on one (image, pothole_mask) pair and fuse them into
a Severity Score. Not its own numbered plan task -- this is what Task 6's calibration
and Task 10's ablation actually call per-image.
"""

from __future__ import annotations

import numpy as np

from depth_proxy import depth_proxy
from fusion import FusionWeights, severity_category, severity_score
from geometric import area_ratio
from irregularity import contour_roughness
from schema import SeverityCues
from shadow_heuristic import shadow_darkness_ratio


def compute_severity_cues(
    image_bgr: np.ndarray,
    pothole_mask: np.ndarray,
    road_mask: np.ndarray | None = None,
    use_midas: bool = True,
) -> SeverityCues:
    a = area_ratio(pothole_mask, road_mask)
    i = contour_roughness(pothole_mask)

    if use_midas:
        try:
            d = depth_proxy(image_bgr, pothole_mask)
            depth_source = "midas"
        except Exception:
            # MiDaS unavailable (no torch/GPU, model download failed, etc.) -- fall
            # back to the shadow heuristic rather than hard-failing the whole pipeline.
            d = shadow_darkness_ratio(image_bgr, pothole_mask)
            depth_source = "shadow_heuristic"
    else:
        d = shadow_darkness_ratio(image_bgr, pothole_mask)
        depth_source = "shadow_heuristic"

    return SeverityCues(area_ratio=a, depth=d, irregularity=i, depth_source=depth_source)


def compute_severity(
    image_bgr: np.ndarray,
    pothole_mask: np.ndarray,
    road_mask: np.ndarray | None = None,
    use_midas: bool = True,
    weights: FusionWeights | None = None,
    depth_max: float = 30.0,
    irregularity_max: float = 5.0,
) -> tuple[float, str, SeverityCues]:
    """Returns (S, category, cues) for one pothole instance."""
    cues = compute_severity_cues(image_bgr, pothole_mask, road_mask, use_midas)
    score = severity_score(cues.area_ratio, cues.depth, cues.irregularity, weights, depth_max, irregularity_max)
    return score, severity_category(score), cues
