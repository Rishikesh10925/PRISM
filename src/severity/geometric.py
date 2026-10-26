"""Geometric severity cue (Phase 4 Task 1): area ratio a = pothole mask area / estimated
road-surface area in frame.

Normalizing by road-surface area (rather than raw pixel count) accounts for camera
distance -- the same physical pothole covers fewer pixels the farther the camera is
from it, and so does the road around it, so the *ratio* stays roughly stable while the
raw pixel count doesn't (blueprint Section 5.2, cue `a`).
"""

from __future__ import annotations

import numpy as np

from mask_utils import binarize_mask, check_same_shape


def road_surface_area_from_mask(road_mask: np.ndarray) -> float:
    """Real road-surface area from a road_surface segmentation mask, when one exists."""
    return float(binarize_mask(road_mask, "road_mask").sum())


def road_surface_area_heuristic(image_height: int, image_width: int, road_fraction: float = 0.6) -> float:
    """Fallback when no road_surface mask is available (true today for every source we
    have -- see docs/phase4/01_severity_modules.md): approximates the road as the
    bottom `road_fraction` of the frame, which is a reasonable default for a forward- or
    downward-facing vehicle/phone camera where the upper frame is mostly sky/surroundings
    rather than road surface. This is a coarse proxy, not a real road-plane estimate --
    documented as a limitation, matching the blueprint's "simple road-plane heuristic"
    fallback option (Section 5.2)."""
    return float(image_height * road_fraction * image_width)


def area_ratio(pothole_mask: np.ndarray, road_mask: np.ndarray | None = None, road_fraction: float = 0.6) -> float:
    """Returns a in [0, 1] (clipped) -- pothole pixels over estimated road-surface pixels.

    Range/normalization: a is already a ratio, so fusion.norm_area only clips it to [0, 1]
    (SeverityConfig.area_max can rescale it). Empty mask -> 0.0. Accepts 0/1, 0/255, bool
    or float masks; raises ValueError if road_mask's shape differs from pothole_mask's.
    Limitation: without a road_mask the denominator is the bottom-`road_fraction` heuristic,
    which ignores the actual road extent, perspective and occlusions."""
    if not 0.0 < road_fraction <= 1.0:
        raise ValueError(f"road_fraction must be in (0, 1], got {road_fraction}")
    mask = binarize_mask(pothole_mask)
    pothole_area = float(mask.sum())
    height, width = mask.shape[:2]
    if road_mask is not None:
        check_same_shape(binarize_mask(road_mask, "road_mask"), (height, width), "road_mask", "pothole_mask")

    road_area = (
        road_surface_area_from_mask(road_mask) if road_mask is not None else road_surface_area_heuristic(height, width, road_fraction)
    )
    if road_area <= 0:
        return 0.0

    return float(np.clip(pothole_area / road_area, 0.0, 1.0))
