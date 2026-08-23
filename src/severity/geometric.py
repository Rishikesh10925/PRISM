"""Geometric severity cue (Phase 4 Task 1): area ratio a = pothole mask area / estimated
road-surface area in frame.

Normalizing by road-surface area (rather than raw pixel count) accounts for camera
distance -- the same physical pothole covers fewer pixels the farther the camera is
from it, and so does the road around it, so the *ratio* stays roughly stable while the
raw pixel count doesn't (blueprint Section 5.2, cue `a`).
"""

from __future__ import annotations

import numpy as np


def road_surface_area_from_mask(road_mask: np.ndarray) -> float:
    """Real road-surface area from a road_surface segmentation mask, when one exists."""
    return float((road_mask > 0).sum())


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
    """Returns a in [0, 1] (clipped) -- pothole pixels over estimated road-surface pixels."""
    pothole_area = float((pothole_mask > 0).sum())
    height, width = pothole_mask.shape[:2]

    road_area = (
        road_surface_area_from_mask(road_mask) if road_mask is not None else road_surface_area_heuristic(height, width, road_fraction)
    )
    if road_area <= 0:
        return 0.0

    return float(np.clip(pothole_area / road_area, 0.0, 1.0))
