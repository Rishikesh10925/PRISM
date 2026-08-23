"""Structural irregularity severity cue (Phase 4 Task 4): contour roughness
i = perimeter^2 / (4*pi*area).

A perfect circle scores 1.0 (the minimum possible value by the isoperimetric
inequality); a more jagged, irregular edge -- indicating structural degradation,
crack propagation, or crumbling around the pothole -- scores progressively higher
(blueprint Section 5.2, cue `i`).
"""

from __future__ import annotations

import cv2
import numpy as np


def contour_roughness(pothole_mask: np.ndarray) -> float:
    """Returns i >= 1.0 for the largest contour in the mask, or 0.0 if the mask is
    empty/degenerate (no contour, or zero area/perimeter)."""
    mask = (pothole_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, closed=True)

    if area <= 0 or perimeter <= 0:
        return 0.0

    return float((perimeter**2) / (4 * np.pi * area))
