"""Shadow-darkness depth fallback (Phase 4 Task 3): a lightweight, model-free stand-in
for depth_proxy.py's MiDaS-based estimate.

Rationale: a pothole's interior walls and base are typically shadowed/darker than the
flat surrounding road surface, especially in daylight conditions -- so darkness ratio is
a cheap, dependency-free proxy for depth. It is *not* meant to replace MiDaS; it exists
as (a) a fallback when a GPU/MiDaS isn't available, and (b) a sanity check that MiDaS's
differential and this heuristic agree in direction (see agreement() below), per the
blueprint's "validated against a manually depth-scored subset" note (Section 5.2).
"""

from __future__ import annotations

import cv2
import numpy as np


def shadow_darkness_ratio(image_bgr: np.ndarray, pothole_mask: np.ndarray, surround_dilation_px: int = 15) -> float:
    """Returns d_shadow: mean surrounding-road brightness minus mean pothole brightness
    (grayscale, 0-255 scale), so a darker (deeper-looking) pothole gives a larger
    positive value -- same sign convention as depth_proxy.depth_differential."""
    mask = (pothole_mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return 0.0

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (surround_dilation_px * 2 + 1,) * 2)
    dilated = cv2.dilate(mask, kernel)
    surround_ring = (dilated > 0) & (mask == 0)

    if surround_ring.sum() == 0:
        return 0.0

    pothole_mean = float(gray[mask > 0].mean())
    surround_mean = float(gray[surround_ring].mean())

    return surround_mean - pothole_mean


def agreement(midas_d: float, shadow_d: float) -> bool:
    """True when the two depth proxies agree on direction (both say "deeper than
    surroundings" or both say "not deeper") -- a quick per-image validation signal
    for flagging cases where MiDaS's estimate looks unreliable (e.g. heavy motion blur,
    unusual lighting) and the shadow heuristic disagrees."""
    return (midas_d > 0) == (shadow_d > 0)
