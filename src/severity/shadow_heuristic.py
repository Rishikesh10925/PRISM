"""Shadow-darkness depth fallback (Phase 4 Task 3): a lightweight, model-free stand-in
for depth_proxy.py's MiDaS-based estimate.

Rationale: a pothole's interior walls and base are typically shadowed/darker than the
flat surrounding road surface, especially in daylight conditions -- so darkness is a
cheap, dependency-free proxy for depth. It is *not* meant to replace MiDaS; it exists
as (a) a fallback when a GPU/MiDaS isn't available, and (b) a sanity check that MiDaS's
differential and this heuristic agree in direction (see agreement() below), per the
blueprint's "validated against a manually depth-scored subset" note (Section 5.2).

Two variants
------------
* shadow_darkness_ratio -- original raw value: ring mean minus pothole mean gray level on
  the 0-255 scale. Depends on exposure: the same pothole photographed half as bright gives
  half the value.
* shadow_contrast -- normalized (Weber) contrast: (ring_mean - pothole_mean) / ring_mean.
  Dimensionless, invariant to a global brightness/exposure gain, range (-inf, 1]
  (1 = pothole pitch black against any lit ring; 0 = same brightness; negative = pothole
  brighter, e.g. a water reflection). This is what the pipeline uses when MiDaS is off or
  unavailable. Returns 0.0 if the ring is essentially black (mean < 1 gray level) since
  contrast is undefined there.

Darkness is NOT always depth. Known confounders: wet patches and puddles (dark without
depth, or bright from sky reflections when water-filled), shadows cast by vehicles, poles
or trees across the pothole, fresh dark asphalt patches (repaired potholes read as
"deep"), oil stains, and low-sun / night lighting. Treat this cue as weak evidence.
"""

from __future__ import annotations

import numpy as np

from mask_utils import binarize_mask, check_same_shape, surround_ring, to_gray

_MIN_RING_BRIGHTNESS = 1.0


def _means(image_bgr: np.ndarray, pothole_mask: np.ndarray, surround_dilation_px: int) -> tuple[float, float] | None:
    mask = binarize_mask(pothole_mask)
    gray = to_gray(image_bgr)
    check_same_shape(mask, gray.shape, "pothole_mask", "image_bgr")
    if mask.sum() == 0:
        return None
    ring = surround_ring(mask, surround_dilation_px)
    if ring.sum() == 0:
        return None
    return float(gray[ring].mean()), float(gray[mask > 0].mean())


def shadow_darkness_ratio(image_bgr: np.ndarray, pothole_mask: np.ndarray, surround_dilation_px: int = 15) -> float:
    """RAW d_shadow: mean surrounding-road brightness minus mean pothole brightness
    (grayscale, 0-255 scale for uint8 images), so a darker (deeper-looking) pothole gives a
    larger positive value -- same sign convention as depth_proxy.depth_differential.
    Returns 0.0 for an empty mask or when there is no surrounding ring."""
    means = _means(image_bgr, pothole_mask, surround_dilation_px)
    if means is None:
        return 0.0
    surround_mean, pothole_mean = means
    return surround_mean - pothole_mean


def shadow_contrast(image_bgr: np.ndarray, pothole_mask: np.ndarray, surround_dilation_px: int = 15) -> float:
    """Normalized d_shadow = (ring_mean - pothole_mean) / ring_mean (Weber contrast), in
    (-inf, 1]. Invariant to a global brightness gain; positive = pothole darker than its
    ring. Returns 0.0 for an empty mask, no ring, or a black ring."""
    means = _means(image_bgr, pothole_mask, surround_dilation_px)
    if means is None:
        return 0.0
    surround_mean, pothole_mean = means
    if surround_mean < _MIN_RING_BRIGHTNESS:
        return 0.0
    return float((surround_mean - pothole_mean) / surround_mean)


def agreement(midas_d: float, shadow_d: float) -> bool:
    """True when the two depth proxies agree on direction (both say "deeper than
    surroundings" or both say "not deeper") -- a quick per-image validation signal
    for flagging cases where MiDaS's estimate looks unreliable (e.g. heavy motion blur,
    unusual lighting) and the shadow heuristic disagrees. Works on raw or normalized
    values (only the sign is used)."""
    return (midas_d > 0) == (shadow_d > 0)
