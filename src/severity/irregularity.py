"""Structural irregularity severity cue (Phase 4 Task 4): contour roughness
i = perimeter^2 / (4*pi*area).

A perfect circle scores 1.0 (the minimum possible value by the isoperimetric
inequality); a more jagged, irregular edge -- indicating structural degradation,
crack propagation, or crumbling around the pothole -- scores progressively higher
(blueprint Section 5.2, cue `i`).

Resolution robustness (revised): measured on the raw pixel contour, i depends on mask
resolution rather than shape -- the pixel staircase is counted as perimeter, so a filled
circle scored 1.12 at native size but 1.51 after 4x nearest-neighbour upscaling, a r=3 px
blob scored 1.48, and a 1-px spur added ~1.2. contour_roughness now:

  1. keeps only the largest connected component and fills its holes (external contour
     only -- holes are mask noise, not boundary roughness);
  2. rescales that component to a canonical size (equivalent-circle radius
     CANONICAL_RADIUS_PX) with area-averaging / bilinear interpolation, so the same shape
     is measured at the same resolution whatever its pixel size;
  3. smooths it with a Gaussian of sigma SMOOTHING_SIGMA_FRACTION * CANONICAL_RADIUS_PX and
     re-thresholds at 0.5, which removes pixel staircases and sub-sigma spurs while
     keeping shape features larger than a few percent of the pothole's radius;
  4. measures perimeter on a cv2.approxPolyDP polygon (epsilon POLY_EPSILON_PX at
     canonical scale) and area on the same polygon.

Masks whose foreground is smaller than MIN_IRREGULARITY_AREA_PX pixels are too small for
any boundary-shape statement to be meaningful and return NEUTRAL_IRREGULARITY (1.0, the
circle value, which normalizes to a zero contribution in fusion.py). An empty mask returns
0.0 (no pothole; also a zero contribution).

Range: i >= 1.0 for any measurable mask (unbounded above). Synthetic reference values
with the defaults below: filled circle 1.00-1.10 for r = 10..200 px, native or 2x/4x
nearest-upscaled; 12-point star (radii 60/25 px) ~3.4; circle with a 2-px spur ~1.01
(the spur is treated as mask noise); circle with additive radial noise (std 12% of r) ~1.4.
The smoothing trades away edge detail finer than ~5% of the pothole's radius -- that is
deliberate (such detail is dominated by mask/annotation noise), but it means i reflects
boundary shape at the scale of the pothole, not micro-texture of its rim.
Note that elongation alone also raises i (a 4:1 ellipse scores ~1.87, a square 1.27) -- i measures
deviation from a circle, which mixes elongation with edge roughness.
"""

from __future__ import annotations

import cv2
import numpy as np

from mask_utils import binarize_mask

MIN_IRREGULARITY_AREA_PX = 200  # ~ a 14x14 px blob / r~8 px disc; below this, return NEUTRAL_IRREGULARITY
NEUTRAL_IRREGULARITY = 1.0  # circle value -> norm_irregularity() == 0
CANONICAL_RADIUS_PX = 64.0  # equivalent-circle radius the component is rescaled to before measuring
SMOOTHING_SIGMA_FRACTION = 0.05  # Gaussian sigma as a fraction of CANONICAL_RADIUS_PX (3.2 px)
POLY_EPSILON_PX = 1.0  # approxPolyDP tolerance at canonical scale (~1.6% of the radius)
_PAD_FRACTION = 0.25


def _largest_component_filled(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=lambda c: (cv2.contourArea(c), len(c)))
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [largest], -1, 1, thickness=cv2.FILLED)
    return filled


def _canonical_component(component: np.ndarray) -> np.ndarray:
    """Crops the component, rescales it so its equivalent-circle radius is
    CANONICAL_RADIUS_PX, smooths, and re-binarizes. Returns a uint8 {0,1} mask."""
    ys, xs = np.nonzero(component)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = component[y0:y1, x0:x1].astype(np.float32)

    area = float(component.sum())
    scale = CANONICAL_RADIUS_PX / np.sqrt(area / np.pi)
    new_w = max(1, int(round(crop.shape[1] * scale)))
    new_h = max(1, int(round(crop.shape[0] * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)

    pad = int(np.ceil(_PAD_FRACTION * CANONICAL_RADIUS_PX)) + 2
    resized = cv2.copyMakeBorder(resized, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0.0)
    sigma = SMOOTHING_SIGMA_FRACTION * CANONICAL_RADIUS_PX
    smoothed = cv2.GaussianBlur(resized, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return (smoothed > 0.5).astype(np.uint8)


def raw_contour_roughness(pothole_mask: np.ndarray) -> float:
    """The original, resolution-dependent measure on the raw pixel contour of the largest
    component (kept for comparison/ablation only; contour_roughness is what the pipeline
    uses). Returns 0.0 for an empty/degenerate mask."""
    mask = binarize_mask(pothole_mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, closed=True)
    if area <= 0 or perimeter <= 0:
        return 0.0
    return float((perimeter**2) / (4 * np.pi * area))


def contour_roughness(pothole_mask: np.ndarray, min_area_px: int = MIN_IRREGULARITY_AREA_PX) -> float:
    """Returns the resolution-normalized roughness i of the largest connected component:
      * 0.0 for an empty mask (no pothole);
      * NEUTRAL_IRREGULARITY (1.0) when that component has fewer than `min_area_px`
        foreground pixels (too small to measure shape reliably);
      * otherwise i >= 1.0 (circle ~1.0-1.05, independent of radius and of nearest-neighbour
        upscaling to within a few percent).
    Accepts 0/1, 0/255, bool or float masks (see mask_utils.binarize_mask)."""
    mask = binarize_mask(pothole_mask)
    if mask.sum() == 0:
        return 0.0

    component = _largest_component_filled(mask)
    if component is None or component.sum() < min_area_px:
        return NEUTRAL_IRREGULARITY

    canonical = _canonical_component(component)
    contours, _ = cv2.findContours(canonical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return NEUTRAL_IRREGULARITY
    largest = max(contours, key=cv2.contourArea)
    poly = cv2.approxPolyDP(largest, POLY_EPSILON_PX, closed=True)
    area = cv2.contourArea(poly)
    perimeter = cv2.arcLength(poly, closed=True)
    if area <= 0 or perimeter <= 0:
        return NEUTRAL_IRREGULARITY

    # Isoperimetric inequality guarantees >= 1 for exact shapes; clamp float noise.
    return float(max(1.0, (perimeter**2) / (4 * np.pi * area)))
