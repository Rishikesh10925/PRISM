"""Resolution robustness of contour_roughness (audit: the raw-pixel measure scored a
circle 1.12 native vs 1.51 at 4x nearest upscaling, r=3 blobs 1.48, a 1-px spur 2.33)."""

import cv2
import numpy as np
import pytest

from irregularity import (
    MIN_IRREGULARITY_AREA_PX,
    NEUTRAL_IRREGULARITY,
    contour_roughness,
    raw_contour_roughness,
)

CIRCLE_LO, CIRCLE_HI = 1.0, 1.15 * 1.10  # "within ~10% of 1.0..1.15"


def _circle(r: int) -> np.ndarray:
    size = 2 * r + 20
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), r, 1, thickness=-1)
    return mask


def _upscale(mask: np.ndarray, k: int) -> np.ndarray:
    return cv2.resize(mask, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)


def _star(r_outer=60, r_inner=25, points=12, size=200) -> np.ndarray:
    c = size // 2
    pts = []
    for k in range(points):
        r = r_outer if k % 2 == 0 else r_inner
        a = 2 * np.pi * k / points
        pts.append((int(c + r * np.cos(a)), int(c + r * np.sin(a))))
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1)
    return mask


@pytest.mark.parametrize("r", [10, 15, 25, 50, 100, 200])
def test_circle_near_one_across_radii(r):
    assert CIRCLE_LO <= contour_roughness(_circle(r)) <= CIRCLE_HI


@pytest.mark.parametrize("r", [10, 20, 50])
@pytest.mark.parametrize("k", [2, 4])
def test_circle_stable_under_nearest_upscaling(r, k):
    native = contour_roughness(_circle(r))
    upscaled = contour_roughness(_upscale(_circle(r), k))
    assert CIRCLE_LO <= upscaled <= CIRCLE_HI
    assert abs(upscaled - native) / native < 0.05


def test_raw_measure_still_shows_the_resolution_problem():
    # documents why the smoothing exists: the raw measure drifts with upscaling
    assert raw_contour_roughness(_upscale(_circle(10), 4)) - raw_contour_roughness(_circle(10)) > 0.2


def test_star_scores_clearly_higher_and_is_stable():
    star = contour_roughness(_star())
    assert star > 2.5  # vs ~1.0 for a circle
    assert star > 2 * contour_roughness(_circle(60))
    assert abs(contour_roughness(_upscale(_star(), 4)) - star) / star < 0.05


def test_thin_spur_is_treated_as_mask_noise():
    mask = _circle(50)
    mask[59:61, 0:60] = 1  # 2-px spur running out of the circle
    assert raw_contour_roughness(mask) > 1.2
    assert contour_roughness(mask) < 1.15


def test_holes_are_ignored():
    mask = _circle(50)
    mask[50:70, 50:70] = 0  # hole in the middle
    assert contour_roughness(mask) == pytest.approx(contour_roughness(_circle(50)), rel=0.02)


def test_tiny_mask_returns_neutral_value():
    mask = _circle(3)
    assert mask.sum() < MIN_IRREGULARITY_AREA_PX
    assert contour_roughness(mask) == NEUTRAL_IRREGULARITY == 1.0


def test_single_pixel_mask_returns_neutral_value():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5, 5] = 1
    assert contour_roughness(mask) == NEUTRAL_IRREGULARITY


def test_largest_component_is_measured():
    mask = _circle(50)
    mask[0:3, 0:3] = 1  # small separate speck
    assert contour_roughness(mask) == pytest.approx(contour_roughness(_circle(50)), rel=0.01)


def test_min_area_is_configurable():
    assert contour_roughness(_circle(12), min_area_px=10**6) == NEUTRAL_IRREGULARITY
