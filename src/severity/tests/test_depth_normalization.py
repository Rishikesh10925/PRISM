"""Scale-normalized MiDaS depth differential, on synthetic inverse-depth maps (no model)."""

import numpy as np
import pytest

from depth_proxy import depth_differential, depth_scale, normalized_depth_differential


def _scene(h=120, w=100):
    """Perspective-like inverse depth (larger = closer, increasing toward the bottom of
    the frame) with a pothole whose floor reads FARTHER (lower inverse depth)."""
    rows = np.linspace(100.0, 500.0, h, dtype=np.float64)[:, None]
    depth = np.repeat(rows, w, axis=1)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[80:100, 40:60] = 1
    depth[mask > 0] -= 40.0
    return depth, mask


def test_pothole_farther_than_ring_gives_positive_d():
    depth, mask = _scene()
    assert depth_differential(depth, mask, 5) > 0
    assert normalized_depth_differential(depth, mask, surround_dilation_px=5) > 0


def test_pothole_closer_than_ring_gives_negative_d():
    depth, mask = _scene()
    depth[mask > 0] += 80.0  # now a bump, not a pothole
    assert normalized_depth_differential(depth, mask, surround_dilation_px=5) < 0


def test_scaling_depth_map_does_not_change_normalized_d():
    depth, mask = _scene()
    base = normalized_depth_differential(depth, mask, surround_dilation_px=5)
    assert normalized_depth_differential(depth * 10.0, mask, surround_dilation_px=5) == pytest.approx(base, rel=1e-9)
    assert normalized_depth_differential(depth * 0.01, mask, surround_dilation_px=5) == pytest.approx(base, rel=1e-9)
    # the raw differential is NOT scale-invariant -- the reason for the normalized variant
    assert depth_differential(depth * 10.0, mask, 5) == pytest.approx(10 * depth_differential(depth, mask, 5))


def test_shifting_depth_map_does_not_change_normalized_d():
    depth, mask = _scene()
    base = normalized_depth_differential(depth, mask, surround_dilation_px=5)
    assert normalized_depth_differential(depth + 1234.0, mask, surround_dilation_px=5) == pytest.approx(base, rel=1e-9)


def test_scale_uses_road_mask_when_given():
    depth, mask = _scene()
    road = np.zeros_like(mask)
    road[90:, :] = 1
    assert depth_scale(depth, mask, road_mask=road) < depth_scale(depth, mask)  # narrower band of rows


def test_flat_depth_map_falls_back_to_median_scale():
    depth = np.full((50, 50), 10.0)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[30:40, 20:30] = 1
    depth[mask > 0] = 4.0
    assert normalized_depth_differential(depth, mask, surround_dilation_px=5) == pytest.approx(0.6)
    assert normalized_depth_differential(depth * 7, mask, surround_dilation_px=5) == pytest.approx(0.6)


def test_all_zero_depth_map_gives_zero():
    depth = np.zeros((40, 40))
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    assert normalized_depth_differential(depth, mask) == 0.0


def test_nan_depth_values_are_ignored():
    depth, mask = _scene()
    depth[0:5, :] = np.nan
    depth[85, 45] = np.nan
    assert np.isfinite(normalized_depth_differential(depth, mask, surround_dilation_px=5))
