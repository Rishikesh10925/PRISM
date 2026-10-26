"""Empty / tiny / holed / non-binary / mis-shaped masks through every cue and the pipeline."""

import cv2
import numpy as np
import pytest

from depth_proxy import depth_differential, normalized_depth_differential
from geometric import area_ratio
from irregularity import contour_roughness
from mask_utils import binarize_mask
from pipeline import compute_severity, compute_severity_cues
from shadow_heuristic import shadow_contrast, shadow_darkness_ratio

H, W = 80, 100


def _image():
    image = np.full((H, W, 3), 180, dtype=np.uint8)
    image[40:60, 40:60] = 60
    return image


def _depth():
    return np.repeat(np.linspace(100, 300, H)[:, None], W, axis=1)


def _disc01():
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 12, 1, thickness=-1)
    return mask


def _all_cues(mask):
    image = _image()
    return {
        "area": area_ratio(mask),
        "irregularity": contour_roughness(mask),
        "shadow_raw": shadow_darkness_ratio(image, mask),
        "shadow": shadow_contrast(image, mask),
        "depth_raw": depth_differential(_depth(), mask),
        "depth": normalized_depth_differential(_depth(), mask),
    }


def test_all_zero_mask_every_cue_zero_and_pipeline_zero():
    mask = np.zeros((H, W), dtype=np.uint8)
    assert all(v == 0.0 for v in _all_cues(mask).values())
    score, category, cues = compute_severity(_image(), mask, use_midas=False)
    assert score == 0.0 and category == "Very Low"
    assert cues.depth_source == "none"
    # use_midas=True must not even try to run the model on an empty mask
    assert compute_severity(_image(), mask, use_midas=True)[0] == 0.0


def test_single_pixel_mask_is_finite_and_small():
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[50, 50] = 1
    cues = _all_cues(mask)
    assert all(np.isfinite(v) for v in cues.values())
    assert cues["irregularity"] == 1.0  # neutral
    score, _, _ = compute_severity(_image(), mask, use_midas=False)
    assert 0.0 <= score <= 100.0


def test_mask_with_hole_is_handled():
    mask = _disc01()
    mask[48:52, 48:52] = 0
    cues = _all_cues(mask)
    assert all(np.isfinite(v) for v in cues.values())
    assert cues["irregularity"] == pytest.approx(contour_roughness(_disc01()), rel=0.02)


@pytest.mark.parametrize(
    "convert",
    [
        lambda m: m * 255,  # 0/255 uint8
        lambda m: m.astype(bool),
        lambda m: m.astype(np.float32),  # 0.0/1.0
        lambda m: m.astype(np.float64) * 255.0,  # float 0/255
        lambda m: m.astype(np.float32) * 0.9 + 0.05,  # soft probabilities 0.05/0.95
        lambda m: m[:, :, None],  # H x W x 1
    ],
    ids=["uint8_0_255", "bool", "float_0_1", "float_0_255", "soft_probs", "hw1"],
)
def test_non_binary_masks_match_binary(convert):
    reference = _all_cues(_disc01())
    converted = _all_cues(convert(_disc01()))
    for key, value in reference.items():
        assert converted[key] == pytest.approx(value), key
    ref_score = compute_severity(_image(), _disc01(), use_midas=False)[0]
    assert compute_severity(_image(), convert(_disc01()), use_midas=False)[0] == pytest.approx(ref_score)


def test_soft_mask_thresholded_at_half():
    soft = np.zeros((10, 10), dtype=np.float32)
    soft[2:5, 2:5] = 0.4
    soft[6:8, 6:8] = 0.6
    assert binarize_mask(soft).sum() == 4


def test_nan_in_float_mask_is_background():
    mask = _disc01().astype(np.float32)
    mask[0, 0] = np.nan
    assert binarize_mask(mask).sum() == _disc01().sum()


def test_mismatched_shapes_raise_clear_value_error():
    small = np.zeros((H // 2, W // 2), dtype=np.uint8)
    small[10:20, 10:20] = 1
    with pytest.raises(ValueError, match="does not match"):
        shadow_contrast(_image(), small)
    with pytest.raises(ValueError, match="does not match"):
        shadow_darkness_ratio(_image(), small)
    with pytest.raises(ValueError, match="does not match"):
        depth_differential(_depth(), small)
    with pytest.raises(ValueError, match="does not match"):
        normalized_depth_differential(_depth(), small)
    with pytest.raises(ValueError, match="does not match"):
        area_ratio(_disc01(), road_mask=small)
    with pytest.raises(ValueError, match="does not match"):
        compute_severity(_image(), small, use_midas=False)
    with pytest.raises(ValueError, match="does not match"):
        compute_severity_cues(_image(), small, use_midas=True)  # must raise, not silently fall back


def test_non_2d_mask_raises():
    with pytest.raises(ValueError, match="2-D"):
        contour_roughness(np.zeros((5, 5, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="2-D"):
        area_ratio(np.zeros(10, dtype=np.uint8))


def test_mask_filling_frame_gives_zero_depth_but_full_area():
    mask = np.ones((H, W), dtype=np.uint8)
    cues = _all_cues(mask)
    assert cues["depth"] == 0.0 and cues["shadow"] == 0.0
    assert cues["area"] == 1.0
