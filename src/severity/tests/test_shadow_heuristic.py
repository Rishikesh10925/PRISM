import numpy as np

from shadow_heuristic import agreement, shadow_darkness_ratio


def test_shadow_darkness_ratio_darker_pothole_gives_positive_value():
    image = np.full((50, 50, 3), 200, dtype=np.uint8)  # bright road
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[20:30, 20:30] = 1
    image[mask > 0] = 40  # dark pothole interior

    d = shadow_darkness_ratio(image, mask, surround_dilation_px=5)

    assert d > 0
    assert d == 200 - 40


def test_shadow_darkness_ratio_empty_mask_returns_zero():
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    mask = np.zeros((30, 30), dtype=np.uint8)

    assert shadow_darkness_ratio(image, mask) == 0.0


def test_agreement_same_sign_true():
    assert agreement(midas_d=5.0, shadow_d=12.0) is True
    assert agreement(midas_d=-2.0, shadow_d=-0.5) is True


def test_agreement_opposite_sign_false():
    assert agreement(midas_d=5.0, shadow_d=-1.0) is False
