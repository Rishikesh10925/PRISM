import numpy as np
import pytest

from shadow_heuristic import shadow_contrast, shadow_darkness_ratio


def _scene(road=200, pothole=80):
    image = np.full((60, 60, 3), road, dtype=np.uint8)
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[25:35, 25:35] = 1
    image[mask > 0] = pothole
    return image, mask


def test_darker_pothole_positive_contrast():
    image, mask = _scene()
    assert shadow_contrast(image, mask, 5) == pytest.approx((200 - 80) / 200)


def test_uniformly_darker_or_brighter_image_gives_same_contrast():
    image, mask = _scene(200, 80)
    darker = (image.astype(np.float32) * 0.5).round().astype(np.uint8)  # 100 / 40
    brighter, _ = _scene(250, 100)  # x1.25
    base = shadow_contrast(image, mask, 5)
    assert shadow_contrast(darker, mask, 5) == pytest.approx(base, abs=1e-6)
    assert shadow_contrast(brighter, mask, 5) == pytest.approx(base, abs=1e-6)
    # the raw value changes with exposure -- the reason for normalizing
    assert shadow_darkness_ratio(darker, mask, 5) != shadow_darkness_ratio(image, mask, 5)


def test_brighter_pothole_negative_contrast():
    image, mask = _scene(100, 180)  # e.g. a water reflection
    assert shadow_contrast(image, mask, 5) < 0


def test_black_ring_gives_zero():
    image, mask = _scene(0, 0)
    assert shadow_contrast(image, mask, 5) == 0.0


def test_grayscale_image_accepted():
    image, mask = _scene()
    assert shadow_contrast(image[:, :, 0], mask, 5) == pytest.approx(shadow_contrast(image, mask, 5))


def test_contrast_bounded_above_by_one():
    image, mask = _scene(200, 0)
    assert shadow_contrast(image, mask, 5) == pytest.approx(1.0)
