import cv2
import numpy as np

from irregularity import contour_roughness


def test_circle_scores_close_to_one():
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 50, 1, thickness=-1)

    i = contour_roughness(mask)

    assert 0.95 <= i <= 1.15  # discretized circle isn't perfectly smooth, allow slack


def test_jagged_star_scores_higher_than_circle():
    circle_mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(circle_mask, (100, 100), 50, 1, thickness=-1)

    # a star-ish jagged polygon with the same rough footprint
    star_points = []
    for k in range(12):
        angle = k * np.pi / 6
        radius = 60 if k % 2 == 0 else 25
        star_points.append((int(100 + radius * np.cos(angle)), int(100 + radius * np.sin(angle))))
    star_mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.fillPoly(star_mask, [np.array(star_points, dtype=np.int32)], 1)

    assert contour_roughness(star_mask) > contour_roughness(circle_mask)


def test_empty_mask_returns_zero():
    mask = np.zeros((50, 50), dtype=np.uint8)
    assert contour_roughness(mask) == 0.0
