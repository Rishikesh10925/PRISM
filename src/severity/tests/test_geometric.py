import numpy as np

from geometric import area_ratio, road_surface_area_heuristic


def test_area_ratio_with_explicit_road_mask():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 1  # 100 px pothole

    road_mask = np.zeros((100, 100), dtype=np.uint8)
    road_mask[:, :] = 1  # full frame is road, 10000 px

    assert area_ratio(mask, road_mask) == 0.01


def test_area_ratio_falls_back_to_heuristic_without_road_mask():
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[0:10, 0:10] = 1  # 100 px pothole

    expected_road_area = road_surface_area_heuristic(100, 200, road_fraction=0.6)
    result = area_ratio(mask, road_mask=None, road_fraction=0.6)

    assert result == 100 / expected_road_area


def test_area_ratio_is_clipped_to_one():
    mask = np.ones((10, 10), dtype=np.uint8)
    tiny_road_mask = np.zeros((10, 10), dtype=np.uint8)
    tiny_road_mask[0, 0] = 1  # road smaller than the pothole mask -- pathological input

    assert area_ratio(mask, tiny_road_mask) == 1.0


def test_area_ratio_zero_pothole():
    mask = np.zeros((50, 50), dtype=np.uint8)
    road_mask = np.ones((50, 50), dtype=np.uint8)

    assert area_ratio(mask, road_mask) == 0.0
