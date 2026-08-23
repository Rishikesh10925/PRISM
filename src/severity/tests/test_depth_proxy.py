from pathlib import Path

import cv2
import numpy as np
import pytest

from depth_proxy import depth_differential, depth_proxy

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_IMAGE = REPO_ROOT / "data" / "merged" / "images" / "pothole600_testing_0000.png"
SAMPLE_LABEL = REPO_ROOT / "data" / "merged" / "labels" / "pothole600_testing_0000.txt"


def test_depth_differential_synthetic_deeper_reads_larger_d():
    # pothole pixels have LOWER inverse-depth (farther away) than their surroundings
    depth_map = np.full((50, 50), 10.0, dtype=np.float32)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[20:30, 20:30] = 1
    depth_map[mask > 0] = 4.0  # pothole reads as farther away -> lower inverse-depth

    d = depth_differential(depth_map, mask, surround_dilation_px=5)

    assert d == pytest.approx(6.0, abs=0.5)  # surround(10) - pothole(4)


def test_depth_differential_empty_mask_returns_zero():
    depth_map = np.random.rand(30, 30).astype(np.float32)
    mask = np.zeros((30, 30), dtype=np.uint8)

    assert depth_differential(depth_map, mask) == 0.0


def test_depth_differential_mask_fills_frame_returns_zero():
    depth_map = np.random.rand(20, 20).astype(np.float32)
    mask = np.ones((20, 20), dtype=np.uint8)  # no surrounding road pixels at all

    assert depth_differential(depth_map, mask) == 0.0


@pytest.mark.skipif(not SAMPLE_IMAGE.exists(), reason="real Pothole-600 sample not present in data/merged/")
def test_depth_proxy_runs_midas_on_a_real_pothole_image():
    image = cv2.imread(str(SAMPLE_IMAGE))
    assert image is not None

    # build a real polygon mask from the actual YOLO-seg label for this image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    height, width = image.shape[:2]
    for line in SAMPLE_LABEL.read_text(encoding="utf-8").splitlines():
        parts = [float(v) for v in line.split()[1:]]
        pts = np.array(
            [(parts[i] * width, parts[i + 1] * height) for i in range(0, len(parts), 2)], dtype=np.int32
        )
        cv2.fillPoly(mask, [pts], 1)

    d = depth_proxy(image, mask)

    assert isinstance(d, float)
    assert not np.isnan(d)
