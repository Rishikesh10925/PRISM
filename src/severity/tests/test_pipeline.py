from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline import compute_severity

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_IMAGE = REPO_ROOT / "data" / "merged" / "images" / "pothole600_testing_0000.png"
SAMPLE_LABEL = REPO_ROOT / "data" / "merged" / "labels" / "pothole600_testing_0000.txt"


def _mask_from_label(label_path: Path, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = [float(v) for v in line.split()[1:]]
        pts = np.array([(parts[i] * width, parts[i + 1] * height) for i in range(0, len(parts), 2)], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def test_compute_severity_with_shadow_fallback_no_midas():
    image = np.full((100, 100, 3), 200, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 15, 1, thickness=-1)
    image[mask > 0] = 30  # dark pothole

    score, category, cues = compute_severity(image, mask, use_midas=False)

    assert 0.0 <= score <= 100.0
    assert category in ("Very Low", "Low", "Medium", "High", "Critical")
    assert cues.depth_source == "shadow_heuristic"
    assert cues.depth > 0  # darker interior -> positive depth cue


@pytest.mark.skipif(not SAMPLE_IMAGE.exists(), reason="real Pothole-600 sample not present in data/merged/")
def test_compute_severity_end_to_end_on_real_image_with_midas():
    image = cv2.imread(str(SAMPLE_IMAGE))
    assert image is not None
    mask = _mask_from_label(SAMPLE_LABEL, *image.shape[:2])

    score, category, cues = compute_severity(image, mask, use_midas=True)

    assert 0.0 <= score <= 100.0
    assert category in ("Very Low", "Low", "Medium", "High", "Critical")
    assert cues.depth_source == "midas"
    assert 0.0 <= cues.area_ratio <= 1.0
    assert cues.irregularity >= 1.0
