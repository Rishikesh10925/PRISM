from pathlib import Path

import cv2
import numpy as np

from convert_pothole600 import convert_pothole600_split


def test_convert_pothole600_split(tmp_path: Path):
    split_dir = tmp_path / "training"
    (split_dir / "rgb").mkdir(parents=True)
    (split_dir / "label").mkdir(parents=True)

    cv2.imwrite(str(split_dir / "rgb" / "0000.png"), np.zeros((40, 60, 3), dtype=np.uint8))
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[10:20, 15:30] = 255
    cv2.imwrite(str(split_dir / "label" / "0000.png"), mask)

    # a second image with an empty (all-black) label -> zero instances, still included
    cv2.imwrite(str(split_dir / "rgb" / "0001.png"), np.zeros((40, 60, 3), dtype=np.uint8))
    cv2.imwrite(str(split_dir / "label" / "0001.png"), np.zeros((40, 60), dtype=np.uint8))

    results = convert_pothole600_split(split_dir)

    assert len(results) == 2
    ann0 = next(a for a in results if a.image_path.endswith("0000.png"))
    assert ann0.width == 60 and ann0.height == 40
    assert len(ann0.instances) == 1
    assert ann0.instances[0].class_name == "pothole"
    assert ann0.instances[0].needs_mask is False

    ann1 = next(a for a in results if a.image_path.endswith("0001.png"))
    assert len(ann1.instances) == 0
