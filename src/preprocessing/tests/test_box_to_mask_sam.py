from pathlib import Path

import cv2
import numpy as np
import pytest

from box_to_mask_sam import convert_boxes_to_masks
from schema import ImageAnnotation, Instance, box_to_polygon

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_IMAGE = REPO_ROOT / "data" / "merged" / "images" / "pothole600_testing_0000.png"
SAMPLE_LABEL = REPO_ROOT / "data" / "merged" / "labels" / "pothole600_testing_0000.txt"


def _real_mask_and_box(image_shape) -> tuple[np.ndarray, list[float]]:
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for line in SAMPLE_LABEL.read_text(encoding="utf-8").splitlines():
        parts = [float(v) for v in line.split()[1:]]
        pts = np.array([(parts[i] * width, parts[i + 1] * height) for i in range(0, len(parts), 2)], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
    ys, xs = np.where(mask > 0)
    box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    return mask, box


@pytest.mark.skipif(not SAMPLE_IMAGE.exists(), reason="real Pothole-600 sample not present in data/merged/")
def test_convert_boxes_to_masks_recovers_a_reasonable_mask_from_a_real_box(tmp_path):
    image = cv2.imread(str(SAMPLE_IMAGE))
    assert image is not None
    ground_truth_mask, box = _real_mask_and_box(image.shape)

    ann = ImageAnnotation(
        image_path=str(SAMPLE_IMAGE),
        width=image.shape[1],
        height=image.shape[0],
        source="test",
        instances=[Instance(class_name="pothole", polygon=box_to_polygon(*box), needs_mask=True)],
    )

    results = convert_boxes_to_masks([ann], manifest_csv=tmp_path / "manifest.csv")

    assert len(results) == 1
    inst = results[0].instances[0]
    assert inst.needs_mask is False
    assert len(inst.polygon) >= 3

    sam_mask = np.zeros_like(ground_truth_mask)
    pts = np.array(inst.polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(sam_mask, [pts], 1)

    intersection = np.logical_and(sam_mask, ground_truth_mask).sum()
    union = np.logical_or(sam_mask, ground_truth_mask).sum()
    iou = intersection / union
    assert iou > 0.3  # SAM prompted with the real box should land roughly on the real pothole

    assert (tmp_path / "manifest.csv").exists()


@pytest.mark.skipif(not SAMPLE_IMAGE.exists(), reason="real Pothole-600 sample not present in data/merged/")
def test_convert_boxes_to_masks_batches_multiple_boxes_per_image(tmp_path):
    image = cv2.imread(str(SAMPLE_IMAGE))
    assert image is not None
    _, box = _real_mask_and_box(image.shape)

    # two box-only instances on the SAME image -- exercises the batched sam() call
    # (one image encoder pass, two mask prompts) rather than the single-box path
    second_box = [box[0] + 5, box[1] + 5, min(box[2] + 5, image.shape[1] - 1), min(box[3] + 5, image.shape[0] - 1)]
    ann = ImageAnnotation(
        image_path=str(SAMPLE_IMAGE),
        width=image.shape[1],
        height=image.shape[0],
        source="test",
        instances=[
            Instance(class_name="pothole", polygon=box_to_polygon(*box), needs_mask=True),
            Instance(class_name="pothole", polygon=box_to_polygon(*second_box), needs_mask=True),
        ],
    )

    results = convert_boxes_to_masks([ann], manifest_csv=tmp_path / "manifest.csv")

    assert len(results[0].instances) == 2
    assert all(not inst.needs_mask for inst in results[0].instances)
    assert all(len(inst.polygon) >= 3 for inst in results[0].instances)


def test_convert_boxes_to_masks_leaves_real_masks_untouched(tmp_path):
    ann = ImageAnnotation(
        image_path="doesnt_matter.png",
        width=10,
        height=10,
        source="test",
        instances=[Instance(class_name="pothole", polygon=[(1, 1), (5, 1), (5, 5), (1, 5)], needs_mask=False)],
    )

    # image_path doesn't exist -> cv2.imread returns None -> annotation passed through unchanged
    results = convert_boxes_to_masks([ann])

    assert results[0].instances[0].needs_mask is False
    assert results[0].instances[0].polygon == [(1, 1), (5, 1), (5, 5), (1, 5)]
