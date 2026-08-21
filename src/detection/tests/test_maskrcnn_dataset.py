from pathlib import Path

import numpy as np
from PIL import Image

from maskrcnn_dataset import PotholeSegDataset


def test_dataset_returns_boxes_labels_masks(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    img_path = images_dir / "a.png"
    Image.fromarray(np.zeros((50, 100, 3), dtype=np.uint8)).save(img_path)
    # class 0, a rectangle covering roughly x:[0.1,0.3] y:[0.2,0.6] normalized
    (labels_dir / "a.txt").write_text(
        "0 0.1 0.2 0.3 0.2 0.3 0.6 0.1 0.6\n", encoding="utf-8"
    )

    manifest = tmp_path / "manifest.txt"
    manifest.write_text(str(img_path), encoding="utf-8")

    dataset = PotholeSegDataset(manifest, labels_dir)
    assert len(dataset) == 1

    image_t, target = dataset[0]
    assert image_t.shape == (3, 50, 100)
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].tolist() == [1]  # class 0 -> torchvision label 1
    assert target["masks"].shape == (1, 50, 100)
    assert target["masks"].sum().item() > 0


def test_dataset_handles_image_with_no_annotations(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    img_path = images_dir / "empty.png"
    Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(img_path)
    (labels_dir / "empty.txt").write_text("", encoding="utf-8")

    manifest = tmp_path / "manifest.txt"
    manifest.write_text(str(img_path), encoding="utf-8")

    dataset = PotholeSegDataset(manifest, labels_dir)
    _, target = dataset[0]

    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)
    assert target["masks"].shape == (0, 20, 20)
