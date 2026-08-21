from pathlib import Path

import cv2
import numpy as np

import apply_offline_augmentation as aoa


def test_augment_train_split_copies_labels_unchanged(tmp_path, monkeypatch):
    annotations_dir = tmp_path / "annotations"
    splits_dir = annotations_dir / "splits"
    labels_dir = annotations_dir / "labels"
    images_dir = tmp_path / "images"
    aug_dir = tmp_path / "images_aug"
    splits_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)

    for i in range(3):
        img_path = images_dir / f"img{i}.png"
        cv2.imwrite(str(img_path), (np.random.rand(40, 40, 3) * 255).astype(np.uint8))
        (labels_dir / f"img{i}.txt").write_text(f"0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    (splits_dir / "train.txt").write_text(
        "\n".join(str(images_dir / f"img{i}.png") for i in range(3)), encoding="utf-8"
    )

    monkeypatch.setattr(aoa, "ANNOTATIONS_DIR", annotations_dir)
    monkeypatch.setattr(aoa, "SPLITS_DIR", splits_dir)
    monkeypatch.setattr(aoa, "AUG_IMAGES_DIR", aug_dir)

    combined = aoa.augment_train_split(copies_per_image=2, seed=0)

    assert len(combined) == 3 + 3 * 2  # 3 originals + 2 augmented copies each
    aug_files = list(aug_dir.glob("*.png"))
    assert len(aug_files) == 6

    for aug_file in aug_files:
        label_file = labels_dir / f"{aug_file.stem}.txt"
        assert label_file.exists()
        assert label_file.read_text(encoding="utf-8") == "0 0.5 0.5 0.2 0.2\n"

    manifest = (splits_dir / "train_augmented.txt").read_text(encoding="utf-8").splitlines()
    assert len(manifest) == 9
