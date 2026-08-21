from pathlib import Path

import cv2
import numpy as np

import apply_offline_augmentation as aoa


def test_augment_train_split_copies_labels_unchanged(tmp_path, monkeypatch):
    merged_dir = tmp_path / "merged"
    splits_dir = tmp_path / "annotations" / "splits"
    images_dir = merged_dir / "images"
    labels_dir = merged_dir / "labels"
    splits_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)

    for i in range(3):
        img_path = images_dir / f"img{i}.png"
        cv2.imwrite(str(img_path), (np.random.rand(40, 40, 3) * 255).astype(np.uint8))
        (labels_dir / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    (splits_dir / "train.txt").write_text(
        "\n".join(str(images_dir / f"img{i}.png") for i in range(3)), encoding="utf-8"
    )

    monkeypatch.setattr(aoa, "SPLITS_DIR", splits_dir)
    monkeypatch.setattr(aoa, "MERGED_DIR", merged_dir)
    monkeypatch.setattr(aoa, "AUG_IMAGES_DIR", images_dir)
    monkeypatch.setattr(aoa, "AUG_LABELS_DIR", labels_dir)

    combined = aoa.augment_train_split(copies_per_image=2, seed=0)

    assert len(combined) == 3 + 3 * 2  # 3 originals + 2 augmented copies each
    aug_files = sorted(images_dir.glob("*_aug*.png"))
    assert len(aug_files) == 6  # augmented copies land alongside the originals

    for aug_file in aug_files:
        label_file = labels_dir / f"{aug_file.stem}.txt"
        assert label_file.exists()
        assert label_file.read_text(encoding="utf-8") == "0 0.5 0.5 0.2 0.2\n"

    manifest = (splits_dir / "train_augmented.txt").read_text(encoding="utf-8").splitlines()
    assert len(manifest) == 9
