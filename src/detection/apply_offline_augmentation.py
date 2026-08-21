"""Expand the training split with weather/blur-augmented copies (offline augmentation).

All transforms in augmentations.build_train_augmentations are photometric/blur only
(no crop, flip, or warp), so a copy's polygon labels are identical to the original's —
augmented copies just get their label .txt hard-linked/copied unchanged.

This is deliberately offline (write augmented images to disk once) rather than an
on-the-fly Ultralytics hook: it's simple to test, inspect, and — for the Phase 6
augmentation ablation — trivial to toggle by pointing training at train.txt vs
train_augmented.txt.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2

from augmentations import build_train_augmentations

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS_DIR = REPO_ROOT / "data" / "annotations"
SPLITS_DIR = ANNOTATIONS_DIR / "splits"
AUG_IMAGES_DIR = REPO_ROOT / "data" / "merged" / "images_aug"


def augment_train_split(copies_per_image: int = 1, seed: int = 0) -> list[str]:
    train_manifest = SPLITS_DIR / "train.txt"
    if not train_manifest.exists():
        raise SystemExit(f"{train_manifest} not found — run prepare_splits.py first")

    original_paths = [Path(p) for p in train_manifest.read_text(encoding="utf-8").splitlines() if p.strip()]
    AUG_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = build_train_augmentations()
    augmented_paths: list[str] = []

    for img_path in original_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        label_path = ANNOTATIONS_DIR / "labels" / f"{img_path.stem}.txt"

        for copy_idx in range(copies_per_image):
            out = pipeline(image=img)["image"]
            aug_name = f"{img_path.stem}_aug{copy_idx}{img_path.suffix}"
            aug_path = AUG_IMAGES_DIR / aug_name
            cv2.imwrite(str(aug_path), out)

            if label_path.exists():
                shutil.copy2(label_path, ANNOTATIONS_DIR / "labels" / f"{aug_name.rsplit('.', 1)[0]}.txt")

            augmented_paths.append(str(aug_path))

    combined = [str(p) for p in original_paths] + augmented_paths
    (SPLITS_DIR / "train_augmented.txt").write_text("\n".join(combined), encoding="utf-8")

    print(
        f"[apply_offline_augmentation] {len(original_paths)} original + "
        f"{len(augmented_paths)} augmented = {len(combined)} train images "
        f"-> {SPLITS_DIR / 'train_augmented.txt'}"
    )
    return combined


if __name__ == "__main__":
    augment_train_split()
