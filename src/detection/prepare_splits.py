"""Build train/val/test manifest files (lists of image paths) from data/merged/.

Plan target: 70/15/15, stratified by source dataset to avoid domain leakage. When a
source dataset ships its own predefined split (Pothole-600's stereo-video sequences are
deliberately split to keep contiguous frames together and avoid near-duplicate frames
crossing train/test), that source-native split is honored instead of a fresh random
split, since re-splitting randomly would risk leaking near-identical video frames across
train/test. Sources without a native split fall back to a stratified 70/15/15 random split.

Every file in data/merged/images/ is named "<source>_<original filename>" by
build_merged_dataset.py, so the source (and, for Pothole-600, its native split) is read
straight off the filename prefix rather than re-deriving it from data/raw/.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGED_IMAGES_DIR = REPO_ROOT / "data" / "merged" / "images"
ANNOTATIONS_DIR = REPO_ROOT / "data" / "annotations"
SPLITS_DIR = REPO_ROOT / "data" / "annotations" / "splits"

sys.path.insert(0, str(REPO_ROOT / "src" / "preprocessing"))

NATIVE_SPLIT_SOURCE_SUFFIXES = {"_training": "train", "_validation": "val", "_testing": "test"}


def _source_and_native_split(image_path: Path) -> tuple[str, str | None]:
    """image_path.name looks like '<source>_<original_name>', e.g.
    'pothole600_training_0000.png'. Returns (source, native_split_or_None)."""
    stem_source = image_path.name.rsplit("_", 1)[0]  # drop the trailing original filename... but
    # original filenames can themselves contain underscores, so instead match known suffixes
    # directly against the full name rather than trying to split it positionally.
    for suffix, split in NATIVE_SPLIT_SOURCE_SUFFIXES.items():
        marker = suffix + "_"
        if marker in image_path.name:
            source = image_path.name.split(marker)[0] + suffix
            return source, split
    return stem_source, None


def build_splits(seed: int = 0, train_frac: float = 0.70, val_frac: float = 0.15) -> dict[str, list[str]]:
    images = sorted(MERGED_IMAGES_DIR.glob("*.png")) + sorted(MERGED_IMAGES_DIR.glob("*.jpg"))
    if not images:
        raise SystemExit(f"No images found in {MERGED_IMAGES_DIR} — run build_merged_dataset.py first")

    by_split: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    unassigned: dict[str, list[Path]] = {}  # source name -> images needing a random split

    for img in images:
        source, native_split = _source_and_native_split(img)
        if native_split:
            by_split[native_split].append(img)
        else:
            unassigned.setdefault(source, []).append(img)

    rng = random.Random(seed)
    for source, imgs in unassigned.items():
        imgs = imgs.copy()
        rng.shuffle(imgs)
        n_train = round(len(imgs) * train_frac)
        n_val = round(len(imgs) * val_frac)
        by_split["train"].extend(imgs[:n_train])
        by_split["val"].extend(imgs[n_train : n_train + n_val])
        by_split["test"].extend(imgs[n_train + n_val :])

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, list[str]] = {}
    for split, imgs in by_split.items():
        paths = [str(p) for p in sorted(imgs)]
        manifest[split] = paths
        (SPLITS_DIR / f"{split}.txt").write_text("\n".join(paths), encoding="utf-8")

    with open(SPLITS_DIR / "split_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "num_images"])
        for split, imgs in by_split.items():
            writer.writerow([split, len(imgs)])

    total = sum(len(v) for v in by_split.values())
    for split, imgs in by_split.items():
        print(f"[prepare_splits] {split}: {len(imgs)} images ({len(imgs) / total:.1%})")

    from write_yolo_seg import write_data_yaml

    write_data_yaml(
        SPLITS_DIR.parent / "data.yaml",
        train=str(SPLITS_DIR / "train.txt"),
        val=str(SPLITS_DIR / "val.txt"),
        test=str(SPLITS_DIR / "test.txt"),
    )

    return manifest


if __name__ == "__main__":
    build_splits()
