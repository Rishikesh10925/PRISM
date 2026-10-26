"""Derived split manifests for Phase 3 experiments. Never modifies the frozen
train/val/test manifests -- writes new files under data/annotations/splits/experiments/.

Pothole-definition experiment (docs/phase2/05_data_quality_report.md, section 5):
kaggle_indian_roads labels some large broken/unpaved road patches as `pothole`, while the
other four sources label compact holes only. "Oversized" = any instance covering more
than OVERSIZED_FRACTION of the image. Whole images are dropped rather than single
instances, because Ultralytics derives each label path from its image path, so editing
labels would mean duplicating images.

  train_augmented_compact.txt  train_augmented minus oversized kaggle_indian_roads images
                               (and their *_augN copies)
  val_compact.txt              val minus the same kind of images -- a subset where both the
                               broad and the compact definition agree, for a fair comparison
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "annotations" / "splits"
LABELS_DIR = REPO_ROOT / "data" / "merged" / "labels"
OUT_DIR = SPLITS_DIR / "experiments"
OVERSIZED_FRACTION = 0.20
OVERSIZED_SOURCE = "kaggle_indian_roads"


def max_instance_fraction(label_text: str) -> float:
    """Largest polygon area as a fraction of the image (labels are normalized to [0, 1])."""
    areas = [
        cv2.contourArea(np.array(line.split()[1:], dtype=float).reshape(-1, 2).astype(np.float32))
        for line in label_text.splitlines()
        if len(line.split()) >= 7
    ]
    return max(areas, default=0.0)


def original_stem(path: Path) -> str:
    return re.sub(r"_aug\d+$", "", path.stem)


def is_oversized(path: Path, labels_dir: Path = LABELS_DIR) -> bool:
    if not path.name.startswith(OVERSIZED_SOURCE):
        return False
    label = labels_dir / f"{original_stem(path)}.txt"
    return max_instance_fraction(label.read_text(encoding="utf-8")) > OVERSIZED_FRACTION


def filter_manifest(paths: list[Path], labels_dir: Path = LABELS_DIR) -> tuple[list[Path], list[Path]]:
    kept, dropped = [], []
    for path in paths:
        (dropped if is_oversized(path, labels_dir) else kept).append(path)
    return kept, dropped


def _read(name: str) -> list[Path]:
    return [Path(p) for p in (SPLITS_DIR / name).read_text(encoding="utf-8").splitlines() if p.strip()]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src, dst in (("train_augmented.txt", "train_augmented_compact.txt"), ("val.txt", "val_compact.txt")):
        kept, dropped = filter_manifest(_read(src))
        (OUT_DIR / dst).write_text("\n".join(str(p) for p in kept), encoding="utf-8")
        print(f"[build_experiment_manifests] {dst}: kept {len(kept)}, dropped {len(dropped)} oversized {OVERSIZED_SOURCE} images")


if __name__ == "__main__":
    main()
