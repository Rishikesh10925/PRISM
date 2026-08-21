"""ImageAnnotation list -> unified YOLO-seg label files + data.yaml.

YOLO-seg label line format: `class_id x1 y1 x2 y2 ... xn yn` with all coordinates
normalized to [0, 1]. One .txt per image, same stem as the image file.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from class_map import CLASSES, NAME_TO_ID
from schema import ImageAnnotation


def write_yolo_seg(annotations: list[ImageAnnotation], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    for ann in annotations:
        stem = Path(ann.image_path).stem
        lines = []
        for inst in ann.instances:
            if inst.needs_mask:
                # Not yet converted to a real mask (box_to_mask_sam.py handles that);
                # skip writing this instance so partially-converted data can't leak
                # into training silently.
                continue
            cls_id = NAME_TO_ID[inst.class_name]
            coords = []
            for x, y in inst.polygon:
                coords.append(f"{x / ann.width:.6f}")
                coords.append(f"{y / ann.height:.6f}")
            lines.append(" ".join([str(cls_id), *coords]))

        (out_dir / "labels" / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")


def write_data_yaml(
    path: Path,
    train: str | None = None,
    val: str | None = None,
    test: str | None = None,
) -> None:
    """train/val/test are paths to the split manifest .txt files written by
    src/detection/prepare_splits.py (each one line per image path), the format
    Ultralytics expects when a directory-per-split layout isn't used. Falls back to a
    generic "images" placeholder for any split not yet known (e.g. right after
    build_merged_dataset.py, before prepare_splits.py has run)."""
    data = {
        "train": train or "images",
        "val": val or "images",
        "test": test or "images",
        "names": {i: name for i, name in enumerate(CLASSES)},
    }
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
