"""Pothole-600 (binary label PNGs) -> unified ImageAnnotation list.

Pothole-600 ships its own train/validation/testing split, each with rgb/<id>.png and
label/<id>.png (a binary mask: nonzero = pothole). There is only one damage class here,
and the mask is a real, human-verified segmentation (not a box), so needs_mask=False.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from schema import ImageAnnotation, Instance


def _mask_to_polygons(mask_path: Path) -> list[list[tuple[float, float]]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < 4:
            continue
        polygons.append([(float(p[0][0]), float(p[0][1])) for p in contour])
    return polygons


def convert_pothole600_split(split_dir: Path, source: str = "pothole600") -> list[ImageAnnotation]:
    """split_dir is e.g. data/raw/pothole600/training (contains rgb/ and label/)."""
    split_dir = Path(split_dir)
    rgb_dir = split_dir / "rgb"
    label_dir = split_dir / "label"

    results: list[ImageAnnotation] = []
    for rgb_path in sorted(rgb_dir.glob("*.png")):
        label_path = label_dir / rgb_path.name
        with_alpha = cv2.imread(str(rgb_path))
        if with_alpha is None:
            continue
        height, width = with_alpha.shape[:2]

        instances = [
            Instance(class_name="pothole", polygon=poly, needs_mask=False)
            for poly in _mask_to_polygons(label_path)
        ]

        results.append(
            ImageAnnotation(
                image_path=str(rgb_path), width=width, height=height, source=source, instances=instances
            )
        )

    return results
