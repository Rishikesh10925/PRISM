"""Supervisely per-image JSON -> unified ImageAnnotation list.

Used for the Indian Roads segmentation dataset. Supervisely objects come in two
geometry types: "polygon" (points given directly) and "bitmap" (a base64-encoded,
zlib-compressed PNG mask plus a pixel offset). Both are converted to absolute-pixel
polygons; bitmap objects need cv2 to trace contours out of the decoded mask.
"""

from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path

import cv2
import numpy as np

from class_map import map_label
from schema import ImageAnnotation, Instance


def _bitmap_to_polygon(obj: dict) -> list[tuple[float, float]] | None:
    raw = zlib.decompress(base64.b64decode(obj["bitmap"]["data"]))
    mask = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:  # PNG may decode with an alpha channel; use it as the mask
        mask = mask[:, :, -1] if mask.shape[2] == 4 else mask[:, :, 0]

    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 4:
        return None

    ox, oy = obj["bitmap"]["origin"]
    return [(float(p[0][0] + ox), float(p[0][1] + oy)) for p in largest]


def _polygon_object_to_polygon(obj: dict) -> list[tuple[float, float]] | None:
    exterior = obj.get("points", {}).get("exterior", [])
    if len(exterior) < 3:
        return None
    return [(float(x), float(y)) for x, y in exterior]


def convert_supervisely_dir(images_dir: Path, annotations_dir: Path, source: str) -> list[ImageAnnotation]:
    """annotations_dir holds one <image_name>.json per image (Supervisely project layout)."""
    results: list[ImageAnnotation] = []
    dropped_labels: set[str] = set()

    for json_path in sorted(Path(annotations_dir).glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        image_name = json_path.stem  # Supervisely names <image>.<ext>.json
        width = data["size"]["width"]
        height = data["size"]["height"]

        instances: list[Instance] = []
        for obj in data.get("objects", []):
            class_name = map_label(source, obj.get("classTitle", ""))
            if class_name is None:
                dropped_labels.add(obj.get("classTitle", ""))
                continue

            geom = obj.get("geometryType")
            if geom == "bitmap":
                polygon = _bitmap_to_polygon(obj)
            elif geom == "polygon":
                polygon = _polygon_object_to_polygon(obj)
            else:
                polygon = None  # rectangle/point/line objects unsupported, skip

            if polygon is None:
                continue

            instances.append(Instance(class_name=class_name, polygon=polygon, needs_mask=False))

        results.append(
            ImageAnnotation(
                image_path=str(Path(images_dir) / image_name),
                width=width,
                height=height,
                source=source,
                instances=instances,
            )
        )

    if dropped_labels:
        print(f"[convert_supervisely:{source}] dropped unmapped labels: {sorted(dropped_labels)}")

    return results
