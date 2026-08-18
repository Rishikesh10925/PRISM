"""COCO JSON -> unified ImageAnnotation list.

Used for Roboflow instance-segmentation exports. Segmentation polygons are kept as-is
when present (needs_mask=False); annotations that only carry a bbox (no polygon, or an
RLE mask) fall back to the box rectangle with needs_mask=True.
"""

from __future__ import annotations

import json
from pathlib import Path

from class_map import map_label
from schema import ImageAnnotation, Instance, box_to_polygon


def _polygon_from_segmentation(seg) -> list[tuple[float, float]] | None:
    # COCO polygon segmentation: list of [x1, y1, x2, y2, ...] (usually one ring for
    # a single-part instance). RLE dicts ({"counts": ..., "size": ...}) are not
    # decoded here — treated as "no polygon available", falls back to the bbox.
    if not isinstance(seg, list) or not seg:
        return None
    flat = seg[0]
    if len(flat) < 6:  # fewer than 3 points
        return None
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


def convert_coco_file(json_path: Path, images_dir: Path, source: str) -> list[ImageAnnotation]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    cat_id_to_name = {c["id"]: c["name"] for c in data.get("categories", [])}
    images_by_id = {img["id"]: img for img in data.get("images", [])}

    per_image: dict[int, list[Instance]] = {img_id: [] for img_id in images_by_id}
    dropped_labels: set[str] = set()

    for ann in data.get("annotations", []):
        raw_label = cat_id_to_name.get(ann["category_id"], "")
        class_name = map_label(source, raw_label)
        if class_name is None:
            dropped_labels.add(raw_label)
            continue

        polygon = _polygon_from_segmentation(ann.get("segmentation"))
        needs_mask = polygon is None
        if polygon is None:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            polygon = box_to_polygon(x, y, x + w, y + h)

        per_image.setdefault(ann["image_id"], []).append(
            Instance(class_name=class_name, polygon=polygon, needs_mask=needs_mask)
        )

    results: list[ImageAnnotation] = []
    for img_id, img in images_by_id.items():
        results.append(
            ImageAnnotation(
                image_path=str(Path(images_dir) / img["file_name"]),
                width=img["width"],
                height=img["height"],
                source=source,
                instances=per_image.get(img_id, []),
            )
        )

    if dropped_labels:
        print(f"[convert_coco:{source}] dropped unmapped labels: {sorted(dropped_labels)}")

    return results
