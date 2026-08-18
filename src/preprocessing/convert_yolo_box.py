"""Plain YOLO detection txt (class cx cy w h, normalized) -> unified ImageAnnotation list.

Used for Kaggle/Roboflow sources already exported in YOLOv8 detection (box-only) format.
Boxes are marked needs_mask=True — box_to_mask_sam.py fills the real polygon in later.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from schema import ImageAnnotation, Instance, box_to_polygon


def convert_yolo_box_dir(
    images_dir: Path, labels_dir: Path, source: str, source_classes: list[str]
) -> list[ImageAnnotation]:
    """source_classes maps the label file's integer class id -> raw label name,
    which then goes through class_map.map_label the same way other converters do."""
    from class_map import map_label

    results: list[ImageAnnotation] = []
    dropped_labels: set[str] = set()
    image_exts = {".jpg", ".jpeg", ".png"}

    for label_path in sorted(Path(labels_dir).glob("*.txt")):
        image_path = next(
            (p for ext in image_exts if (p := Path(images_dir) / f"{label_path.stem}{ext}").exists()),
            None,
        )
        if image_path is None:
            continue

        with Image.open(image_path) as im:
            width, height = im.size

        instances: list[Instance] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:5])

            raw_label = source_classes[cls_id] if cls_id < len(source_classes) else str(cls_id)
            class_name = map_label(source, raw_label)
            if class_name is None:
                dropped_labels.add(raw_label)
                continue

            xmin = (cx - w / 2) * width
            xmax = (cx + w / 2) * width
            ymin = (cy - h / 2) * height
            ymax = (cy + h / 2) * height
            if xmax <= xmin or ymax <= ymin:
                continue

            instances.append(
                Instance(class_name=class_name, polygon=box_to_polygon(xmin, ymin, xmax, ymax), needs_mask=True)
            )

        results.append(
            ImageAnnotation(
                image_path=str(image_path), width=width, height=height, source=source, instances=instances
            )
        )

    if dropped_labels:
        print(f"[convert_yolo_box:{source}] dropped unmapped labels: {sorted(dropped_labels)}")

    return results
