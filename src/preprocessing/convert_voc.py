"""PASCAL VOC XML -> unified ImageAnnotation list.

Used for RDD2022 (labels D00/D10/D20/D40 ...) and any other bounding-box-only source
that ships VOC XML. Boxes are marked needs_mask=True since VOC gives no segmentation —
box_to_mask_sam.py fills that in later.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from class_map import map_label
from schema import ImageAnnotation, Instance, box_to_polygon


def convert_voc_dir(images_dir: Path, annotations_dir: Path, source: str) -> list[ImageAnnotation]:
    """Convert every .xml file in annotations_dir into an ImageAnnotation.

    Assumes the VOC-standard <annotation><object><bndbox> layout and that each XML's
    <filename> (or matching stem) exists in images_dir.
    """
    results: list[ImageAnnotation] = []
    dropped_labels: set[str] = set()

    for xml_path in sorted(Path(annotations_dir).glob("*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        width = int(size.findtext("width", default="0"))
        height = int(size.findtext("height", default="0"))

        filename = root.findtext("filename") or f"{xml_path.stem}.jpg"
        image_path = str(Path(images_dir) / filename)

        instances: list[Instance] = []
        for obj in root.findall("object"):
            raw_label = obj.findtext("name", default="")
            class_name = map_label(source, raw_label)
            if class_name is None:
                dropped_labels.add(raw_label)
                continue

            box = obj.find("bndbox")
            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))
            if xmax <= xmin or ymax <= ymin:
                continue  # degenerate box, drop rather than write a zero-area polygon

            instances.append(
                Instance(
                    class_name=class_name,
                    polygon=box_to_polygon(xmin, ymin, xmax, ymax),
                    needs_mask=True,
                )
            )

        results.append(
            ImageAnnotation(
                image_path=image_path,
                width=width,
                height=height,
                source=source,
                instances=instances,
            )
        )

    if dropped_labels:
        print(f"[convert_voc:{source}] dropped unmapped labels: {sorted(dropped_labels)}")

    return results
