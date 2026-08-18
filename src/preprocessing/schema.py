"""Common in-memory annotation schema every per-source converter produces.

Every converter (VOC, COCO, Supervisely, plain-YOLO-box) reads its source format and
produces a list of ImageAnnotation objects. write_yolo_seg.py then writes those out
in one unified YOLO-seg label format, regardless of which source they came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Instance:
    class_name: str
    # Absolute pixel-coordinate polygon, e.g. [(x1, y1), (x2, y2), ...].
    # For box-only sources this is the 4-corner box rectangle and needs_mask=True,
    # meaning it still needs a real segmentation polygon from box_to_mask_sam.py.
    polygon: list[tuple[float, float]]
    needs_mask: bool = False


@dataclass
class ImageAnnotation:
    image_path: str
    width: int
    height: int
    source: str
    instances: list[Instance] = field(default_factory=list)


def box_to_polygon(xmin: float, ymin: float, xmax: float, ymax: float) -> list[tuple[float, float]]:
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
