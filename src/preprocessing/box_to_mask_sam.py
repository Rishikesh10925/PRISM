"""Box -> mask conversion via SAM (Segment Anything), for sources that only ship
bounding boxes (VOC XML / YOLO txt), not real segmentation masks.

Every needs_mask=True Instance carries a rectangle placeholder polygon (its box
corners), which write_yolo_seg.py deliberately skips rather than train on. This module
replaces that rectangle with a real polygon: SAM is prompted with the box, and the
largest contour of its output mask becomes the new polygon, with needs_mask cleared.

A manifest of which instances were SAM-generated (vs. real source masks) is written
alongside the output so a human reviewer can prioritize spot-checking SAM's output
first (Phase 2 Task 3's "manually spot-check ~10-15%" step) rather than treating this
as equivalent to a human-verified mask.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from schema import ImageAnnotation, Instance


@lru_cache(maxsize=1)
def _load_sam(model_name: str = "sam_b.pt"):
    from ultralytics import SAM

    return SAM(model_name)


def _largest_polygon_from_mask(mask: np.ndarray) -> list[tuple[float, float]] | None:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 4:
        return None
    return [(float(p[0][0]), float(p[0][1])) for p in largest]


def convert_boxes_to_masks(
    annotations: list[ImageAnnotation], model_name: str = "sam_b.pt", manifest_csv: Path | None = None
) -> list[ImageAnnotation]:
    """Returns a new list of ImageAnnotation with every needs_mask=True instance's
    polygon replaced by a real SAM-derived mask polygon (needs_mask cleared).
    Instances SAM fails to segment (empty/degenerate output) are dropped, matching
    auto_clean's "don't write a broken instance" convention -- logged, not silent."""
    sam = _load_sam(model_name)
    manifest_rows = []
    results: list[ImageAnnotation] = []
    dropped = 0

    for ann in annotations:
        needs_mask_instances = [inst for inst in ann.instances if inst.needs_mask]
        if not needs_mask_instances:
            results.append(ann)
            continue

        image = cv2.imread(ann.image_path)
        if image is None:
            results.append(ann)
            continue

        # One SAM call per IMAGE with every box at once -- SAM's image encoder (the
        # expensive part) runs once per image regardless of how many boxes are passed,
        # so batching here instead of one sam() call per instance is a large speedup
        # whenever an image has more than one box-only instance.
        boxes = []
        for inst in needs_mask_instances:
            xs = [p[0] for p in inst.polygon]
            ys = [p[1] for p in inst.polygon]
            boxes.append([min(xs), min(ys), max(xs), max(ys)])

        sam_result = sam(image, bboxes=boxes, verbose=False)
        mask_data = sam_result[0].masks
        masks = mask_data.data.cpu().numpy() if mask_data is not None else []

        new_instances: list[Instance] = [inst for inst in ann.instances if not inst.needs_mask]
        for inst, mask in zip(needs_mask_instances, masks):
            polygon = _largest_polygon_from_mask(mask)
            if polygon is None:
                dropped += 1
                manifest_rows.append(
                    {"image": ann.image_path, "source": ann.source, "class": inst.class_name, "sam_generated": "FAILED"}
                )
                continue

            new_instances.append(Instance(class_name=inst.class_name, polygon=polygon, needs_mask=False))
            manifest_rows.append(
                {"image": ann.image_path, "source": ann.source, "class": inst.class_name, "sam_generated": "TRUE"}
            )

        for _ in range(len(needs_mask_instances) - len(masks)):
            dropped += 1
            manifest_rows.append(
                {"image": ann.image_path, "source": ann.source, "class": "unknown", "sam_generated": "FAILED"}
            )

        results.append(
            ImageAnnotation(
                image_path=ann.image_path, width=ann.width, height=ann.height, source=ann.source, instances=new_instances
            )
        )

    if manifest_csv:
        manifest_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "source", "class", "sam_generated"])
            writer.writeheader()
            writer.writerows(manifest_rows)

    print(f"[box_to_mask_sam] converted {len(manifest_rows) - dropped} boxes to masks, {dropped} failed/dropped")
    return results
