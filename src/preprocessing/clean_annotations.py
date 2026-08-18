"""Automated sanity pass + manual-review sample generation.

Two things happen here, matching the plan's Task 5 ("Clean noisy annotations"):

1. Automatic cleaning: drop annotations that are objectively broken (degenerate/zero-area
   polygons, polygons with <3 points, coordinates entirely outside the image bounds).
   This is deterministic and needs no human judgement.
2. Manual-review sampling: a random 10-15% sample per source is written out with the
   polygon overlaid on the image, plus a review_checklist.csv with one row per sampled
   instance for a human to fill in keep/fix/discard. Deciding whether an annotation is
   *wrong* (as opposed to malformed) is a judgement call this script cannot make --
   see docs/phase2/05_annotation_cleaning.md for how to use the output.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import numpy as np

from schema import ImageAnnotation, Instance


def _polygon_area(polygon: list[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return 0.5 * abs(sum(xs[i] * ys[i - 1] - xs[i - 1] * ys[i] for i in range(len(polygon))))


def auto_clean(annotations: list[ImageAnnotation]) -> tuple[list[ImageAnnotation], int]:
    """Returns (cleaned_annotations, num_instances_dropped)."""
    dropped = 0
    cleaned: list[ImageAnnotation] = []

    for ann in annotations:
        kept_instances: list[Instance] = []
        for inst in ann.instances:
            if len(inst.polygon) < 3:
                dropped += 1
                continue
            if _polygon_area(inst.polygon) < 1.0:  # <1px^2, degenerate
                dropped += 1
                continue
            in_bounds = any(0 <= x <= ann.width and 0 <= y <= ann.height for x, y in inst.polygon)
            if not in_bounds:
                dropped += 1
                continue
            kept_instances.append(inst)
        cleaned.append(
            ImageAnnotation(
                image_path=ann.image_path,
                width=ann.width,
                height=ann.height,
                source=ann.source,
                instances=kept_instances,
            )
        )

    return cleaned, dropped


def sample_for_manual_review(
    annotations: list[ImageAnnotation],
    out_dir: Path,
    fraction: float = 0.12,
    seed: int = 0,
) -> None:
    """Writes overlay images + review_checklist.csv, stratified per source at `fraction`
    (12% by default, within the plan's 10-15% target)."""
    out_dir = Path(out_dir)
    overlays_dir = out_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    by_source: dict[str, list[ImageAnnotation]] = {}
    for ann in annotations:
        by_source.setdefault(ann.source, []).append(ann)

    rng = random.Random(seed)
    rows = []

    for source, source_anns in by_source.items():
        sample_size = max(1, round(len(source_anns) * fraction))
        sampled = rng.sample(source_anns, min(sample_size, len(source_anns)))

        for ann in sampled:
            image_path = Path(ann.image_path)
            if not image_path.exists():
                continue
            img = cv2.imread(str(image_path))
            if img is None:
                continue

            for inst in ann.instances:
                pts = np.array(inst.polygon, dtype=np.int32).reshape(-1, 1, 2)
                color = (0, 0, 255) if inst.needs_mask else (0, 255, 0)  # red=box-only, green=real mask
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

            overlay_path = overlays_dir / f"{source}_{image_path.stem}.jpg"
            cv2.imwrite(str(overlay_path), img)

            rows.append(
                {
                    "source": source,
                    "image": image_path.name,
                    "overlay_file": overlay_path.name,
                    "num_instances": len(ann.instances),
                    "decision": "",  # reviewer fills in: keep / fix / discard
                    "notes": "",
                }
            )

    with open(out_dir / "review_checklist.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["source", "image", "overlay_file", "num_instances", "decision", "notes"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[clean_annotations] wrote {len(rows)} images for manual review to {out_dir}")
