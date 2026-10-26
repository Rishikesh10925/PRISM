"""Build a self-contained, offline HTML page for blind manual review of segmentation
masks (Phase 2: measuring SAM pseudo-label quality).

The SAM-generated masks are pseudo-labels, not ground truth, so their quality has to be
measured by a person. This samples the box-only (SAM-masked) images from the existing
review sample (data/merged/review_sample/review_checklist.csv, built by
build_merged_dataset.py) plus a few Pothole-600 images whose masks are human-drawn, as
hidden controls: the reviewer can't see which is which, so the control ratings show how
strict the reviewer is -- if human-drawn masks are often rated "bad", SAM's ratings
should be read against that baseline, not against perfection.

Masks are drawn from data/merged/labels -- the exact polygons training used. The page
saves progress in the browser and exports a CSV; summarize_mask_review.py turns that CSV
into the reported numbers.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGED_DIR = REPO_ROOT / "data" / "merged"
REVIEW_DIR = MERGED_DIR / "review_sample"
MAX_SIDE = 640


def read_polygons(label_path: Path, width: int, height: int) -> list[list[list[float]]]:
    polygons = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 7:
            coords = np.array(parts[1:], dtype=float).reshape(-1, 2) * [width, height]
            polygons.append(coords.round(1).tolist())
    return polygons


def select_items(checklist: Path, num_controls: int, seed: int) -> list[dict]:
    with open(checklist, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sam = [r for r in rows if not r["source"].startswith("pothole600")]
    controls = [r for r in rows if r["source"].startswith("pothole600")]
    rng = random.Random(seed)
    items = sam + rng.sample(controls, min(num_controls, len(controls)))
    rng.shuffle(items)
    return items


def encode_item(index: int, row: dict) -> dict | None:
    image_path = MERGED_DIR / "images" / row["image"]
    label_path = MERGED_DIR / "labels" / f"{image_path.stem}.txt"
    image = cv2.imread(str(image_path))
    if image is None or not label_path.exists():
        return None
    height, width = image.shape[:2]
    scale = min(1.0, MAX_SIDE / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    polygons = [[[x * scale, y * scale] for x, y in poly] for poly in read_polygons(label_path, width, height)]
    ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return {
        "id": f"R{index:03d}",
        "image": row["image"],
        "w": image.shape[1],
        "h": image.shape[0],
        "polygons": polygons,
        "src": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode("ascii"),
    }


def build(out_html: Path, num_controls: int = 20, seed: int = 0) -> int:
    items = [encode_item(i, row) for i, row in enumerate(select_items(REVIEW_DIR / "review_checklist.csv", num_controls, seed))]
    items = [it for it in items if it is not None]
    template = (Path(__file__).parent / "mask_review_template.html").read_text(encoding="utf-8")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(template.replace("/*__ITEMS__*/[]", json.dumps(items)), encoding="utf-8")
    print(f"[build_mask_review] {len(items)} items -> {out_html}")
    return len(items)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REVIEW_DIR / "mask_review.html"))
    parser.add_argument("--controls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build(Path(args.out), args.controls, args.seed)
