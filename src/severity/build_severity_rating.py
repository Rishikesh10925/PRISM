"""Build the human severity-rating set (Phase 4: validating the Severity Score).

Selects one ground-truth pothole instance per image from the frozen TEST split --
stratified, NUM_PER_SOURCE per source, seeded -- so the ratings are never used for
tuning or calibration, only for the final validation of the score. Each pothole is
outlined on its photo and rated 1-5 (or 0 = can't judge) in a page served by
rating_server.py, which saves every answer straight to disk.

Rating on ground-truth instances (not model predictions) isolates severity scoring from
detection errors: evaluate_severity.py scores the same instance masks.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_MANIFEST = REPO_ROOT / "data" / "annotations" / "splits" / "test.txt"
LABELS_DIR = REPO_ROOT / "data" / "merged" / "labels"
OUT_DIR = REPO_ROOT / "data" / "severity_val_subset"
SOURCES = ("pothole600", "kaggle_annotated_potholes", "kaggle_potholes_yolov8", "kaggle_indian_roads", "kaggle_severity_levels")
NUM_PER_SOURCE = 12
MIN_AREA_FRACTION = 0.002  # skip specks too small to judge by eye
MAX_SIDE = 720


def instance_polygons(label_path: Path) -> list[np.ndarray]:
    return [np.array(l.split()[1:], dtype=float).reshape(-1, 2) for l in label_path.read_text(encoding="utf-8").splitlines() if len(l.split()) >= 7]


def select_instances(manifest: Path = TEST_MANIFEST, labels_dir: Path = LABELS_DIR, per_source: int = NUM_PER_SOURCE, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    candidates = defaultdict(list)
    for line in manifest.read_text(encoding="utf-8").splitlines():
        image = Path(line)
        source = next((s for s in SOURCES if image.name.startswith(s)), None)
        if source is None:
            continue
        polys = instance_polygons(labels_dir / f"{image.stem}.txt")
        eligible = [i for i, p in enumerate(polys) if cv2.contourArea(p.astype(np.float32)) >= MIN_AREA_FRACTION]
        if eligible:
            candidates[source].append({"image_path": str(image), "image": image.name, "source": source, "instance": rng.choice(eligible)})
    selected = []
    for source in SOURCES:
        pool = sorted(candidates[source], key=lambda c: c["image"])
        selected += rng.sample(pool, min(per_source, len(pool)))
    rng.shuffle(selected)
    for i, item in enumerate(selected):
        item["rating_id"] = f"S{i:03d}"
    return selected


def encode(item: dict) -> dict:
    image = cv2.imread(item["image_path"])
    h, w = image.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    poly = instance_polygons(LABELS_DIR / f"{Path(item['image_path']).stem}.txt")[item["instance"]]
    points = (poly * [image.shape[1], image.shape[0]]).round(1).tolist()
    ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return {"id": item["rating_id"], "image": item["image"], "instance": item["instance"], "w": image.shape[1], "h": image.shape[0],
            "polygon": points, "src": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode("ascii")}


def build(out_dir: Path = OUT_DIR, seed: int = 0) -> int:
    items = select_instances(seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rating_set.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rating_id", "image", "source", "instance"])
        writer.writeheader()
        writer.writerows({k: it[k] for k in ("rating_id", "image", "source", "instance")} for it in items)
    template = (Path(__file__).parent / "severity_rating_template.html").read_text(encoding="utf-8")
    (out_dir / "severity_rating.html").write_text(template.replace("/*__ITEMS__*/[]", json.dumps([encode(it) for it in items])), encoding="utf-8")
    print(f"[build_severity_rating] {len(items)} instances -> {out_dir / 'severity_rating.html'}")
    return len(items)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    build(seed=parser.parse_args().seed)
