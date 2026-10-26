"""Failure analysis of the final test evaluation (Phase 3): per-source TP/FP/FN from
evaluate_detection.py's per-image CSV, plus a contact sheet of the worst images (ground
truth outlined green, predictions at the deployed threshold outlined red).

    python failure_breakdown.py --name deployed_multisource --weights ../../models/yolov8n_seg_multisource.pt --conf 0.35
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = REPO_ROOT / "data" / "merged" / "images"
LABELS_DIR = REPO_ROOT / "data" / "merged" / "labels"
OUT_DIR = REPO_ROOT / "evaluation" / "detection"
SOURCES = ("pothole600", "kaggle_annotated_potholes", "kaggle_potholes_yolov8", "kaggle_indian_roads", "kaggle_severity_levels")
TILE = 320
NUM_WORST = 12


def per_source(rows: list[dict]) -> dict:
    agg = defaultdict(lambda: {"images": 0, "gt": 0, "tp": 0, "fp": 0, "fn": 0, "no_prediction": 0})
    for r in rows:
        source = next((s for s in SOURCES if r["image"].startswith(s)), "unknown")
        a = agg[source]
        a["images"] += 1
        a["no_prediction"] += int(r["num_pred"]) == 0
        for k in ("tp", "fp", "fn"):
            a[k] += int(r[k])
        a["gt"] += int(r["num_gt"])
    for a in agg.values():
        a["recall"] = round(a["tp"] / (a["tp"] + a["fn"]), 3) if a["tp"] + a["fn"] else None
        a["precision"] = round(a["tp"] / (a["tp"] + a["fp"]), 3) if a["tp"] + a["fp"] else None
    return dict(agg)


def tile(model: YOLO, image_name: str, conf: float, caption: str) -> np.ndarray:
    image = cv2.imread(str(IMAGES_DIR / image_name))
    h, w = image.shape[:2]
    for line in (LABELS_DIR / f"{Path(image_name).stem}.txt").read_text(encoding="utf-8").splitlines():
        if len(line.split()) >= 7:
            poly = (np.array(line.split()[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
            cv2.polylines(image, [poly], True, (0, 200, 0), max(2, w // 300))
    result = model.predict(image, conf=conf, verbose=False, retina_masks=True)[0]
    if result.masks is not None:
        for poly in result.masks.xy:
            if len(poly) >= 3:
                cv2.polylines(image, [poly.astype(np.int32)], True, (0, 0, 255), max(2, w // 300))
    scale = TILE / max(h, w)
    image = cv2.resize(image, (round(w * scale), round(h * scale)))
    canvas = np.zeros((TILE, TILE, 3), np.uint8)
    canvas[: image.shape[0], : image.shape[1]] = image
    cv2.rectangle(canvas, (0, 0), (TILE, 18), (0, 0, 0), -1)
    cv2.putText(canvas, caption, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return canvas


def main(name: str, weights: str, conf: float) -> None:
    with open(OUT_DIR / f"test_per_image_{name}.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    breakdown = per_source(rows)
    (OUT_DIR / f"test_per_source_{name}.json").write_text(json.dumps(breakdown, indent=2), encoding="utf-8")
    print(json.dumps(breakdown, indent=2))

    worst = sorted(rows, key=lambda r: int(r["fn"]) + int(r["fp"]), reverse=True)[:NUM_WORST]
    model = YOLO(weights)
    tiles = [tile(model, r["image"], conf, f"{r['image'][:30]} FN={r['fn']} FP={r['fp']}") for r in worst]
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)])
    out = REPO_ROOT / "evaluation" / "failure_gallery" / f"test_worst_{name}.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(f"[failure_breakdown] {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="deployed_multisource")
    parser.add_argument("--weights", default=str(REPO_ROOT / "models" / "yolov8n_seg_multisource.pt"))
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()
    main(args.name, args.weights, args.conf)
