"""Instance-level detection evaluation and confidence-threshold selection (Phase 3).

mAP (from Ultralytics' validator) summarizes the whole precision-recall curve, but the
deployed backend has to pick ONE confidence threshold. This script:

  select  -- on the VALIDATION split, sweeps thresholds and picks the lowest one whose
             F1 (precision/recall of instances matched by mask IoU >= 0.5) is within
             F1_TIE of the best -- a missed pothole is costlier for a municipality than
             a false alarm an inspector can dismiss. Max-F2 is reported alongside.
  test    -- run ONCE on the frozen test split for the final model, at the threshold
             chosen on val: Ultralytics mAP metrics + instance-level P/R/F1 + per-image
             TP/FP/FN, empty-prediction images, latency and model size.

    python evaluate_detection.py select models/runs/E0_baseline_retrain/weights/best.pt
    python evaluate_detection.py test   <weights> --conf 0.30 --name final
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "annotations" / "splits"
LABELS_DIR = REPO_ROOT / "data" / "merged" / "labels"
OUT_DIR = REPO_ROOT / "evaluation" / "detection"
MATCH_IOU = 0.5
THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 0.91, 0.05)]
F1_TIE = 0.005


def gt_masks(image_stem: str, height: int, width: int) -> list[np.ndarray]:
    masks = []
    for line in (LABELS_DIR / f"{image_stem}.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 7:
            m = np.zeros((height, width), np.uint8)
            cv2.fillPoly(m, [(np.array(parts[1:], float).reshape(-1, 2) * [width, height]).astype(np.int32)], 1)
            masks.append(m.astype(bool))
    return masks


def iou_matrix(preds: list[np.ndarray], gts: list[np.ndarray]) -> np.ndarray:
    out = np.zeros((len(preds), len(gts)))
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            union = np.logical_or(p, g).sum()
            out[i, j] = np.logical_and(p, g).sum() / union if union else 0.0
    return out


def greedy_match(ious: np.ndarray, scores: np.ndarray, iou_threshold: float = MATCH_IOU) -> tuple[int, int, int, list[float]]:
    """Highest-confidence predictions claim their best unmatched GT first (COCO-style).
    Returns (tp, fp, fn, matched IoUs)."""
    matched_gt: set[int] = set()
    matched_ious = []
    tp = 0
    for i in np.argsort(-scores):
        if ious.shape[1] == 0:
            break
        candidates = [(ious[i, j], j) for j in range(ious.shape[1]) if j not in matched_gt]
        if candidates:
            best_iou, best_j = max(candidates)
            if best_iou >= iou_threshold:
                matched_gt.add(best_j)
                matched_ious.append(float(best_iou))
                tp += 1
    fp = len(scores) - tp
    fn = ious.shape[1] - tp
    return tp, fp, fn, matched_ious


def predict_split(weights: str, split: str) -> list[dict]:
    """One low-threshold pass; thresholds are applied afterwards so the sweep is cheap."""
    model = YOLO(weights)
    records = []
    for path in [Path(p) for p in (SPLITS_DIR / f"{split}.txt").read_text(encoding="utf-8").splitlines() if p.strip()]:
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        start = time.perf_counter()
        result = model.predict(image, conf=0.05, verbose=False, retina_masks=True)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        masks = [] if result.masks is None else [(m.cpu().numpy() > 0.5) for m in result.masks.data]
        scores = np.array([]) if result.boxes is None else result.boxes.conf.cpu().numpy()
        gts = gt_masks(path.stem, h, w)
        records.append({"image": path.name, "scores": scores, "ious": iou_matrix(masks, gts), "num_gt": len(gts), "seconds": elapsed})
    return records


def counts_at(records: list[dict], conf: float) -> dict:
    tp = fp = fn = 0
    ious_all = []
    for r in records:
        keep = r["scores"] >= conf
        t, f, n, ious = greedy_match(r["ious"][keep], r["scores"][keep])
        tp, fp, fn = tp + t, fp + f, fn + n
        ious_all += ious
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    return {"conf": conf, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "f2": f2,
            "mean_matched_iou": float(np.mean(ious_all)) if ious_all else 0.0}


def select(weights: str) -> dict:
    records = predict_split(weights, "val")
    sweep = [counts_at(records, t) for t in THRESHOLDS]
    # F1 is usually flat near its peak; among thresholds within F1_TIE of the best, take
    # the lowest (highest recall) -- a missed pothole costs more than a dismissable alarm.
    top_f1 = max(r["f1"] for r in sweep)
    best_f1 = min((r for r in sweep if r["f1"] >= top_f1 - F1_TIE), key=lambda r: r["conf"])
    best_f2 = max(sweep, key=lambda r: r["f2"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "conf_threshold_sweep_val.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        writer.writeheader()
        writer.writerows(sweep)
    choice = {"weights": weights, "split": "val", "match_iou": MATCH_IOU, "chosen_conf": best_f1["conf"], "criterion": f"lowest threshold within {F1_TIE} of max F1",
              "at_chosen": best_f1, "max_f2": best_f2, "at_0.25_previous_default": counts_at(records, 0.25)}
    (OUT_DIR / "conf_threshold_choice.json").write_text(json.dumps(choice, indent=2), encoding="utf-8")
    print(json.dumps(choice, indent=2))
    return choice


def test(weights: str, conf: float, name: str) -> dict:
    data_yaml = REPO_ROOT / "data" / "annotations" / "data.yaml"
    m = YOLO(weights).val(data=str(data_yaml), split="test", plots=False, verbose=False,
                          project=str(REPO_ROOT / "models" / "runs"), name=f"final_test_{name}", exist_ok=True)
    records = predict_split(weights, "test")
    per_image = []
    for r in records:
        keep = r["scores"] >= conf
        tp, fp, fn, ious = greedy_match(r["ious"][keep], r["scores"][keep])
        per_image.append({"image": r["image"], "num_gt": r["num_gt"], "num_pred": int(keep.sum()), "tp": tp, "fp": fp, "fn": fn,
                          "mean_iou": round(float(np.mean(ious)), 4) if ious else "", "ms": round(1000 * r["seconds"], 1)})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"test_per_image_{name}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)
    ms = np.array([1000 * r["seconds"] for r in records[5:]])  # first few include warm-up
    summary = {
        "weights": weights, "split": "test", "num_images": len(records), "conf": conf,
        "map": {"box_map50": float(m.box.map50), "box_map50_95": float(m.box.map), "mask_map50": float(m.seg.map50),
                "mask_map50_95": float(m.seg.map), "box_precision": float(m.box.mp), "box_recall": float(m.box.mr),
                "mask_precision": float(m.seg.mp), "mask_recall": float(m.seg.mr)},
        "instances_at_conf": counts_at(records, conf),
        "images_with_no_prediction": sum(1 for p in per_image if p["num_pred"] == 0),
        "images_with_missed_pothole": sum(1 for p in per_image if p["fn"] > 0),
        "images_with_false_positive": sum(1 for p in per_image if p["fp"] > 0),
        "latency_ms_end_to_end": {"mean": float(ms.mean()), "p50": float(np.percentile(ms, 50)), "p95": float(np.percentile(ms, 95)),
                                  "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
        "model_size_mb": round(Path(weights).stat().st_size / 1e6, 2),
    }
    (OUT_DIR / f"test_summary_{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["select", "test"])
    parser.add_argument("weights")
    parser.add_argument("--conf", type=float, default=None, help="test mode: threshold chosen on val")
    parser.add_argument("--name", default="final")
    args = parser.parse_args()
    if args.mode == "select":
        select(args.weights)
    else:
        if args.conf is None:
            raise SystemExit("test mode needs --conf (choose it on val with `select` first)")
        test(args.weights, args.conf, args.name)
