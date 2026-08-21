"""Compute box mAP@0.5, mAP@0.5:0.95, and mask mAP for a trained Mask R-CNN checkpoint
on the held-out test split (Phase 3 Task 6/8), mirroring evaluate_yolo.py's metrics so
the two are directly comparable in the results table.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from maskrcnn_dataset import PotholeSegDataset, collate_fn
from train_maskrcnn import build_model

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "annotations" / "splits"
LABELS_DIR = REPO_ROOT / "data" / "merged" / "labels"
EVAL_DIR = REPO_ROOT / "evaluation"


@torch.no_grad()
def evaluate(
    weights_path: str,
    split: str = "test",
    score_threshold: float = 0.5,
    out_csv: str | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    manifest = SPLITS_DIR / f"{split}.txt"
    dataset = PotholeSegDataset(manifest, LABELS_DIR)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = build_model().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    box_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=False)
    mask_metric = MeanAveragePrecision(iou_type="segm", class_metrics=False)

    num_images = 0
    inference_times = []

    for images, targets in loader:
        images_dev = [img.to(device) for img in images]

        start = time.time()
        preds = model(images_dev)
        if device == "cuda":
            torch.cuda.synchronize()
        inference_times.append((time.time() - start) / len(images))
        num_images += len(images)

        preds_cpu = []
        for p in preds:
            keep = p["scores"] >= score_threshold
            preds_cpu.append(
                {
                    "boxes": p["boxes"][keep].cpu(),
                    "scores": p["scores"][keep].cpu(),
                    "labels": p["labels"][keep].cpu(),
                    "masks": (p["masks"][keep, 0] > 0.5).cpu().to(torch.uint8),
                }
            )
        targets_cpu = [
            {
                "boxes": t["boxes"].cpu(),
                "labels": t["labels"].cpu(),
                "masks": t["masks"].cpu(),
            }
            for t in targets
        ]

        box_metric.update(preds_cpu, targets_cpu)
        mask_metric.update(preds_cpu, targets_cpu)

    box_result = box_metric.compute()
    mask_result = mask_metric.compute()

    summary = {
        "box_map50": float(box_result["map_50"]),
        "box_map50_95": float(box_result["map"]),
        "mask_map50": float(mask_result["map_50"]),
        "mask_map50_95": float(mask_result["map"]),
        "mean_inference_time_s": sum(inference_times) / len(inference_times) if inference_times else float("nan"),
        "num_images": num_images,
    }

    out_csv_path = Path(out_csv) if out_csv else EVAL_DIR / f"maskrcnn_metrics_{split}_summary.csv"
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(summary.keys()))
        writer.writerow(list(summary.values()))

    print(f"[evaluate_maskrcnn] summary -> {out_csv_path}: {summary}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", help="path to a trained Mask R-CNN state_dict .pt file")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    evaluate(args.weights, split=args.split, score_threshold=args.score_threshold, out_csv=args.out_csv)
