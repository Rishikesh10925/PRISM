"""Latency/throughput benchmark (Phase 6 Task 7): mean and p95 end-to-end inference
latency, plus images/sec, for YOLOv8n-seg and Mask R-CNN on real test images, CPU vs GPU.

"End-to-end" here means image decode (cv2.imread) through model forward pass and
postprocessing to detections -- the same per-request cost the FastAPI backend
(backend/app/services/inference.py) actually pays for one citizen-report photo, run
one image at a time (batch=1) since that's how the backend calls it, not a batched
throughput number that wouldn't reflect real request latency.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "annotations" / "splits"
MODELS_DIR = REPO_ROOT / "models"
EVAL_DIR = REPO_ROOT / "evaluation"

N_WARMUP = 5
N_TIMED = 50


def _sample_image_paths(n: int, seed: int = 0) -> list[Path]:
    import random

    manifest = SPLITS_DIR / "test.txt"
    paths = [Path(p) for p in manifest.read_text(encoding="utf-8").splitlines() if p.strip()]
    rng = random.Random(seed)
    rng.shuffle(paths)
    return paths[:n]


def _percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


def benchmark_yolo(weights_path: str, device: str, image_paths: list[Path]) -> dict:
    from ultralytics import YOLO

    model = YOLO(weights_path)
    model.to(device)

    for path in image_paths[:N_WARMUP]:
        image = cv2.imread(str(path))
        model.predict(image, device=device, verbose=False)

    latencies = []
    for path in image_paths:
        start = time.perf_counter()
        image = cv2.imread(str(path))
        model.predict(image, device=device, verbose=False)
        if device != "cpu":
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)

    return _summarize("yolov8n-seg", device, latencies)


def benchmark_maskrcnn(weights_path: str, device: str, image_paths: list[Path]) -> dict:
    import torchvision.transforms.functional as F

    from train_maskrcnn import build_model

    model = build_model().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    def _run_one(path: Path):
        image = cv2.imread(str(path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(image_rgb).to(device)
        with torch.no_grad():
            model([tensor])

    for path in image_paths[:N_WARMUP]:
        _run_one(path)

    latencies = []
    for path in image_paths:
        start = time.perf_counter()
        _run_one(path)
        if device != "cpu":
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)

    return _summarize("maskrcnn", device, latencies)


def _summarize(model_name: str, device: str, latencies: list[float]) -> dict:
    latencies_ms = [t * 1000 for t in latencies]
    mean_ms = float(np.mean(latencies_ms))
    return {
        "model": model_name,
        "device": device,
        "num_images": len(latencies),
        "mean_latency_ms": mean_ms,
        "p50_latency_ms": _percentile(latencies_ms, 50),
        "p95_latency_ms": _percentile(latencies_ms, 95),
        "images_per_sec": float(1000.0 / mean_ms) if mean_ms > 0 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=N_TIMED)
    parser.add_argument("--out-csv", default=str(EVAL_DIR / "latency_benchmark.csv"))
    args = parser.parse_args()

    image_paths = _sample_image_paths(args.n_images)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    results = []
    for device in devices:
        yolo_weights = MODELS_DIR / "yolov8n_seg_multisource.pt"
        print(f"[benchmark] yolov8n-seg on {device} ...")
        results.append(benchmark_yolo(str(yolo_weights), device, image_paths))

        maskrcnn_weights = MODELS_DIR / "maskrcnn_multisource.pt"
        if not maskrcnn_weights.exists():
            maskrcnn_weights = MODELS_DIR / "maskrcnn_baseline.pt"
        print(f"[benchmark] maskrcnn ({maskrcnn_weights.name}) on {device} ...")
        results.append(benchmark_maskrcnn(str(maskrcnn_weights), device, image_paths))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"[benchmark] wrote {out_path}")
    for r in results:
        print(
            f"  {r['model']:12s} {r['device']:4s} mean={r['mean_latency_ms']:.1f}ms "
            f"p95={r['p95_latency_ms']:.1f}ms throughput={r['images_per_sec']:.2f} img/s"
        )


if __name__ == "__main__":
    main()
