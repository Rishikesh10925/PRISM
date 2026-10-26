"""Reproducible YOLOv8-seg training experiment (Phase 3).

One call = one experiment, recorded in full so it can be reproduced and compared:
  - refuses to reuse an experiment name (never overwrites an earlier run or checkpoint)
  - writes its own data yaml (never touches data.yaml / data.aug_override.yaml)
  - evaluates the selected checkpoint on VALIDATION manifests only -- the frozen test
    split is evaluated separately, once, for the final chosen model
  - records config, seed, git commit, library/GPU versions, SHA-256 of the train manifest
    and checkpoint, training time and val metrics in evaluation/experiments/<name>.json,
    plus one summary row in evaluation/experiments/experiment_log.csv

    python run_experiment.py E0_baseline_retrain --train-manifest data/annotations/splits/train_augmented.txt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "annotations" / "splits"
EXPERIMENT_YAML_DIR = REPO_ROOT / "data" / "annotations" / "experiments"
RUNS_DIR = REPO_ROOT / "models" / "runs"
RESULTS_DIR = REPO_ROOT / "evaluation" / "experiments"
CLASS_NAMES = ["pothole", "crack_longitudinal", "crack_transverse", "crack_alligator", "road_surface", "footpath"]
DEFAULT_VAL_MANIFESTS = {
    "val": SPLITS_DIR / "val.txt",
    "val_compact": SPLITS_DIR / "experiments" / "val_compact.txt",
}
LOG_FIELDS = [
    "name", "finished_at", "description", "model", "imgsz", "epochs_run", "train_images", "train_minutes",
    "val_mask_map50", "val_mask_map50_95", "val_mask_recall", "val_box_map50",
    "val_compact_mask_map50", "val_compact_mask_map50_95", "val_compact_mask_recall", "ms_per_image",
]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def write_data_yaml(name: str, train_manifest: Path, val_manifest: Path) -> Path:
    EXPERIMENT_YAML_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPERIMENT_YAML_DIR / f"{name}__{val_manifest.stem}.yaml"
    data = {"train": str(train_manifest), "val": str(val_manifest), "names": dict(enumerate(CLASS_NAMES))}
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def metrics_dict(metrics) -> dict:
    return {
        "box_precision": float(metrics.box.mp),
        "box_recall": float(metrics.box.mr),
        "box_map50": float(metrics.box.map50),
        "box_map50_95": float(metrics.box.map),
        "mask_precision": float(metrics.seg.mp),
        "mask_recall": float(metrics.seg.mr),
        "mask_map50": float(metrics.seg.map50),
        "mask_map50_95": float(metrics.seg.map),
        "ms_per_image": float(metrics.speed["inference"]),
    }


def evaluate_on_val(weights: Path, name: str, train_manifest: Path, val_manifests: dict[str, Path], imgsz: int) -> dict:
    results = {}
    for label, manifest in val_manifests.items():
        if not manifest.exists():
            continue
        data_yaml = write_data_yaml(name, train_manifest, manifest)
        metrics = YOLO(str(weights)).val(
            data=str(data_yaml), split="val", imgsz=imgsz, project=str(RUNS_DIR / name), name=f"eval_{label}",
            exist_ok=True, plots=False, verbose=False,
        )
        results[label] = metrics_dict(metrics) | {"num_images": len(manifest.read_text(encoding="utf-8").splitlines())}
    return results


def append_log(record: dict) -> None:
    log_path = RESULTS_DIR / "experiment_log.csv"
    val, compact = record["val_metrics"].get("val", {}), record["val_metrics"].get("val_compact", {})
    row = {
        "name": record["name"], "finished_at": record["finished_at"], "description": record["description"],
        "model": record["config"]["model"], "imgsz": record["config"]["imgsz"], "epochs_run": record["epochs_run"],
        "train_images": record["train_images"], "train_minutes": round(record["train_seconds"] / 60, 1),
        "val_mask_map50": val.get("mask_map50"), "val_mask_map50_95": val.get("mask_map50_95"),
        "val_mask_recall": val.get("mask_recall"), "val_box_map50": val.get("box_map50"),
        "val_compact_mask_map50": compact.get("mask_map50"), "val_compact_mask_map50_95": compact.get("mask_map50_95"),
        "val_compact_mask_recall": compact.get("mask_recall"), "ms_per_image": val.get("ms_per_image"),
    }
    new_file = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


def run(
    name: str,
    train_manifest: Path,
    description: str = "",
    model: str = "yolov8n-seg.pt",
    epochs: int = 60,
    imgsz: int = 640,
    batch: int = 16,
    patience: int = 100,
    seed: int = 0,
    extra_train_args: dict | None = None,
    evaluate_only: Path | None = None,
    resume: bool = False,
    workers: int | None = None,
) -> dict:
    record_path = RESULTS_DIR / f"{name}.json"
    last_checkpoint = RUNS_DIR / name / "weights" / "last.pt"
    if record_path.exists():
        raise SystemExit(f"experiment '{name}' already finished -- pick a new name (earlier runs are never overwritten)")
    if resume and not last_checkpoint.exists():
        raise SystemExit(f"nothing to resume: {last_checkpoint} not found")
    if not resume and evaluate_only is None and (RUNS_DIR / name).exists():
        raise SystemExit(f"experiment '{name}' already exists -- pass --resume to continue an interrupted run")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = {"model": str(evaluate_only) if evaluate_only else model, "epochs": epochs, "imgsz": imgsz, "batch": batch, "patience": patience, "seed": seed,
              "deterministic": True, **(extra_train_args or {})}
    started = time.time()
    if resume:
        # Ultralytics restores the original args (data yaml, seed, epochs...) from the checkpoint.
        # workers may be lowered on resume (data loading doesn't change the result) -- a
        # memory-starved laptop crashed E1's DataLoader with the default 8.
        YOLO(str(last_checkpoint)).train(resume=True, **({"workers": workers} if workers is not None else {}))
    if evaluate_only is None:
        data_yaml = write_data_yaml(name, train_manifest, DEFAULT_VAL_MANIFESTS["val"])
        if not resume:
            YOLO(model).train(data=str(data_yaml), project=str(RUNS_DIR), name=name, exist_ok=False, device=0,
                              **{k: v for k, v in config.items() if k != "model"})
        weights = RUNS_DIR / name / "weights" / "best.pt"
        results_csv = RUNS_DIR / name / "results.csv"
        epochs_run = sum(1 for _ in open(results_csv, encoding="utf-8")) - 1
    else:
        weights, epochs_run = evaluate_only, None
    train_seconds = time.time() - started

    record = {
        "name": name,
        "description": description,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": sha256_of(train_manifest),
        "train_images": len(train_manifest.read_text(encoding="utf-8").splitlines()),
        "epochs_run": epochs_run,
        "resumed": resume,  # if true, train_seconds covers only the resumed part
        "train_seconds": train_seconds,
        "checkpoint": str(weights),
        "checkpoint_sha256": sha256_of(weights),
        "git": git_state(),
        "environment": environment(),
        "val_metrics": evaluate_on_val(weights, name, train_manifest, DEFAULT_VAL_MANIFESTS, imgsz),
    }
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    append_log(record)
    print(json.dumps(record["val_metrics"], indent=2))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--model", default="yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluate-only", default=None, help="record an existing checkpoint instead of training")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run from its last.pt")
    parser.add_argument("--workers", type=int, default=None, help="with --resume: DataLoader workers (lower = less RAM)")
    args = parser.parse_args()

    manifest = Path(args.train_manifest)
    run(
        args.name,
        manifest if manifest.is_absolute() else (REPO_ROOT / manifest).resolve(),
        description=args.description,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
        evaluate_only=Path(args.evaluate_only) if args.evaluate_only else None,
        resume=args.resume,
        workers=args.workers,
    )
