"""Fine-tune YOLOv8-seg on the merged pothole dataset (Phase 3, Tasks 4-5).

Starts from Ultralytics' COCO-pretrained yolov8n-seg weights (transfer learning, per
blueprint Section 5.4) and fine-tunes on data/annotations/data.yaml. Pass
--train-manifest to point at train_augmented.txt instead of train.txt for the
augmentation ablation (Phase 6 Task 5).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = REPO_ROOT / "data" / "annotations" / "data.yaml"
MODELS_DIR = REPO_ROOT / "models"


def _data_yaml_with_train_override(train_manifest: str | None) -> Path:
    """If a non-default train manifest is requested (e.g. train_augmented.txt), write a
    sibling data.yaml with just that field swapped, rather than mutating the shared one."""
    if train_manifest is None:
        return DATA_YAML
    data = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    data["train"] = train_manifest
    override_path = DATA_YAML.with_name("data.aug_override.yaml")
    override_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return override_path


def train(
    model_size: str = "n",
    epochs: int = 60,
    imgsz: int = 640,
    batch: int = 16,
    train_manifest: str | None = None,
    run_name: str = "yolov8_seg_baseline",
    device: str | int = 0,
) -> Path:
    data_yaml = _data_yaml_with_train_override(train_manifest)

    model = YOLO(f"yolov8{model_size}-seg.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(MODELS_DIR / "runs"),
        name=run_name,
        exist_ok=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    MODELS_DIR.mkdir(exist_ok=True)
    dest = MODELS_DIR / f"{run_name}.pt"
    if best_weights.exists():
        shutil.copy2(best_weights, dest)
        print(f"[train_yolo] best checkpoint copied to {dest}")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--train-manifest", default=None, help="e.g. path to train_augmented.txt")
    parser.add_argument("--run-name", default="yolov8_seg_baseline")
    parser.add_argument("--device", default=0)
    args = parser.parse_args()

    train(
        model_size=args.model_size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        train_manifest=args.train_manifest,
        run_name=args.run_name,
        device=args.device,
    )
