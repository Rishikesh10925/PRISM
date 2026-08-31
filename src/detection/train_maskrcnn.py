"""Fine-tune torchvision's Mask R-CNN (ResNet-50-FPN, COCO-pretrained) as the accuracy/
speed comparison baseline against YOLOv8-seg (Phase 3 Task 7).

See maskrcnn_dataset.py's docstring for why torchvision replaces the blueprint's
original Detectron2 choice (no Windows wheel).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

from maskrcnn_dataset import PotholeSegDataset, collate_fn

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "annotations" / "splits"
LABELS_DIR = REPO_ROOT / "data" / "merged" / "labels"
MODELS_DIR = REPO_ROOT / "models"

NUM_CLASSES = 7  # 6 unified classes (class_map.CLASSES) + background


def build_model(num_classes: int = NUM_CLASSES):
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)

    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    return model


def train(
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 0.005,
    train_manifest: str | None = None,
    run_name: str = "maskrcnn_baseline",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Path:
    manifest = Path(train_manifest) if train_manifest else SPLITS_DIR / "train.txt"
    dataset = PotholeSegDataset(manifest, LABELS_DIR)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
    )

    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    # Standard torchvision detection recipe: decay LR twice over training (StepLR) plus
    # a linear warmup during the first epoch only, to avoid the loss-explosion risk of
    # starting a randomly-initialized detection head at full LR. Earlier runs used a
    # flat LR the whole way through, which is a real fairness gap against YOLOv8's
    # tuned cosine+warmup schedule -- see docs/phase3/04_detection_results.md.
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)

    model.train()
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0

        warmup_scheduler = None
        if epoch == 0:
            warmup_iters = min(len(loader) - 1, 500)
            if warmup_iters > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1.0 / 1000, total_iters=warmup_iters
                )

        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if warmup_scheduler is not None:
                warmup_scheduler.step()

            epoch_loss += float(loss)
            num_batches += 1

        lr_scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[train_maskrcnn] epoch {epoch + 1}/{epochs} loss={avg_loss:.4f} "
            f"lr={current_lr:.6f} time={time.time() - epoch_start:.1f}s",
            flush=True,
        )

        # Save after every epoch, not just at the end -- a long run (GPU-bound, tens of
        # minutes) shouldn't lose all progress if the process is interrupted partway.
        MODELS_DIR.mkdir(exist_ok=True)
        dest = MODELS_DIR / f"{run_name}.pt"
        torch.save(model.state_dict(), dest)
        print(f"[train_maskrcnn] checkpoint saved -> {dest} (epoch {epoch + 1}/{epochs})", flush=True)

    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--run-name", default="maskrcnn_baseline")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_manifest=args.train_manifest,
        run_name=args.run_name,
    )
