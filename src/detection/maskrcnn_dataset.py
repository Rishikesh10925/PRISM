"""torchvision Dataset reading our unified YOLO-seg labels for the Mask R-CNN baseline.

Detectron2 (the blueprint's original Mask R-CNN choice) has no official Windows wheel
and building it from source needs a matching MSVC + CUDA toolchain — impractical here.
torchvision's maskrcnn_resnet50_fpn is the standard, pip-installable, cross-platform
implementation of the same architecture and is used instead (see
docs/phase3/03_maskrcnn_baseline_note.md for the full rationale).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PotholeSegDataset(Dataset):
    def __init__(self, manifest_path: str | Path, labels_dir: str | Path, transforms=None):
        self.image_paths = [
            Path(p) for p in Path(manifest_path).read_text(encoding="utf-8").splitlines() if p.strip()
        ]
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        label_path = self.labels_dir / f"{img_path.stem}.txt"
        boxes, labels, masks = [], [], []

        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 7:  # class_id + at least 3 (x, y) pairs
                    continue
                cls_id = int(parts[0])
                coords = [float(v) for v in parts[1:]]
                xs = [coords[i] * width for i in range(0, len(coords), 2)]
                ys = [coords[i] * height for i in range(1, len(coords), 2)]

                mask = np.zeros((height, width), dtype=np.uint8)
                pts = np.array(list(zip(xs, ys)), dtype=np.int32)
                cv2.fillPoly(mask, [pts], 1)
                if mask.sum() == 0:
                    continue

                boxes.append([min(xs), min(ys), max(xs), max(ys)])
                labels.append(cls_id + 1)  # torchvision reserves label 0 for background
                masks.append(mask)

        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            masks_t = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            masks_t = torch.zeros((0, height, width), dtype=torch.uint8)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([idx]),
        }

        image_t = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        if self.transforms:
            image_t, target = self.transforms(image_t, target)

        return image_t, target


def collate_fn(batch):
    return tuple(zip(*batch))
