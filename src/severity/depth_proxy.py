"""Depth-proxy severity cue (Phase 4 Task 2): relative depth differential between the
pothole region and the surrounding road, estimated with a monocular depth model (MiDaS).

No public pothole dataset gives real metric depth from a single 2D image (blueprint
Section 5.2), so this is explicitly a *relative* proxy: MiDaS outputs inverse-depth
("closer = larger value"), not metres, and its absolute scale isn't calibrated across
images -- only the differential between the pothole and its immediate surroundings
*within the same image* is used, which cancels out most of that scale ambiguity.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
import torch


@lru_cache(maxsize=1)
def _load_midas(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Cached so repeated calls across many images in a batch don't reload the model."""
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    return model, transforms.small_transform, device


@torch.no_grad()
def estimate_depth_map(image_bgr: np.ndarray) -> np.ndarray:
    """Returns a per-pixel inverse-depth map, same H x W as the input image."""
    model, transform, device = _load_midas()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image_rgb).to(device)

    prediction = model(input_tensor)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1), size=image_bgr.shape[:2], mode="bicubic", align_corners=False
    ).squeeze()

    return prediction.cpu().numpy()


def depth_differential(
    depth_map: np.ndarray, pothole_mask: np.ndarray, surround_dilation_px: int = 15
) -> float:
    """d: the pothole region's mean inverse-depth minus the mean inverse-depth of a ring
    of surrounding road pixels just outside the mask. Since MiDaS outputs *inverse*
    depth (larger = closer to the camera), a pothole -- being physically lower than the
    surrounding road surface, i.e. farther from the camera at the same (x, y) -- reads
    as *lower* inverse-depth than its surroundings. d is defined as
    surround_mean - pothole_mean so that a deeper pothole gives a *larger positive* d,
    matching the "larger depth differential = deeper pothole" convention from the
    blueprint."""
    mask = (pothole_mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (surround_dilation_px * 2 + 1,) * 2)
    dilated = cv2.dilate(mask, kernel)
    surround_ring = (dilated > 0) & (mask == 0)

    if surround_ring.sum() == 0:
        return 0.0  # pothole fills the whole frame -- no surrounding road to compare against

    pothole_mean = float(depth_map[mask > 0].mean())
    surround_mean = float(depth_map[surround_ring].mean())

    return surround_mean - pothole_mean


def depth_proxy(image_bgr: np.ndarray, pothole_mask: np.ndarray, surround_dilation_px: int = 15) -> float:
    depth_map = estimate_depth_map(image_bgr)
    return depth_differential(depth_map, pothole_mask, surround_dilation_px)
