"""Shared input validation for the severity cue modules.

Every cue function accepts the pothole mask in whatever form upstream code produces it
(uint8 0/1, uint8 0/255, bool, float probabilities, an H x W x 1 channel) and normalizes it
here, so the cue modules themselves only ever see a clean H x W uint8 {0, 1} array.

Binarization rule:
  * bool / integer masks: any value > 0 is foreground (covers 0/1 and 0/255).
  * floating-point masks: foreground is value > 0.5 when the mask lies in [0, 1] (soft
    probabilities, e.g. Mask R-CNN / YOLO mask logits after sigmoid); otherwise (e.g. a
    float 0/255 mask) any value > 0 is foreground. NaN is treated as background.
"""

from __future__ import annotations

import cv2
import numpy as np


def binarize_mask(mask: np.ndarray, name: str = "pothole_mask") -> np.ndarray:
    """Returns an H x W uint8 {0, 1} copy of `mask`. Raises ValueError on input that can't
    be interpreted as a single-channel 2-D mask."""
    if mask is None:
        raise ValueError(f"{name} must be a 2-D array, got None")
    arr = np.asarray(mask)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D (H x W) array, got shape {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got shape {arr.shape}")

    if arr.dtype == bool:
        return arr.astype(np.uint8)
    if np.issubdtype(arr.dtype, np.floating):
        clean = np.nan_to_num(arr.astype(np.float64), nan=0.0)
        if clean.size and clean.min() >= 0.0 and clean.max() <= 1.0:
            return (clean > 0.5).astype(np.uint8)
        return (clean > 0).astype(np.uint8)
    if np.issubdtype(arr.dtype, np.number):
        return (arr > 0).astype(np.uint8)
    raise ValueError(f"{name} has unsupported dtype {arr.dtype}")


def check_same_shape(mask: np.ndarray, reference_hw: tuple[int, int], mask_name: str, reference_name: str) -> None:
    """Raises ValueError when a mask's H x W doesn't match the image / depth map it
    describes -- silently broadcasting or cropping would compute cues on the wrong pixels."""
    if tuple(mask.shape[:2]) != tuple(reference_hw[:2]):
        raise ValueError(
            f"{mask_name} shape {tuple(mask.shape[:2])} does not match {reference_name} shape "
            f"{tuple(reference_hw[:2])}; resize the mask to the image resolution first "
            f"(e.g. YOLO predict(..., retina_masks=True))"
        )


def to_gray(image_bgr: np.ndarray, name: str = "image_bgr") -> np.ndarray:
    """Returns a float32 H x W grayscale copy of a BGR / BGRA / single-channel image."""
    arr = np.asarray(image_bgr)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        src = arr if arr.dtype in (np.uint8, np.uint16, np.float32) else arr.astype(np.float32)
        code = cv2.COLOR_BGR2GRAY if arr.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        return cv2.cvtColor(src, code).astype(np.float32)
    raise ValueError(f"{name} must be H x W, H x W x 1, H x W x 3 (BGR) or H x W x 4, got shape {arr.shape}")


def surround_ring(mask01: np.ndarray, dilation_px: int) -> np.ndarray:
    """Boolean ring of pixels within `dilation_px` (elliptical kernel) of the mask but not
    in it -- the 'immediately surrounding road' that depth/shadow cues compare against."""
    if dilation_px < 1:
        raise ValueError(f"surround_dilation_px must be >= 1, got {dilation_px}")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1,) * 2)
    dilated = cv2.dilate(mask01, kernel)
    return (dilated > 0) & (mask01 == 0)


def road_region(shape_hw: tuple[int, int], road_mask: np.ndarray | None = None, road_fraction: float = 0.6) -> np.ndarray:
    """Boolean road-surface region: the given road_mask, or (heuristic, same as
    geometric.road_surface_area_heuristic) the bottom `road_fraction` of the frame."""
    h, w = shape_hw[:2]
    if road_mask is not None:
        road = binarize_mask(road_mask, "road_mask")
        check_same_shape(road, (h, w), "road_mask", "pothole_mask")
        return road > 0
    region = np.zeros((h, w), dtype=bool)
    top = int(round(h * (1.0 - road_fraction)))
    region[min(max(top, 0), h - 1):, :] = True
    return region
