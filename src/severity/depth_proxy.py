"""Depth-proxy severity cue (Phase 4 Task 2): relative depth differential between the
pothole region and the surrounding road, estimated with a monocular depth model (MiDaS).

What d is -- and is not
-----------------------
No public pothole dataset gives real metric depth from a single 2D image (blueprint
Section 5.2), so this is explicitly RELATIVE depth evidence, **not physical depth in cm**.

MiDaS orientation (checked against the hub model's documented output): MiDaS predicts
*relative inverse depth* (disparity-like), i.e. LARGER value = CLOSER to the camera. Its
output is only defined up to an unknown per-image scale AND shift (the model is trained
with a scale-and-shift-invariant loss), so raw values cannot be compared across images.

Sign convention (unchanged): a pothole floor is physically lower than the surrounding road,
i.e. farther from the camera, so it reads as LOWER inverse depth than its surroundings.
All d values here are surround_mean - pothole_mean, so a deeper-looking pothole gives a
LARGER POSITIVE d; d <= 0 means "no depth evidence" and normalizes to 0 in fusion.py.

Two variants
------------
* depth_differential / depth_proxy -- the original raw differential in MiDaS units.
  Cancels the per-image shift but NOT the scale, so a fixed clip bound (the old
  DEFAULT_DEPTH_MAX = 30) means different things on different images. Kept for
  comparison/back-compat.
* normalized_depth_differential / depth_proxy_normalized -- the raw differential divided by
  a robust per-image scale: the 5th-95th percentile spread of the depth map over the road
  region (road_mask if given, else the bottom-60% heuristic geometric.py also uses),
  excluding the pothole itself. Differences and percentile spreads both transform the same
  way under MiDaS's unknown affine ambiguity (x -> s*x + t), so the ratio is dimensionless
  and invariant to both s and t. Read it as "how far the pothole sinks below its ring, as a
  fraction of the inverse-depth range the visible road spans". This is what
  pipeline.compute_severity uses by default.
  Fallback: if the road spread is degenerate (flat/constant map) the scale falls back to
  |median| of the road region (scale- but not shift-invariant); if that is also ~0, d = 0.

Limitations: single image, relative only; the ratio still depends on how much road (and
how much perspective depth range) is in frame; MiDaS_small is unreliable on motion blur,
water-filled potholes (reflections) and night images.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from mask_utils import binarize_mask, check_same_shape, road_region, surround_ring

SCALE_LOW_PERCENTILE = 5.0
SCALE_HIGH_PERCENTILE = 95.0
_MIN_SCALE_PIXELS = 16
_DEGENERATE_REL_TOL = 1e-6


@lru_cache(maxsize=1)
def _load_midas(device: str | None = None):
    """Cached so repeated calls across many images in a batch don't reload the model.
    torch is imported lazily so the pure-numpy cue functions (and their tests) don't need
    it."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    return model, transforms.small_transform, device


def estimate_depth_map(image_bgr: np.ndarray) -> np.ndarray:
    """Returns a per-pixel relative INVERSE-depth map (larger = closer), same H x W as the
    input image. Units are arbitrary per image (unknown scale and shift)."""
    import torch

    model, transform, device = _load_midas()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image_rgb).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=image_bgr.shape[:2], mode="bicubic", align_corners=False
        ).squeeze()

    return prediction.cpu().numpy()


def _validate(depth_map: np.ndarray, pothole_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_map, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth_map must be a 2-D (H x W) array, got shape {depth.shape}")
    mask = binarize_mask(pothole_mask)
    check_same_shape(mask, depth.shape, "pothole_mask", "depth_map")
    return depth, mask


def depth_differential(
    depth_map: np.ndarray, pothole_mask: np.ndarray, surround_dilation_px: int = 15
) -> float:
    """RAW d (MiDaS inverse-depth units): mean inverse depth of a ring of road pixels just
    outside the mask minus the pothole's mean inverse depth. Since MiDaS outputs *inverse*
    depth (larger = closer), a pothole -- physically lower than the road, i.e. farther from
    the camera -- reads as lower inverse depth, so a deeper pothole gives a *larger
    positive* d. Returns 0.0 for an empty mask or when there's no surrounding ring (mask
    fills the frame). Non-finite depth values are ignored."""
    depth, mask = _validate(depth_map, pothole_mask)
    if mask.sum() == 0:
        return 0.0

    ring = surround_ring(mask, surround_dilation_px)
    finite = np.isfinite(depth)
    inside = (mask > 0) & finite
    ring = ring & finite
    if inside.sum() == 0 or ring.sum() == 0:
        return 0.0  # no surrounding road to compare against

    return float(depth[ring].mean() - depth[inside].mean())


def depth_scale(
    depth_map: np.ndarray,
    pothole_mask: np.ndarray,
    road_mask: np.ndarray | None = None,
    road_fraction: float = 0.6,
) -> float:
    """Robust per-image scale of the depth map: P95 - P5 over the road region (road_mask,
    else bottom `road_fraction` of the frame) with pothole pixels excluded. Falls back to
    the whole frame minus the pothole if the road region has < 16 usable pixels, and to
    |median| if the spread is degenerate. Returns 0.0 if no usable scale exists."""
    depth, mask = _validate(depth_map, pothole_mask)
    finite = np.isfinite(depth)
    region = road_region(depth.shape, road_mask, road_fraction) & (mask == 0) & finite
    if region.sum() < _MIN_SCALE_PIXELS:
        region = (mask == 0) & finite
    if region.sum() < _MIN_SCALE_PIXELS:
        region = finite
    if region.sum() == 0:
        return 0.0

    values = depth[region]
    lo, hi = np.percentile(values, [SCALE_LOW_PERCENTILE, SCALE_HIGH_PERCENTILE])
    spread = float(hi - lo)
    median_abs = float(abs(np.median(values)))
    if spread > _DEGENERATE_REL_TOL * max(median_abs, 1e-12):
        return spread
    return median_abs if median_abs > 1e-12 else 0.0


def normalized_depth_differential(
    depth_map: np.ndarray,
    pothole_mask: np.ndarray,
    road_mask: np.ndarray | None = None,
    surround_dilation_px: int = 15,
    road_fraction: float = 0.6,
) -> float:
    """Dimensionless d = depth_differential / depth_scale (see module docstring). Invariant
    to multiplying the depth map by any positive constant (and, in the normal non-degenerate
    case, to adding a constant). Positive = pothole reads farther than its ring = deeper.
    RELATIVE evidence only -- not centimetres. Returns 0.0 when undefined."""
    raw = depth_differential(depth_map, pothole_mask, surround_dilation_px)
    if raw == 0.0:
        return 0.0
    scale = depth_scale(depth_map, pothole_mask, road_mask, road_fraction)
    if scale <= 0.0:
        return 0.0
    return float(raw / scale)


def depth_proxy(image_bgr: np.ndarray, pothole_mask: np.ndarray, surround_dilation_px: int = 15) -> float:
    """RAW MiDaS differential for one image (original behaviour; see depth_differential)."""
    mask = binarize_mask(pothole_mask)
    check_same_shape(mask, np.asarray(image_bgr).shape[:2], "pothole_mask", "image_bgr")
    depth_map = estimate_depth_map(image_bgr)
    return depth_differential(depth_map, mask, surround_dilation_px)


def depth_proxy_normalized(
    image_bgr: np.ndarray,
    pothole_mask: np.ndarray,
    road_mask: np.ndarray | None = None,
    surround_dilation_px: int = 15,
) -> float:
    """Scale-normalized MiDaS differential for one image (see
    normalized_depth_differential) -- the default depth cue used by the pipeline."""
    mask = binarize_mask(pothole_mask)
    check_same_shape(mask, np.asarray(image_bgr).shape[:2], "pothole_mask", "image_bgr")
    depth_map = estimate_depth_map(image_bgr)
    return normalized_depth_differential(depth_map, mask, road_mask, surround_dilation_px)
