"""Glue: run all three cue modules on one (image, pothole_mask) pair and fuse them into
a Severity Score. Not its own numbered plan task -- this is what Task 6's calibration
and Task 10's ablation actually call per-image.

Inputs are validated once up front: the mask may be 0/1, 0/255, bool or float (see
mask_utils.binarize_mask) but must have the image's H x W, otherwise ValueError. Empty
masks give S = 0 ("Very Low") with all-zero cues rather than an error, so a detector that
emits an empty instance doesn't crash the request.

Defaults: depth is the scale-normalized (dimensionless, relative -- NOT cm) MiDaS
differential, or Weber shadow contrast when MiDaS is off/unavailable; bounds come from
fusion.SeverityConfig (or a cue_bounds.json written by fit_cue_bounds.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from depth_proxy import depth_proxy, depth_proxy_normalized
from fusion import LEGACY_RAW_DEPTH_MAX, FusionWeights, SeverityConfig, severity_category, severity_score
from geometric import area_ratio
from irregularity import contour_roughness
from mask_utils import binarize_mask, check_same_shape
from schema import SeverityCues
from shadow_heuristic import shadow_contrast, shadow_darkness_ratio


def _validate_inputs(image_bgr: np.ndarray, pothole_mask: np.ndarray) -> np.ndarray:
    image = np.asarray(image_bgr)
    if image.ndim not in (2, 3) or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"image_bgr must be an H x W (x C) image, got shape {image.shape}")
    mask = binarize_mask(pothole_mask)
    check_same_shape(mask, image.shape[:2], "pothole_mask", "image_bgr")
    return mask


def compute_severity_cues(
    image_bgr: np.ndarray,
    pothole_mask: np.ndarray,
    road_mask: np.ndarray | None = None,
    use_midas: bool = True,
    normalize_depth: bool = True,
) -> SeverityCues:
    """Raw cues (a, d, i) for one instance. normalize_depth=False reproduces the legacy
    raw-unit depth cue (MiDaS units / 0-255 gray difference)."""
    mask = _validate_inputs(image_bgr, pothole_mask)
    a = area_ratio(mask, road_mask)
    i = contour_roughness(mask)

    if mask.sum() == 0:
        return SeverityCues(area_ratio=0.0, depth=0.0, irregularity=i, depth_source="none", depth_normalized=normalize_depth)

    def shadow() -> float:
        return shadow_contrast(image_bgr, mask) if normalize_depth else shadow_darkness_ratio(image_bgr, mask)

    if use_midas:
        try:
            d = depth_proxy_normalized(image_bgr, mask, road_mask) if normalize_depth else depth_proxy(image_bgr, mask)
            depth_source = "midas"
        except Exception:
            # MiDaS unavailable (no torch/GPU, model download failed, etc.) -- fall
            # back to the shadow heuristic rather than hard-failing the whole pipeline.
            d = shadow()
            depth_source = "shadow_heuristic"
    else:
        d = shadow()
        depth_source = "shadow_heuristic"

    return SeverityCues(area_ratio=a, depth=d, irregularity=i, depth_source=depth_source, depth_normalized=normalize_depth)


def _resolve_config(config: SeverityConfig | str | Path | None) -> SeverityConfig:
    if config is None:
        return SeverityConfig()
    if isinstance(config, SeverityConfig):
        return config
    return SeverityConfig.from_json(config)


def compute_severity(
    image_bgr: np.ndarray,
    pothole_mask: np.ndarray,
    road_mask: np.ndarray | None = None,
    use_midas: bool = True,
    weights: FusionWeights | None = None,
    depth_max: float | None = None,
    irregularity_max: float | None = None,
    config: SeverityConfig | str | Path | None = None,
    normalize_depth: bool = True,
) -> tuple[float, str, SeverityCues]:
    """Returns (S, category, cues) for one pothole instance.

    `config` is a SeverityConfig or a path to a JSON file from fit_cue_bounds.py (e.g.
    evaluation/severity/cue_bounds.json); default SeverityConfig(). Explicit `weights`,
    `depth_max` and `irregularity_max` override the config. With normalize_depth=False
    and no explicit depth_max, the legacy raw bound (30) is used."""
    cfg = _resolve_config(config)
    cues = compute_severity_cues(image_bgr, pothole_mask, road_mask, use_midas, normalize_depth)
    if depth_max is None and not normalize_depth:
        depth_max = LEGACY_RAW_DEPTH_MAX
    score = severity_score(
        cues.area_ratio,
        cues.depth,
        cues.irregularity,
        weights,
        depth_max,
        irregularity_max,
        config=cfg,
        depth_source=cues.depth_source,
    )
    return score, severity_category(score, cfg.category_thresholds), cues
