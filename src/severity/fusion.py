"""Severity Score fusion (Phase 4 Task 5): S = 100 * (w1*norm(a) + w2*norm(d) + w3*norm(i)).

Each raw cue lives on a different natural scale, so each gets its own clipped-linear
normalization to [0, 1] before the weighted sum, per blueprint Section 5.2:

  cue  produced by                                   raw range        norm()
  a    geometric.area_ratio                          [0, 1]           clip(a / area_max)
  d    depth_proxy.normalized_depth_differential     dimensionless,   clip(d / depth_max)
       (MiDaS; default) -- relative, NOT cm          typically ~±0.3
  d    shadow_heuristic.shadow_contrast (fallback)   (-inf, 1]        clip(d / shadow_depth_max)
  i    irregularity.contour_roughness                0 (empty) or     clip((i-1)/(irregularity_max-1))
                                                     >= 1 (circle=1)

All bounds, weights and the category thresholds live in ONE place: SeverityConfig below.
Its defaults reproduce the previous behaviour except where the depth cue changed units:

  area_max            1.0    (unchanged: a was only clipped to [0, 1])
  depth_max           0.10   NEW -- for the scale-normalized MiDaS differential. The old
                             bound (LEGACY_RAW_DEPTH_MAX = 30 raw MiDaS units) divided by a
                             typical MiDaS_small road inverse-depth spread of ~300-600
                             gives ~0.05-0.1; 0.10 is a round placeholder in that range.
  shadow_depth_max    0.25   NEW -- for Weber shadow contrast; equals the old raw bound of
                             30 gray levels on a mid-gray (120) road ring.
  irregularity_max    5.0    (unchanged)
  category_thresholds (20, 40, 60, 80) (unchanged: Very Low/Low/Medium/High/Critical)
  weights             (1/3, 1/3, 1/3) (unchanged)

These are *uncalibrated placeholders*. Known problem with them: on 100 real test images
the old configuration gave a median S of 20.5, max 53, and no image reached High/Critical
-- mostly because a real pothole covers a few percent of the road (norm(a) ~ 0.02) and
the other bounds saturate rarely. calibrate_bounds() fixes the *normalization* part
from data: it sets each cue's upper bound to a percentile (default 95th) of that cue on a
reference set (intended: the TRAIN split via fit_cue_bounds.py), so each cue spans
[0, 1] over the real distribution. Fusion *weights* still need human ratings
(calibrate.py, Phase 4 Task 6 -- blocked on the severity validation subset).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

DEFAULT_AREA_MAX = 1.0
DEFAULT_DEPTH_MAX = 0.10  # scale-normalized MiDaS differential (dimensionless); see module docstring
DEFAULT_SHADOW_DEPTH_MAX = 0.25  # Weber shadow contrast (dimensionless)
DEFAULT_IRREGULARITY_MAX = 5.0  # contour roughness multiple of a perfect circle (i=1.0)
DEFAULT_CATEGORY_THRESHOLDS = (20.0, 40.0, 60.0, 80.0)
LEGACY_RAW_DEPTH_MAX = 30.0  # old bound, raw MiDaS units / raw 0-255 gray difference

SEVERITY_CATEGORIES = ["Very Low", "Low", "Medium", "High", "Critical"]


@dataclass
class FusionWeights:
    w_area: float = 1 / 3
    w_depth: float = 1 / 3
    w_irregularity: float = 1 / 3

    def __post_init__(self):
        for name in ("w_area", "w_depth", "w_irregularity"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        total = self.w_area + self.w_depth + self.w_irregularity
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"weights must sum to 1.0, got {total}")


@dataclass
class SeverityConfig:
    """Every tunable number in the severity score, in one documented place (see the module
    docstring for each default's provenance). Pass to pipeline.compute_severity(config=...)
    or fusion.severity_score(config=...); load fitted bounds with SeverityConfig.from_json."""

    area_max: float = DEFAULT_AREA_MAX
    depth_max: float = DEFAULT_DEPTH_MAX
    shadow_depth_max: float = DEFAULT_SHADOW_DEPTH_MAX
    irregularity_max: float = DEFAULT_IRREGULARITY_MAX
    category_thresholds: tuple[float, float, float, float] = DEFAULT_CATEGORY_THRESHOLDS
    weights: FusionWeights = field(default_factory=FusionWeights)

    def __post_init__(self):
        if isinstance(self.weights, Mapping):
            self.weights = FusionWeights(**self.weights)
        self.category_thresholds = tuple(float(t) for t in self.category_thresholds)
        for name in ("area_max", "depth_max", "shadow_depth_max"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a positive finite number, got {value}")
        if not (math.isfinite(self.irregularity_max) and self.irregularity_max > 1.0):
            raise ValueError(f"irregularity_max must be > 1.0, got {self.irregularity_max}")
        t = self.category_thresholds
        if len(t) != len(SEVERITY_CATEGORIES) - 1 or any(b <= a for a, b in zip(t, t[1:])):
            raise ValueError(f"category_thresholds must be 4 strictly increasing values, got {t}")

    def depth_max_for(self, depth_source: str) -> float:
        """Bound for the depth cue, which depends on which estimator produced it."""
        return self.shadow_depth_max if depth_source == "shadow_heuristic" else self.depth_max

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["category_thresholds"] = list(self.category_thresholds)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SeverityConfig":
        """Accepts a flat config dict or {"config": {...}, ...} (fit_cue_bounds.py output).
        Unknown keys (e.g. metadata) are ignored; missing keys keep their defaults."""
        if "config" in data and isinstance(data["config"], Mapping):
            data = data["config"]
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, path: str | Path) -> "SeverityConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self, path: str | Path, metadata: Mapping[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"config": self.to_dict()}
        if metadata:
            payload["metadata"] = dict(metadata)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


DEFAULT_CONFIG = SeverityConfig()


def norm_area(a: float, area_max: float = DEFAULT_AREA_MAX) -> float:
    """Clipped-linear: 0 at a <= 0, 1 at a >= area_max. With the default area_max = 1.0 this
    just guards against out-of-range input (a is already a ratio in [0, 1])."""
    if area_max <= 0:
        raise ValueError("area_max must be positive")
    return float(np.clip(_finite(a) / area_max, 0.0, 1.0))


def norm_depth(d: float, depth_max: float = DEFAULT_DEPTH_MAX) -> float:
    """Clipped-linear: 0 at d <= 0 (pothole reads no deeper than its surroundings, or
    depth estimation was unreliable), 1 at d >= depth_max."""
    if depth_max <= 0:
        raise ValueError("depth_max must be positive")
    return float(np.clip(_finite(d) / depth_max, 0.0, 1.0))


def norm_irregularity(i: float, irregularity_max: float = DEFAULT_IRREGULARITY_MAX) -> float:
    """Clipped-linear over [1, irregularity_max] -- i=1 (perfect circle) maps to 0,
    i>=irregularity_max maps to 1. i=0 (empty mask) also maps to 0."""
    if irregularity_max <= 1.0:
        raise ValueError("irregularity_max must be > 1.0")
    return float(np.clip((_finite(i, 1.0) - 1.0) / (irregularity_max - 1.0), 0.0, 1.0))


def _finite(x: float, default: float = 0.0) -> float:
    x = float(x)
    return x if math.isfinite(x) else default


def severity_score(
    a: float,
    d: float,
    i: float,
    weights: FusionWeights | None = None,
    depth_max: float | None = None,
    irregularity_max: float | None = None,
    config: SeverityConfig | None = None,
    depth_source: str = "midas",
) -> float:
    """S in [0, 100]. Explicit weights/depth_max/irregularity_max override `config`
    (default: SeverityConfig()); `depth_source` picks config.depth_max vs
    config.shadow_depth_max when depth_max isn't given explicitly."""
    config = config or DEFAULT_CONFIG
    weights = weights or config.weights
    depth_max = config.depth_max_for(depth_source) if depth_max is None else depth_max
    irregularity_max = config.irregularity_max if irregularity_max is None else irregularity_max
    fused = (
        weights.w_area * norm_area(a, config.area_max)
        + weights.w_depth * norm_depth(d, depth_max)
        + weights.w_irregularity * norm_irregularity(i, irregularity_max)
    )
    return float(np.clip(100.0 * fused, 0.0, 100.0))


def severity_category(score: float, thresholds: tuple[float, ...] | None = None) -> str:
    """Maps S to the Very Low/Low/Medium/High/Critical labels the dashboard/worklist
    show (blueprint Section 3C extended with a fifth tier for the demo UI). Default bands
    are even 20-point-wide (SeverityConfig.category_thresholds), uncalibrated for the same
    reason as the fusion weights above. A band includes its lower edge (S=20 is "Low").
    This is a display-layer mapping only; it does not change how S itself is computed."""
    thresholds = DEFAULT_CATEGORY_THRESHOLDS if thresholds is None else thresholds
    for label, upper in zip(SEVERITY_CATEGORIES, thresholds):
        if score < upper:
            return label
    return SEVERITY_CATEGORIES[-1]


CUE_KEYS = ("area_ratio", "depth", "shadow_depth", "irregularity")
_CUE_TO_FIELD = {
    "area_ratio": "area_max",
    "depth": "depth_max",
    "shadow_depth": "shadow_depth_max",
    "irregularity": "irregularity_max",
}


def _collect_cue_values(cue_values: Any) -> dict[str, list[float]]:
    """Accepts {"area_ratio": [...], "depth": [...], "shadow_depth": [...],
    "irregularity": [...]} (any subset) or an iterable of schema.SeverityCues (depth routed
    to "depth" or "shadow_depth" by its depth_source)."""
    if isinstance(cue_values, Mapping):
        unknown = set(cue_values) - set(CUE_KEYS)
        if unknown:
            raise ValueError(f"unknown cue keys {sorted(unknown)}; expected a subset of {CUE_KEYS}")
        out = {k: [] for k in CUE_KEYS}
        out.update({k: [float(v) for v in vals] for k, vals in cue_values.items()})
        return out

    out: dict[str, list[float]] = {k: [] for k in CUE_KEYS}
    for cues in cue_values:
        out["area_ratio"].append(float(cues.area_ratio))
        out["irregularity"].append(float(cues.irregularity))
        key = "shadow_depth" if cues.depth_source == "shadow_heuristic" else "depth"
        out[key].append(float(cues.depth))
    return out


def calibrate_bounds(
    cue_values: Mapping[str, Iterable[float]] | Iterable[Any],
    percentile: float = 95.0,
    base: SeverityConfig | None = None,
    min_samples: int = 5,
) -> tuple[SeverityConfig, dict[str, Any]]:
    """Fits each cue's normalization upper bound to the `percentile`-th percentile of its
    observed values on a reference set (so ~(100 - percentile)% of reference potholes
    saturate that cue at 1.0 and the rest spread over [0, 1]). Returns (config, report):
    config = `base` (default SeverityConfig()) with fitted bounds replaced; weights and
    category thresholds are left untouched. report[cue] says, per cue, the sample count
    and either the fitted bound or why the base bound was kept.

    A cue keeps its base bound when it has < `min_samples` finite values, or when the
    percentile is not a valid bound (<= 0 for a/d, <= 1 for i -- e.g. most depth values
    negative means the cue carries no positive evidence on this set).

    Run on the TRAIN split only (fit_cue_bounds.py) so the test split stays untouched."""
    if not 0.0 < percentile <= 100.0:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")
    base = base or SeverityConfig()
    values = _collect_cue_values(cue_values)

    updates: dict[str, float] = {}
    report: dict[str, Any] = {"percentile": percentile}
    for key, vals in values.items():
        arr = np.asarray([v for v in vals if math.isfinite(v)], dtype=np.float64)
        field_name = _CUE_TO_FIELD[key]
        entry: dict[str, Any] = {"n": int(arr.size), "base_bound": getattr(base, field_name)}
        if arr.size < min_samples:
            entry["status"] = f"kept base bound (only {arr.size} finite values, need {min_samples})"
        else:
            bound = float(np.percentile(arr, percentile))
            floor = 1.0 if key == "irregularity" else 0.0
            if bound > floor + 1e-9:
                updates[field_name] = bound
                entry.update(status="fitted", bound=bound, median=float(np.median(arr)))
            else:
                entry["status"] = f"kept base bound (P{percentile:g} = {bound:.4g} is not > {floor})"
        report[key] = entry

    return replace(base, **updates), report
