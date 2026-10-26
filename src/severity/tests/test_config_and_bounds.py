import json

import numpy as np
import pytest

from fusion import (
    DEFAULT_DEPTH_MAX,
    DEFAULT_IRREGULARITY_MAX,
    DEFAULT_SHADOW_DEPTH_MAX,
    FusionWeights,
    SeverityConfig,
    calibrate_bounds,
    severity_category,
    severity_score,
)
from schema import SeverityCues


def test_default_config_matches_module_defaults():
    cfg = SeverityConfig()
    assert cfg.area_max == 1.0
    assert cfg.depth_max == DEFAULT_DEPTH_MAX == 0.10
    assert cfg.shadow_depth_max == DEFAULT_SHADOW_DEPTH_MAX == 0.25
    assert cfg.irregularity_max == DEFAULT_IRREGULARITY_MAX == 5.0
    assert cfg.category_thresholds == (20.0, 40.0, 60.0, 80.0)


def test_config_validation():
    with pytest.raises(ValueError):
        SeverityConfig(depth_max=0)
    with pytest.raises(ValueError):
        SeverityConfig(irregularity_max=1.0)
    with pytest.raises(ValueError):
        SeverityConfig(category_thresholds=(20, 10, 60, 80))


def test_explicit_args_override_config():
    cfg = SeverityConfig(depth_max=1.0)
    assert severity_score(0, 0.5, 1.0, config=cfg) == pytest.approx(100 / 3 * 0.5)
    assert severity_score(0, 0.5, 1.0, depth_max=0.5, config=cfg) == pytest.approx(100 / 3)


def test_depth_source_selects_bound():
    cfg = SeverityConfig(depth_max=1.0, shadow_depth_max=0.5)
    assert severity_score(0, 0.5, 1.0, config=cfg, depth_source="shadow_heuristic") == pytest.approx(100 / 3)
    assert severity_score(0, 0.5, 1.0, config=cfg, depth_source="midas") == pytest.approx(100 / 6)


def test_area_max_rescales_area():
    cfg = SeverityConfig(area_max=0.05, weights=FusionWeights(1.0, 0.0, 0.0))
    assert severity_score(0.025, 0, 1, config=cfg) == pytest.approx(50.0)


def test_non_finite_cues_do_not_produce_nan():
    assert severity_score(float("nan"), float("inf"), float("nan")) == pytest.approx(0.0)


def test_custom_category_thresholds():
    assert severity_category(30, thresholds=(10, 20, 25, 28)) == "Critical"
    assert severity_category(30) == "Low"


def test_json_round_trip(tmp_path):
    cfg = SeverityConfig(area_max=0.07, depth_max=0.2, weights=FusionWeights(0.5, 0.25, 0.25))
    path = tmp_path / "bounds.json"
    cfg.to_json(path, metadata={"n": 3})
    loaded = SeverityConfig.from_json(path)
    assert loaded == cfg
    assert json.loads(path.read_text())["metadata"] == {"n": 3}


def test_calibrate_bounds_sets_percentiles():
    rng = np.random.default_rng(0)
    values = {
        "area_ratio": rng.uniform(0, 0.1, 200).tolist(),
        "depth": rng.uniform(-0.05, 0.3, 200).tolist(),
        "irregularity": rng.uniform(1.0, 3.0, 200).tolist(),
    }
    cfg, report = calibrate_bounds(values, percentile=95)
    assert cfg.area_max == pytest.approx(np.percentile(values["area_ratio"], 95))
    assert cfg.depth_max == pytest.approx(np.percentile(values["depth"], 95))
    assert cfg.irregularity_max == pytest.approx(np.percentile(values["irregularity"], 95))
    assert cfg.shadow_depth_max == DEFAULT_SHADOW_DEPTH_MAX  # no values -> kept
    assert report["area_ratio"]["status"] == "fitted"
    assert report["shadow_depth"]["status"].startswith("kept")


def test_calibrate_bounds_spreads_compressed_scores():
    """Placeholder bounds compress scores (as seen on real data); fitted bounds spread them."""
    rng = np.random.default_rng(1)
    cues = [(rng.uniform(0, 0.08), rng.uniform(0, 0.05), rng.uniform(1.0, 2.0)) for _ in range(300)]
    cfg, _ = calibrate_bounds(
        {"area_ratio": [c[0] for c in cues], "depth": [c[1] for c in cues], "irregularity": [c[2] for c in cues]}
    )
    old = [severity_score(*c) for c in cues]
    new = [severity_score(*c, config=cfg) for c in cues]
    assert max(old) < 40
    assert np.percentile(new, 90) > 60


def test_calibrate_bounds_from_severity_cues_routes_depth_by_source():
    cues = [SeverityCues(0.01 * k, 0.02 * k, 1.0 + 0.1 * k, "shadow_heuristic") for k in range(1, 11)]
    cfg, _ = calibrate_bounds(cues, percentile=100)
    assert cfg.shadow_depth_max == pytest.approx(0.2)
    assert cfg.depth_max == DEFAULT_DEPTH_MAX
    assert cfg.irregularity_max == pytest.approx(2.0)


def test_calibrate_bounds_keeps_base_for_unusable_values():
    cfg, report = calibrate_bounds({"depth": [-0.1] * 10, "irregularity": [1.0] * 10, "area_ratio": [0.1, 0.2]})
    assert cfg.depth_max == DEFAULT_DEPTH_MAX
    assert cfg.irregularity_max == DEFAULT_IRREGULARITY_MAX
    assert cfg.area_max == 1.0  # too few samples
    with pytest.raises(ValueError):
        calibrate_bounds({"bogus": [1.0]})
