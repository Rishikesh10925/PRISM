import pytest

from fusion import (
    FusionWeights,
    norm_area,
    norm_depth,
    norm_irregularity,
    severity_category,
    severity_score,
)


def test_fusion_weights_must_sum_to_one():
    FusionWeights(0.5, 0.3, 0.2)  # doesn't raise
    with pytest.raises(ValueError):
        FusionWeights(0.5, 0.5, 0.5)


def test_norm_area_passthrough_and_clip():
    assert norm_area(0.3) == 0.3
    assert norm_area(1.5) == 1.0
    assert norm_area(-0.1) == 0.0


def test_norm_depth_clips_at_bounds():
    assert norm_depth(0.0, depth_max=10) == 0.0
    assert norm_depth(5.0, depth_max=10) == 0.5
    assert norm_depth(100.0, depth_max=10) == 1.0
    assert norm_depth(-5.0, depth_max=10) == 0.0


def test_norm_irregularity_clips_at_bounds():
    assert norm_irregularity(1.0, irregularity_max=5) == 0.0
    assert norm_irregularity(3.0, irregularity_max=5) == 0.5
    assert norm_irregularity(50.0, irregularity_max=5) == 1.0


def test_severity_score_extremes():
    zero = severity_score(a=0, d=0, i=1.0, depth_max=10, irregularity_max=5)
    assert zero == 0.0

    max_score = severity_score(a=1, d=10, i=5.0, depth_max=10, irregularity_max=5)
    assert max_score == 100.0


def test_severity_score_respects_custom_weights():
    weights = FusionWeights(w_area=1.0, w_depth=0.0, w_irregularity=0.0)
    score = severity_score(a=0.5, d=999, i=999, weights=weights, depth_max=10, irregularity_max=5)
    assert score == 50.0  # only area contributes


def test_severity_category_thresholds():
    assert severity_category(10) == "Low"
    assert severity_category(30) == "Medium"
    assert severity_category(60) == "High"
    assert severity_category(90) == "Critical"
