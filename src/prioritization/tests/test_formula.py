import pytest

from formula import priority_score, rank_potholes
from schema import PriorityInputs, PriorityWeights


def test_priority_score_all_max_inputs_gives_100():
    inputs = PriorityInputs(severity_score=100, road_type_weight=1.0, traffic_proxy=1.0, recurrence_factor=1.0)
    assert priority_score(inputs) == pytest.approx(100.0)


def test_priority_score_all_zero_gives_zero():
    inputs = PriorityInputs(severity_score=0, road_type_weight=0.0, traffic_proxy=0.0, recurrence_factor=0.0)
    assert priority_score(inputs) == 0.0


def test_priority_score_severity_only_weighting():
    weights = PriorityWeights(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0)
    inputs = PriorityInputs(severity_score=80, road_type_weight=1.0, traffic_proxy=1.0, recurrence_factor=1.0)
    # only severity should matter -- road/traffic/recurrence being maxed out is ignored
    assert priority_score(inputs, weights) == pytest.approx(80.0)


def test_priority_score_unnormalized_weights_still_scale_to_100():
    # weights not summing to 1 -- P should still land on 0-100 by construction
    weights = PriorityWeights(alpha=2.0, beta=2.0, gamma=2.0, delta=2.0)
    inputs = PriorityInputs(severity_score=50, road_type_weight=0.5, traffic_proxy=0.5, recurrence_factor=0.5)
    assert priority_score(inputs, weights) == pytest.approx(50.0)


def test_priority_weights_reject_non_positive_total():
    with pytest.raises(ValueError):
        PriorityWeights(alpha=0.0, beta=0.0, gamma=0.0, delta=0.0)


def test_rank_potholes_orders_descending_by_priority():
    inputs_by_id = {
        "low": PriorityInputs(severity_score=10, road_type_weight=0.4, traffic_proxy=0.2, recurrence_factor=0.0),
        "high": PriorityInputs(severity_score=95, road_type_weight=1.0, traffic_proxy=0.9, recurrence_factor=0.8),
        "mid": PriorityInputs(severity_score=50, road_type_weight=0.7, traffic_proxy=0.5, recurrence_factor=0.3),
    }

    ranked = rank_potholes(inputs_by_id)

    assert [pid for pid, _ in ranked] == ["high", "mid", "low"]
    assert ranked[0][1] > ranked[1][1] > ranked[2][1]
