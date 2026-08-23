from ablation import WEIGHT_CONFIGS, compare_to_default, run_ablation
from priority_schema import PriorityInputs


def test_run_ablation_produces_all_configured_rankings():
    inputs_by_id = {
        "a": PriorityInputs(severity_score=90, road_type_weight=0.4, traffic_proxy=0.2, recurrence_factor=0.0),
        "b": PriorityInputs(severity_score=20, road_type_weight=1.0, traffic_proxy=1.0, recurrence_factor=1.0),
        "c": PriorityInputs(severity_score=50, road_type_weight=0.7, traffic_proxy=0.5, recurrence_factor=0.3),
    }

    rankings = run_ablation(inputs_by_id)

    assert set(rankings.keys()) == set(WEIGHT_CONFIGS.keys())
    for ranking in rankings.values():
        assert {pid for pid, _ in ranking} == {"a", "b", "c"}


def test_severity_only_and_traffic_heavy_reorder_relative_to_default():
    # "a" is high-severity but low-context; "b" is low-severity but high-context --
    # designed so severity_only and traffic_heavy should disagree on the #1 spot
    inputs_by_id = {
        "a": PriorityInputs(severity_score=95, road_type_weight=0.4, traffic_proxy=0.1, recurrence_factor=0.0),
        "b": PriorityInputs(severity_score=15, road_type_weight=1.0, traffic_proxy=1.0, recurrence_factor=1.0),
    }

    rankings = run_ablation(inputs_by_id)

    assert rankings["severity_only"][0][0] == "a"
    assert rankings["traffic_heavy"][0][0] == "b"


def test_compare_to_default_reports_full_agreement_for_default_itself():
    inputs_by_id = {
        "a": PriorityInputs(severity_score=90, road_type_weight=0.4, traffic_proxy=0.2, recurrence_factor=0.0),
        "b": PriorityInputs(severity_score=20, road_type_weight=1.0, traffic_proxy=1.0, recurrence_factor=1.0),
    }
    rankings = run_ablation(inputs_by_id)

    comparison = compare_to_default(rankings)

    assert comparison["default"]["kendall_tau_vs_default"] == 1.0
    assert comparison["default"]["top5_overlap_with_default"] == 2
