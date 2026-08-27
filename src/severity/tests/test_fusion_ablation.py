from fusion_ablation import FUSION_CONFIGS, REFERENCE_CONFIG, compare_to_reference, run_ablation


def test_run_ablation_produces_all_configured_scores():
    cues_by_id = {
        "a": (0.05, 10.0, 1.2),  # small, shallow, fairly round
        "b": (0.30, 25.0, 3.5),  # large, deep, jagged
    }

    results = run_ablation(cues_by_id)

    assert set(results.keys()) == set(FUSION_CONFIGS.keys())
    for per_image in results.values():
        assert set(per_image.keys()) == {"a", "b"}
        for score, category in per_image.values():
            assert 0.0 <= score <= 100.0
            assert category in {"Very Low", "Low", "Medium", "High", "Critical"}


def test_area_only_ignores_depth_and_irregularity_differences():
    # same area, wildly different depth/irregularity -- area_only must score them equal
    cues_by_id = {
        "a": (0.20, 0.0, 1.0),
        "b": (0.20, 30.0, 5.0),
    }

    results = run_ablation(cues_by_id)

    score_a, _ = results["area_only"]["a"]
    score_b, _ = results["area_only"]["b"]
    assert score_a == score_b


def test_compare_to_reference_reports_full_agreement_for_reference_itself():
    cues_by_id = {
        "a": (0.05, 10.0, 1.2),
        "b": (0.30, 25.0, 3.5),
        "c": (0.15, 5.0, 2.0),
    }
    results = run_ablation(cues_by_id)

    comparison = compare_to_reference(results)

    assert comparison[REFERENCE_CONFIG]["spearman_rho_vs_full"] == 1.0
    assert comparison[REFERENCE_CONFIG]["kendall_tau_vs_full"] == 1.0
    assert comparison[REFERENCE_CONFIG]["mean_abs_score_diff_vs_full"] == 0.0
    assert comparison[REFERENCE_CONFIG]["category_agreement_vs_full"] == 1.0
