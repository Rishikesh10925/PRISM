"""Validates the calibration *procedure* against synthetic data with a known-true
weight configuration -- this proves calibrate() correctly recovers good weights and
computes rho correctly. It does NOT claim anything about real human severity ratings;
see calibrate.py's module docstring for why no real calibration exists yet.
"""

import numpy as np

from calibrate import calibrate
from fusion import severity_score, FusionWeights


def test_calibrate_recovers_high_rho_on_noise_free_synthetic_data():
    rng = np.random.default_rng(0)
    true_weights = FusionWeights(w_area=0.5, w_depth=0.3, w_irregularity=0.2)
    depth_max, irregularity_max = 20.0, 5.0

    cues = []
    ratings = []
    for _ in range(40):
        a = rng.uniform(0, 1)
        d = rng.uniform(0, depth_max)
        i = rng.uniform(1, irregularity_max)
        cues.append((a, d, i))

        true_s = severity_score(a, d, i, true_weights, depth_max, irregularity_max)
        ratings.append(1 + true_s / 100 * 4)  # noise-free 0-100 -> 1-5 rescale

    result = calibrate(cues, ratings, weight_step=0.1, depth_max_candidates=(20.0,), irregularity_max_candidates=(5.0,))

    assert result.spearman_rho >= 0.95  # should recover a near-perfect ranking on noise-free data
    assert result.depth_max == 20.0
    assert result.irregularity_max == 5.0


def test_calibrate_raises_on_mismatched_lengths():
    import pytest

    with pytest.raises(ValueError):
        calibrate(cues=[(0.1, 1.0, 1.5)], human_ratings_1_to_5=[3, 4])


def test_calibrate_raises_on_too_few_samples():
    import pytest

    with pytest.raises(ValueError):
        calibrate(cues=[(0.1, 1.0, 1.5), (0.2, 2.0, 2.0)], human_ratings_1_to_5=[3, 4])
