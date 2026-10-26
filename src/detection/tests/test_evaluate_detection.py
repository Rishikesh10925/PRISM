import numpy as np

from evaluate_detection import counts_at, greedy_match


def test_greedy_match_counts():
    # 3 predictions, 2 GT: pred0 matches gt0, pred1 overlaps gt0 too (duplicate -> FP), pred2 misses
    ious = np.array([[0.8, 0.0], [0.6, 0.1], [0.0, 0.3]])
    scores = np.array([0.9, 0.8, 0.7])
    tp, fp, fn, matched = greedy_match(ious, scores)
    assert (tp, fp, fn) == (1, 2, 1)
    assert matched == [0.8]


def test_higher_confidence_claims_gt_first():
    ious = np.array([[0.55], [0.9]])
    tp, fp, fn, matched = greedy_match(ious, np.array([0.95, 0.5]))
    assert (tp, fp, fn) == (1, 1, 0) and matched == [0.55]


def test_no_predictions_and_no_gt():
    assert greedy_match(np.zeros((0, 2)), np.array([]))[:3] == (0, 0, 2)
    assert greedy_match(np.zeros((2, 0)), np.array([0.9, 0.8]))[:3] == (0, 2, 0)


def test_counts_at_threshold_filters_low_confidence():
    records = [{"scores": np.array([0.9, 0.2]), "ious": np.array([[0.7], [0.0]])}]
    high = counts_at(records, 0.5)
    low = counts_at(records, 0.1)
    assert (high["tp"], high["fp"], high["fn"]) == (1, 0, 0) and high["f1"] == 1.0
    assert (low["tp"], low["fp"]) == (1, 1) and low["precision"] == 0.5
