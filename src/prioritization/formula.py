"""Prioritization formula (Phase 4 Task 7):
P = alpha*S + beta*RoadTypeWeight + gamma*TrafficProxy + delta*RecurrenceFactor
(blueprint Section 5.3).

Scale handling (an implementation choice the blueprint's formula doesn't pin down):
S lives on 0-100 while RoadTypeWeight/TrafficProxy/RecurrenceFactor live on 0-1, so S is
divided by 100 before the weighted sum -- otherwise severity would numerically dominate
regardless of the alpha/beta/gamma/delta ratios the admin dashboard's sliders are meant
to control (blueprint Section 4D). The weighted sum is then divided by
(alpha+beta+gamma+delta) and rescaled to 0-100, so P stays on the same 0-100 scale as S
and the four weights don't need to be pre-normalized to sum to 1 -- matching "expose as
admin sliders" (Section 5.3), where a user dragging one slider shouldn't require
renormalizing the others by hand.
"""

from __future__ import annotations

from schema import PriorityInputs, PriorityWeights


def priority_score(inputs: PriorityInputs, weights: PriorityWeights | None = None) -> float:
    weights = weights or PriorityWeights()

    weighted_sum = (
        weights.alpha * (inputs.severity_score / 100.0)
        + weights.beta * inputs.road_type_weight
        + weights.gamma * inputs.traffic_proxy
        + weights.delta * inputs.recurrence_factor
    )
    weight_total = weights.alpha + weights.beta + weights.gamma + weights.delta

    return float(100.0 * weighted_sum / weight_total)


def rank_potholes(
    inputs_by_id: dict[str, PriorityInputs], weights: PriorityWeights | None = None
) -> list[tuple[str, float]]:
    """Returns (pothole_id, P) pairs sorted by P descending -- the ranked repair
    worklist (blueprint Section 4F)."""
    weights = weights or PriorityWeights()
    scored = [(pid, priority_score(inp, weights)) for pid, inp in inputs_by_id.items()]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)
