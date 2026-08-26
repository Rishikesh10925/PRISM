// Mirrors backend/../src/prioritization/formula.py's priority_score()/priority_category()
// exactly, so the admin dashboard's weight sliders (Phase 5 Task 10) can re-rank the
// worklist instantly client-side, without a round trip or re-running detection. Keep
// this in sync with formula.py if the scoring logic there ever changes.

export function recomputePriority(item, weights) {
  const { alpha, beta, gamma, delta } = weights;
  const weightTotal = alpha + beta + gamma + delta;
  if (weightTotal <= 0) return 0;

  const weightedSum =
    alpha * (item.severity_score / 100) +
    beta * item.road_type_weight +
    gamma * item.traffic_proxy +
    delta * item.recurrence_factor;

  return (100 * weightedSum) / weightTotal;
}

export function priorityCategory(score) {
  if (score < 25) return "Less Important";
  if (score < 50) return "Moderate";
  if (score < 75) return "Important";
  return "Very Important";
}
