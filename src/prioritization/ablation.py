"""Weight-sensitivity ablation (Phase 4 Task 10) and ranking evaluation (Phase 6 Task 3):
compare default vs. traffic-heavy vs. severity-only weighting and show how the ranked
worklist reorders; also report Top-K precision for severity-Critical flags.

Severity scores come from real trained-pipeline output on real Pothole-600 test images
(src/severity/pipeline.py). Road-type weights come from real OSM Overpass lookups
(road_type.py) against a small set of real Hyderabad-area coordinates. What is NOT real:
this project has no live citizen-report GPS/timestamp data yet (Pothole-600 has no
geotags), so which of those real coordinates each pothole "sits at", and its
report-hour/recurrence-count, are assigned illustratively (seeded/deterministic, not
fabricated as if they were collected data) purely to exercise the full P formula across
a range of contexts. Re-run this once real citizen reports exist to get a real ablation.

Phase 6 Task 3 asks for Kendall's tau against "an expert/simulated ideal repair order."
This project has no real domain expert available to produce one; rather than fabricate an
"expert" ranking, the severity-only ranking is used as the simulated reference order (the
simplest defensible notion of repair urgency absent human judgment), and the tau between
it and the default-weighted order is reported -- see main()'s printed note.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "severity"))

from formula import priority_score, rank_potholes  # noqa: E402
from pipeline import compute_severity  # noqa: E402
from road_type import road_type_weight  # noqa: E402
from priority_schema import PriorityInputs, PriorityWeights  # noqa: E402
from traffic_recurrence import cluster_recurrence_counts, recurrence_factor, traffic_proxy  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_MANIFEST = REPO_ROOT / "data" / "annotations" / "splits" / "test.txt"
LABELS_DIR = REPO_ROOT / "data" / "merged" / "labels"
OUT_DIR = REPO_ROOT / "evaluation"

# Real Hyderabad-area coordinates (main-road, arterial, and residential areas), queried
# live against OSM at run time -- see module docstring for what's real vs. illustrative.
SAMPLE_COORDINATES = [
    (17.4483, 78.3915),  # Gachibowli
    (17.4239, 78.4738),  # Hyderabad city center
    (17.3850, 78.4867),  # Charminar area
    (17.4400, 78.3489),  # HITEC City
    (17.5040, 78.3968),  # Kompally
]


def _mask_from_label(label_path: Path, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not label_path.exists():
        return mask
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = [float(v) for v in line.split()[1:]]
        if len(parts) < 6:
            continue
        pts = np.array([(parts[i] * width, parts[i + 1] * height) for i in range(0, len(parts), 2)], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def build_priority_inputs_for_sample(
    n_images: int = 20, seed: int = 0
) -> tuple[dict[str, PriorityInputs], dict[str, str]]:
    image_paths = [Path(p) for p in TEST_MANIFEST.read_text(encoding="utf-8").splitlines() if p.strip()]
    rng = random.Random(seed)
    sample = rng.sample(image_paths, min(n_images, len(image_paths)))

    # assign each sampled image one of the real coordinates + a report hour, deterministically
    assigned_coords = [SAMPLE_COORDINATES[i % len(SAMPLE_COORDINATES)] for i in range(len(sample))]
    assigned_hours = [rng.randint(0, 23) for _ in sample]

    recurrence_counts = cluster_recurrence_counts(assigned_coords, radius_m=25.0)

    inputs_by_id: dict[str, PriorityInputs] = {}
    categories_by_id: dict[str, str] = {}
    for img_path, (lat, lon), hour, rec_count in zip(sample, assigned_coords, assigned_hours, recurrence_counts):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        mask = _mask_from_label(LABELS_DIR / f"{img_path.stem}.txt", *image.shape[:2])
        if mask.sum() == 0:
            continue

        score, category, _cues = compute_severity(image, mask, use_midas=True)
        rt_weight = road_type_weight(lat, lon)
        t_proxy = traffic_proxy(rt_weight, hour)
        rec_factor = recurrence_factor(rec_count)

        inputs_by_id[img_path.stem] = PriorityInputs(
            severity_score=score, road_type_weight=rt_weight, traffic_proxy=t_proxy, recurrence_factor=rec_factor
        )
        categories_by_id[img_path.stem] = category

    return inputs_by_id, categories_by_id


WEIGHT_CONFIGS = {
    "default": PriorityWeights(alpha=0.4, beta=0.3, gamma=0.2, delta=0.1),
    "severity_only": PriorityWeights(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0),
    "traffic_heavy": PriorityWeights(alpha=0.2, beta=0.2, gamma=0.5, delta=0.1),
}


def run_ablation(inputs_by_id: dict[str, PriorityInputs]) -> dict[str, list[tuple[str, float]]]:
    return {name: rank_potholes(inputs_by_id, weights) for name, weights in WEIGHT_CONFIGS.items()}


def compare_to_default(rankings: dict[str, list[tuple[str, float]]]) -> dict[str, dict]:
    default_order = [pid for pid, _ in rankings["default"]]
    default_rank = {pid: i for i, pid in enumerate(default_order)}
    top5_default = set(default_order[:5])

    comparison = {}
    for name, ranking in rankings.items():
        order = [pid for pid, _ in ranking]
        tau, _ = kendalltau([default_rank[pid] for pid in order], list(range(len(order))))
        top5_overlap = len(set(order[:5]) & top5_default)
        comparison[name] = {"kendall_tau_vs_default": float(tau), "top5_overlap_with_default": top5_overlap}
    return comparison


def topk_critical_precision(ranking: list[tuple[str, float]], categories_by_id: dict[str, str], k: int) -> float:
    """Of the top-k potholes by priority score, what fraction have severity category
    "Critical"? (Phase 6 Task 3.) "Critical" comes from severity_category() applied to a
    real computed severity score for a real image -- not a fabricated label -- but there
    is no independent expert/citizen ground truth for "should have been in the top k", so
    this measures internal consistency (does high priority correlate with high severity
    the way the formula intends), not accuracy against an external standard."""
    top_k = [pid for pid, _ in ranking[:k]]
    if not top_k:
        return float("nan")
    hits = sum(1 for pid in top_k if categories_by_id.get(pid) == "Critical")
    return hits / len(top_k)


def main(n_images: int = 60, seed: int = 0) -> None:
    inputs_by_id, categories_by_id = build_priority_inputs_for_sample(n_images, seed)
    print(f"[ablation] built priority inputs for {len(inputs_by_id)} real pothole instances")
    n_critical = sum(1 for c in categories_by_id.values() if c == "Critical")
    print(f"[ablation] {n_critical}/{len(categories_by_id)} real instances are severity-Critical")

    rankings = run_ablation(inputs_by_id)
    comparison = compare_to_default(rankings)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "prioritization_ablation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "rank", "pothole_id", "priority_score", "severity_category"])
        for name, ranking in rankings.items():
            for rank, (pid, score) in enumerate(ranking, start=1):
                writer.writerow([name, rank, pid, f"{score:.3f}", categories_by_id.get(pid, "")])

    with open(OUT_DIR / "prioritization_ablation_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "config",
                "kendall_tau_vs_default",
                "top5_overlap_with_default",
                "top5_critical_precision",
                "top10_critical_precision",
            ]
        )
        for name, stats in comparison.items():
            ranking = rankings[name]
            p5 = topk_critical_precision(ranking, categories_by_id, 5)
            p10 = topk_critical_precision(ranking, categories_by_id, 10)
            writer.writerow(
                [
                    name,
                    f"{stats['kendall_tau_vs_default']:.3f}",
                    stats["top5_overlap_with_default"],
                    f"{p5:.3f}",
                    f"{p10:.3f}",
                ]
            )

    print(f"[ablation] wrote {OUT_DIR / 'prioritization_ablation.csv'}")
    print(f"[ablation] wrote {OUT_DIR / 'prioritization_ablation_summary.csv'}")
    print(
        "[ablation] 'severity_only' row's kendall_tau_vs_default is also the tau between the "
        "default-weighted order and a severity-only reference order -- the closest thing this "
        "project has to a simulated 'ideal repair order' absent a real domain expert (see module docstring)."
    )
    for name, stats in comparison.items():
        ranking = rankings[name]
        p5 = topk_critical_precision(ranking, categories_by_id, 5)
        p10 = topk_critical_precision(ranking, categories_by_id, 10)
        print(
            f"  {name}: tau={stats['kendall_tau_vs_default']:.3f}, top5_overlap={stats['top5_overlap_with_default']}/5, "
            f"top5_critical_precision={p5:.3f}, top10_critical_precision={p10:.3f}"
        )


if __name__ == "__main__":
    main()
