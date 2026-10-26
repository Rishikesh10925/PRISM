"""Summarize the manual mask review exported by the build_mask_review.py page.

Reports rating shares separately for SAM pseudo-labels and the hidden human-drawn
Pothole-600 controls, with Wilson 95% confidence intervals -- the sample is small, so a
bare percentage would overstate how precisely SAM quality is known.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / "data" / "merged" / "review_sample" / "mask_review_results.csv"
DEFAULT_OUT = REPO_ROOT / "evaluation" / "data_quality" / "sam_mask_review_summary.json"
RATINGS = ("good", "acceptable", "bad")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def group_of(image_name: str) -> str:
    return "human_drawn_control" if image_name.startswith("pothole600") else "sam_pseudo_label"


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for group in ("sam_pseudo_label", "human_drawn_control"):
        rated = [r for r in rows if group_of(r["image"]) == group and r["rating"] in RATINGS]
        n = len(rated)
        counts = Counter(r["rating"] for r in rated)
        usable = counts["good"] + counts["acceptable"]
        flags = {f: sum(r[f].lower() == "true" for r in rated) for f in ("missed_pothole", "false_mask", "artifact")}
        summary[group] = {
            "num_rated": n,
            "counts": {r: counts[r] for r in RATINGS},
            "usable_share": usable / n if n else float("nan"),
            "usable_share_ci95": wilson_interval(usable, n),
            "good_share": counts["good"] / n if n else float("nan"),
            "good_share_ci95": wilson_interval(counts["good"], n),
            "flag_counts": flags,
        }
    summary["num_unrated"] = sum(1 for r in rows if r["rating"] not in RATINGS)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    with open(args.results, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    summary = summarize(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
