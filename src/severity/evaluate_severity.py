"""Validate the Severity Score against human ratings (Phase 4 / Phase 6 Task 2).

Scores exactly the ground-truth pothole instances that were rated with
build_severity_rating.py (held-out TEST split), so severity is judged independently of
detection errors. Nothing here is fitted to the ratings: bounds come either from the
defaults or from cue_bounds.json fitted on the TRAIN split by fit_cue_bounds.py.

Compared configurations:
  legacy       -- the pre-revision score: raw MiDaS depth units with bound 30, raw
                  pixel-perimeter irregularity, bound 5
  revised      -- scale-normalized depth + robust irregularity, SeverityConfig() defaults
  revised_fit  -- same cues, bounds fitted on the train split (if cue_bounds.json exists)

Reports Spearman's rho (bootstrap 95% CI), Kendall's tau-b, per-cue rho, and agreement
between the five severity categories and the 1-5 ratings (exact, within one level,
quadratic-weighted kappa). Ratings of 0 ("can't judge") are excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import cohen_kappa_score

from depth_proxy import depth_proxy, depth_proxy_normalized
from fusion import LEGACY_RAW_DEPTH_MAX, SEVERITY_CATEGORIES, SeverityConfig, severity_category, severity_score
from geometric import area_ratio
from irregularity import contour_roughness, raw_contour_roughness

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBSET_DIR = REPO_ROOT / "data" / "severity_val_subset"
IMAGES_DIR = REPO_ROOT / "data" / "merged" / "images"
LABELS_DIR = REPO_ROOT / "data" / "merged" / "labels"
BOUNDS_JSON = REPO_ROOT / "evaluation" / "severity" / "cue_bounds.json"
OUT_DIR = REPO_ROOT / "evaluation" / "severity"
N_BOOTSTRAP = 2000


def instance_mask(image_name: str, instance: int, height: int, width: int) -> np.ndarray:
    lines = [l for l in (LABELS_DIR / f"{Path(image_name).stem}.txt").read_text(encoding="utf-8").splitlines() if len(l.split()) >= 7]
    poly = np.array(lines[instance].split()[1:], dtype=float).reshape(-1, 2) * [width, height]
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
    return mask


def bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray, n: int = N_BOOTSTRAP, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(x[idx])) > 1 and len(np.unique(y[idx])) > 1:
            values.append(spearmanr(x[idx], y[idx]).statistic)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def agreement(scores: np.ndarray, ratings: np.ndarray, thresholds) -> dict:
    predicted_level = np.array([SEVERITY_CATEGORIES.index(severity_category(s, thresholds=thresholds)) + 1 for s in scores])
    return {
        "exact": float(np.mean(predicted_level == ratings)),
        "within_one": float(np.mean(np.abs(predicted_level - ratings) <= 1)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(ratings, predicted_level, weights="quadratic", labels=[1, 2, 3, 4, 5])),
        "predicted_level_counts": {lvl: int((predicted_level == lvl).sum()) for lvl in range(1, 6)},
    }


def summarize(name: str, scores: np.ndarray, ratings: np.ndarray, thresholds) -> dict:
    rho = spearmanr(scores, ratings).statistic
    return {
        "config": name,
        "n": int(len(ratings)),
        "spearman_rho": float(rho),
        "spearman_rho_ci95": bootstrap_spearman_ci(scores, ratings),
        "kendall_tau_b": float(kendalltau(scores, ratings).statistic),
        "score_median": float(np.median(scores)),
        "score_max": float(np.max(scores)),
        "category_agreement": agreement(scores, ratings, thresholds),
    }


def main(ratings_csv: Path, out_dir: Path) -> dict:
    with open(SUBSET_DIR / "rating_set.csv", newline="", encoding="utf-8") as f:
        source_of = {r["rating_id"]: r["source"] for r in csv.DictReader(f)}
    with open(ratings_csv, newline="", encoding="utf-8") as f:
        rated = [r for r in csv.DictReader(f) if r["rating"] in {"1", "2", "3", "4", "5"}]

    revised = SeverityConfig()
    fitted = SeverityConfig.from_json(BOUNDS_JSON) if BOUNDS_JSON.exists() else None
    rows = []
    for r in rated:
        image = cv2.imread(str(IMAGES_DIR / r["image"]))
        h, w = image.shape[:2]
        mask = instance_mask(r["image"], int(r["instance"]), h, w)
        a = area_ratio(mask)
        d_raw, d_norm = depth_proxy(image, mask), depth_proxy_normalized(image, mask)
        i_raw, i_new = raw_contour_roughness(mask), contour_roughness(mask)
        row = {
            "rating_id": r["rating_id"], "image": r["image"], "source": source_of.get(r["rating_id"], ""), "rating": int(r["rating"]),
            "area_ratio": a, "depth_raw": d_raw, "depth_normalized": d_norm, "irregularity_raw": i_raw, "irregularity": i_new,
            "S_legacy": severity_score(a, d_raw, i_raw, depth_max=LEGACY_RAW_DEPTH_MAX, irregularity_max=5.0),
            "S_revised": severity_score(a, d_norm, i_new, config=revised),
        }
        if fitted is not None:
            row["S_revised_fit"] = severity_score(a, d_norm, i_new, config=fitted)
        rows.append(row)

    ratings = np.array([r["rating"] for r in rows])
    results = {
        "rater": "single rater (project author); no inter-rater agreement available",
        "excluded_cant_judge": sum(1 for _ in open(ratings_csv, encoding="utf-8")) - 1 - len(rows),
        "configs": [summarize("legacy", np.array([r["S_legacy"] for r in rows]), ratings, (20, 40, 60, 80)),
                    summarize("revised", np.array([r["S_revised"] for r in rows]), ratings, revised.category_thresholds)],
        "per_cue_spearman": {cue: float(spearmanr([r[cue] for r in rows], ratings).statistic)
                             for cue in ("area_ratio", "depth_raw", "depth_normalized", "irregularity_raw", "irregularity")},
    }
    if fitted is not None:
        results["configs"].append(summarize("revised_fit", np.array([r["S_revised_fit"] for r in rows]), ratings, fitted.category_thresholds))
        results["fitted_bounds"] = fitted.to_dict()

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "human_rating_per_instance.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "human_rating_eval.json").write_text(json.dumps(results, indent=2, default=list), encoding="utf-8")
    print(json.dumps(results, indent=2, default=list))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", default=str(SUBSET_DIR / "severity_ratings.csv"))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    main(Path(args.ratings), Path(args.out_dir))
