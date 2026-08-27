"""Severity fusion ablation (Phase 6 Task 4): compare the full model (area+depth+
irregularity) against area-only, area+depth, and area+irregularity variants.

Real cues (a, d, i) are computed once per real test image via pipeline.py (MiDaS depth
included), then re-fused under each weight configuration -- so this isolates the effect
of which cues are included from any randomness in cue extraction itself.

What this ablation can and can't show: there is no severity validation subset with real
human ratings yet (see docs/phase4/01_severity_modules.md -- Phase 4 Task 6 is blocked on
that data, not done), so this can't report which config is "more accurate." What it can
report, honestly: how much each cue moves the score and category assignment relative to
the full model, on real images. Re-run with Spearman's rho against human ratings once the
validation subset exists (calibrate.py already implements that comparison).
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fusion import FusionWeights, severity_category, severity_score  # noqa: E402
from pipeline import compute_severity_cues  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_MANIFEST = REPO_ROOT / "data" / "annotations" / "splits" / "test.txt"
LABELS_DIR = REPO_ROOT / "data" / "merged" / "labels"
OUT_DIR = REPO_ROOT / "evaluation"

FUSION_CONFIGS = {
    "full (area+depth+irregularity)": FusionWeights(w_area=1 / 3, w_depth=1 / 3, w_irregularity=1 / 3),
    "area_only": FusionWeights(w_area=1.0, w_depth=0.0, w_irregularity=0.0),
    "area+depth": FusionWeights(w_area=0.5, w_depth=0.5, w_irregularity=0.0),
    "area+irregularity": FusionWeights(w_area=0.5, w_depth=0.0, w_irregularity=0.5),
}
REFERENCE_CONFIG = "full (area+depth+irregularity)"


def _mask_from_label(label_path: Path, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not label_path.exists():
        return mask
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or int(parts[0]) != 0:  # class 0 = pothole
            continue
        coords = [float(v) for v in parts[1:]]
        if len(coords) < 6:
            continue
        pts = np.array(
            [(coords[i] * width, coords[i + 1] * height) for i in range(0, len(coords), 2)], dtype=np.int32
        )
        cv2.fillPoly(mask, [pts], 1)
    return mask


def collect_real_cues(n_images: int = 100, seed: int = 0) -> dict[str, tuple[float, float, float]]:
    """Returns {image_stem: (area_ratio, depth, irregularity)} for real pothole instances."""
    image_paths = [Path(p) for p in TEST_MANIFEST.read_text(encoding="utf-8").splitlines() if p.strip()]
    rng = random.Random(seed)
    rng.shuffle(image_paths)

    cues_by_id: dict[str, tuple[float, float, float]] = {}
    for img_path in image_paths:
        if len(cues_by_id) >= n_images:
            break
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        mask = _mask_from_label(LABELS_DIR / f"{img_path.stem}.txt", *image.shape[:2])
        if mask.sum() == 0:
            continue
        cues = compute_severity_cues(image, mask, use_midas=True)
        cues_by_id[img_path.stem] = (cues.area_ratio, cues.depth, cues.irregularity)

    return cues_by_id


def run_ablation(cues_by_id: dict[str, tuple[float, float, float]]) -> dict[str, dict[str, tuple[float, str]]]:
    """Returns {config_name: {image_id: (score, category)}}."""
    results: dict[str, dict[str, tuple[float, str]]] = {}
    for config_name, weights in FUSION_CONFIGS.items():
        per_image = {}
        for image_id, (a, d, i) in cues_by_id.items():
            score = severity_score(a, d, i, weights)
            per_image[image_id] = (score, severity_category(score))
        results[config_name] = per_image
    return results


def compare_to_reference(results: dict[str, dict[str, tuple[float, str]]]) -> dict[str, dict]:
    ref = results[REFERENCE_CONFIG]
    image_ids = list(ref.keys())
    ref_scores = [ref[i][0] for i in image_ids]

    comparison = {}
    for name, per_image in results.items():
        scores = [per_image[i][0] for i in image_ids]
        rho, _ = spearmanr(ref_scores, scores)
        tau, _ = kendalltau(ref_scores, scores)
        mean_abs_diff = float(np.mean([abs(s - r) for s, r in zip(scores, ref_scores)]))
        category_agreement = float(
            np.mean([1.0 if per_image[i][1] == ref[i][1] else 0.0 for i in image_ids])
        )
        comparison[name] = {
            "spearman_rho_vs_full": float(rho),
            "kendall_tau_vs_full": float(tau),
            "mean_abs_score_diff_vs_full": mean_abs_diff,
            "category_agreement_vs_full": category_agreement,
        }
    return comparison


def main(n_images: int = 100, seed: int = 0) -> None:
    cues_by_id = collect_real_cues(n_images, seed)
    print(f"[severity ablation] computed real cues for {len(cues_by_id)} real pothole instances")

    results = run_ablation(cues_by_id)
    comparison = compare_to_reference(results)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "severity_fusion_ablation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "image_id", "score", "category"])
        for config_name, per_image in results.items():
            for image_id, (score, category) in per_image.items():
                writer.writerow([config_name, image_id, f"{score:.3f}", category])

    with open(OUT_DIR / "severity_fusion_ablation_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["config", "spearman_rho_vs_full", "kendall_tau_vs_full", "mean_abs_score_diff_vs_full", "category_agreement_vs_full"]
        )
        for name, stats in comparison.items():
            writer.writerow(
                [
                    name,
                    f"{stats['spearman_rho_vs_full']:.3f}",
                    f"{stats['kendall_tau_vs_full']:.3f}",
                    f"{stats['mean_abs_score_diff_vs_full']:.2f}",
                    f"{stats['category_agreement_vs_full']:.3f}",
                ]
            )

    print(f"[severity ablation] wrote {OUT_DIR / 'severity_fusion_ablation.csv'}")
    print(f"[severity ablation] wrote {OUT_DIR / 'severity_fusion_ablation_summary.csv'}")
    for name, stats in comparison.items():
        print(
            f"  {name}: rho={stats['spearman_rho_vs_full']:.3f} tau={stats['kendall_tau_vs_full']:.3f} "
            f"mean_abs_diff={stats['mean_abs_score_diff_vs_full']:.2f} "
            f"category_agreement={stats['category_agreement_vs_full']:.3f}"
        )


if __name__ == "__main__":
    main()
