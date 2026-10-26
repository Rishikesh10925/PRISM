"""Fit the severity cues' normalization bounds on a reference split (fusion.calibrate_bounds).

For every image in a split manifest, rasterizes each ground-truth YOLO-seg polygon
(labels/<image stem>.txt: "<class> x1 y1 x2 y2 ..." in normalized [0, 1] coords) into its
own instance mask, computes the raw cues with pipeline.compute_severity_cues, and sets each
cue's upper bound to the chosen percentile of its values. Writes a JSON that
SeverityConfig.from_json / compute_severity(config=<path>) load directly.

Intended use (by the project lead, on the TRAIN split only -- never test):

    E:\\PRISM\\.venv\\Scripts\\python.exe src\\severity\\fit_cue_bounds.py \\
        --manifest data\\annotations\\splits\\train.txt

Manifest formats: a .txt with one image path per line (data/annotations/splits/*.txt), or
a .csv with an `image` column of file names (+ optional `split` column filtered with
--split, e.g. evaluation/data_quality/split_manifest.csv), resolved against --images-dir.
MiDaS is on by default to match the pipeline; --no-midas fits the shadow-contrast bound
instead (fast, CPU-only). Bounds of cues with no usable values keep their defaults.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fusion import SeverityConfig, calibrate_bounds  # noqa: E402
from pipeline import compute_severity_cues  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS_DIR = REPO_ROOT / "data" / "merged" / "labels"
DEFAULT_IMAGES_DIR = REPO_ROOT / "data" / "merged" / "images"
DEFAULT_OUT = REPO_ROOT / "evaluation" / "severity" / "cue_bounds.json"


def instance_masks_from_yolo_label(label_path: Path, height: int, width: int) -> list[np.ndarray]:
    """One uint8 {0,1} mask per polygon line; lines with < 3 points are skipped."""
    masks = []
    if not label_path.exists():
        return masks
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        coords = np.asarray([float(v) for v in parts[1:]], dtype=np.float64)
        if coords.size % 2:
            coords = coords[:-1]
        pts = coords.reshape(-1, 2) * np.array([width, height], dtype=np.float64)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
        if mask.any():
            masks.append(mask)
    return masks


def read_manifest(manifest: Path, images_dir: Path = DEFAULT_IMAGES_DIR, split: str | None = None) -> list[Path]:
    if manifest.suffix.lower() == ".csv":
        with open(manifest, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows and "image" not in rows[0]:
            raise ValueError(f"{manifest} has no 'image' column")
        return [images_dir / r["image"] for r in rows if split is None or r.get("split") == split]
    paths = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            p = Path(line)
            paths.append(p if p.is_absolute() else images_dir / p)
    return paths


def fit(
    image_paths: list[Path],
    labels_dir: Path = DEFAULT_LABELS_DIR,
    use_midas: bool = True,
    percentile: float = 95.0,
    limit: int | None = None,
    min_samples: int = 5,
) -> tuple[SeverityConfig, dict]:
    all_cues = []
    n_images = 0
    skipped = 0
    for img_path in image_paths[:limit] if limit else image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            skipped += 1
            continue
        masks = instance_masks_from_yolo_label(labels_dir / f"{img_path.stem}.txt", *image.shape[:2])
        if not masks:
            skipped += 1
            continue
        n_images += 1
        for mask in masks:
            all_cues.append(compute_severity_cues(image, mask, use_midas=use_midas))

    config, report = calibrate_bounds(all_cues, percentile=percentile, min_samples=min_samples)
    report.update(
        n_images=n_images,
        n_instances=len(all_cues),
        n_skipped_images=skipped,
        use_midas=use_midas,
        depth_sources=sorted({c.depth_source for c in all_cues}),
    )
    return config, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, required=True, help="split manifest (.txt paths or .csv with 'image' column)")
    ap.add_argument("--split", default=None, help="for .csv manifests: keep only rows with this split value")
    ap.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    ap.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--percentile", type=float, default=95.0)
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (smoke runs)")
    ap.add_argument("--min-samples", type=int, default=5, help="min instances per cue to fit its bound")
    ap.add_argument("--no-midas", dest="use_midas", action="store_false", help="shadow-contrast depth (CPU only)")
    args = ap.parse_args(argv)

    paths = read_manifest(args.manifest, args.images_dir, args.split)
    if not paths:
        print(f"[fit_cue_bounds] no images in {args.manifest}", file=sys.stderr)
        return 1
    config, report = fit(paths, args.labels_dir, args.use_midas, args.percentile, args.limit, args.min_samples)
    if report["n_instances"] == 0:
        print("[fit_cue_bounds] no labelled instances found; nothing written", file=sys.stderr)
        return 1
    report["manifest"] = str(args.manifest)
    config.to_json(args.out, metadata=report)
    print(f"[fit_cue_bounds] {report['n_instances']} instances / {report['n_images']} images -> {args.out}")
    for key in ("area_ratio", "depth", "shadow_depth", "irregularity"):
        print(f"  {key}: {report[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
