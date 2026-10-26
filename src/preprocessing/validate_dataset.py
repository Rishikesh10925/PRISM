"""Automated dataset validation and split freezing (Phase 2 hardening).

Checks every image in the train/val/test split manifests:
  - image exists and decodes; label file exists
  - every label row is a valid YOLO-seg polygon: known class id, >= 3 points, all
    normalized coordinates in [0, 1], non-degenerate area
  - no image appears in more than one split
  - near-duplicate images (pHash) that cross split boundaries (reported, not fatal --
    see docs/phase2/05_data_quality_report.md for how the known ones are handled)

`--freeze` records a SHA-256 of every split image and label into
evaluation/data_quality/split_manifest.csv (versioned in git, unlike data/ itself).
Without `--freeze`, the current files are verified against that frozen manifest, so any
later change to the test set -- a re-split, a re-labelled file, a replaced image -- fails
validation instead of silently changing what the reported metrics were measured on.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

from class_map import CLASSES

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "annotations" / "splits"
LABELS_DIR = REPO_ROOT / "data" / "merged" / "labels"
OUT_DIR = REPO_ROOT / "evaluation" / "data_quality"
SPLITS = ("train", "val", "test")

NEAR_DUP_HAMMING = 6
# Polygons below this fraction of the image are flagged (not rejected) as tiny instances.
TINY_INSTANCE_FRACTION = 0.0005

KNOWN_SOURCES = (
    "pothole600",
    "kaggle_annotated_potholes",
    "kaggle_potholes_yolov8",
    "kaggle_indian_roads",
    "kaggle_severity_levels",
)


def source_of(image_name: str) -> str:
    for source in KNOWN_SOURCES:
        if image_name.startswith(source):
            return source
    return "unknown"


def validate_label_text(text: str, width: int, height: int) -> tuple[list[str], int, int]:
    """Returns (errors, num_instances, num_tiny_instances) for one YOLO-seg label file."""
    errors: list[str] = []
    num_instances = 0
    num_tiny = 0
    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = float(parts[0])
            coords = np.array(parts[1:], dtype=float)
        except ValueError:
            errors.append(f"line {line_no}: non-numeric value")
            continue
        if not class_id.is_integer() or not 0 <= class_id < len(CLASSES):
            errors.append(f"line {line_no}: unknown class id {parts[0]}")
            continue
        if len(coords) < 6 or len(coords) % 2:
            errors.append(f"line {line_no}: polygon needs >= 3 (x, y) points, got {len(coords)} values")
            continue
        if coords.min() < 0.0 or coords.max() > 1.0:
            errors.append(f"line {line_no}: coordinate outside [0, 1]")
            continue
        points = (coords.reshape(-1, 2) * [width, height]).astype(np.float32)
        area = cv2.contourArea(points)
        if area <= 1.0:
            errors.append(f"line {line_no}: degenerate polygon (area {area:.2f} px)")
            continue
        num_instances += 1
        if area < TINY_INSTANCE_FRACTION * width * height:
            num_tiny += 1
    return errors, num_instances, num_tiny


def find_split_overlaps(splits: dict[str, list[Path]]) -> list[tuple[str, str, str]]:
    """(image_name, split_a, split_b) for every image listed in two splits."""
    seen: dict[str, str] = {}
    overlaps = []
    for split, paths in splits.items():
        for path in paths:
            if path.name in seen and seen[path.name] != split:
                overlaps.append((path.name, seen[path.name], split))
            seen.setdefault(path.name, split)
    return overlaps


def find_cross_split_near_duplicates(
    hashes: list[tuple[str, str, imagehash.ImageHash]], max_distance: int = NEAR_DUP_HAMMING
) -> list[tuple[str, str, str, str, int]]:
    """(split_a, name_a, split_b, name_b, distance) for near-duplicate pairs in different splits."""
    if not hashes:
        return []
    bits = np.array([h.hash.flatten() for _, _, h in hashes], dtype=np.uint8)
    pairs = []
    for i in range(len(hashes)):
        distances = (bits[i + 1 :] != bits[i]).sum(axis=1)
        for offset in np.nonzero(distances <= max_distance)[0]:
            j = i + 1 + int(offset)
            if hashes[i][0] != hashes[j][0]:
                pairs.append((hashes[i][0], hashes[i][1], hashes[j][0], hashes[j][1], int(distances[offset])))
    return pairs


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_splits(splits_dir: Path) -> dict[str, list[Path]]:
    return {
        split: [Path(p) for p in (splits_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines() if p.strip()]
        for split in SPLITS
    }


def validate(splits_dir: Path, labels_dir: Path, out_dir: Path, freeze: bool) -> dict:
    splits = load_splits(splits_dir)
    errors: list[str] = []
    manifest_rows = []
    hashes = []
    per_source = Counter()
    instances = Counter()
    tiny = Counter()
    empty_label_images = []

    for split, paths in splits.items():
        for image_path in paths:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not image_path.exists():
                errors.append(f"{image_path.name}: image file missing")
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                errors.append(f"{image_path.name}: image does not decode")
                continue
            if not label_path.exists():
                errors.append(f"{image_path.name}: label file missing")
                continue
            height, width = image.shape[:2]
            label_errors, n_inst, n_tiny = validate_label_text(label_path.read_text(encoding="utf-8"), width, height)
            errors.extend(f"{label_path.name}: {e}" for e in label_errors)
            if n_inst == 0 and not label_errors:
                empty_label_images.append(f"{split}/{image_path.name}")

            source = source_of(image_path.name)
            per_source[(split, source)] += 1
            instances[split] += n_inst
            tiny[split] += n_tiny
            with Image.open(image_path) as im:
                hashes.append((split, image_path.name, imagehash.phash(im.convert("RGB"))))
            manifest_rows.append(
                {
                    "split": split,
                    "source": source,
                    "image": image_path.name,
                    "width": width,
                    "height": height,
                    "num_instances": n_inst,
                    "image_sha256": sha256_of(image_path),
                    "label_sha256": sha256_of(label_path),
                }
            )

    overlaps = find_split_overlaps(splits)
    errors.extend(f"{name}: listed in both {a} and {b}" for name, a, b in overlaps)
    near_dups = find_cross_split_near_duplicates(hashes)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "split_manifest.csv"
    fieldnames = list(manifest_rows[0].keys()) if manifest_rows else []
    if freeze:
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)
    elif manifest_path.exists():
        errors.extend(compare_to_frozen_manifest(manifest_path, manifest_rows))
    else:
        errors.append(f"no frozen manifest at {manifest_path} -- run with --freeze once")

    with open(out_dir / "near_duplicates_cross_split.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split_a", "image_a", "split_b", "image_b", "phash_distance"])
        writer.writerows(near_dups)

    report = {
        "passed": not errors,
        "num_errors": len(errors),
        "errors": errors[:200],
        "split_sizes": {split: len(paths) for split, paths in splits.items()},
        "per_split_source_counts": {f"{split}/{source}": n for (split, source), n in sorted(per_source.items())},
        "instances_per_split": dict(instances),
        "tiny_instances_per_split": dict(tiny),
        "empty_label_images": empty_label_images,
        "split_overlaps": len(overlaps),
        "cross_split_near_duplicate_pairs": len(near_dups),
        "test_images_with_train_near_duplicate": sorted(
            {a if sa == "test" else b for sa, a, sb, b, _ in near_dups if {sa, sb} == {"train", "test"}}
        ),
    }
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def compare_to_frozen_manifest(manifest_path: Path, current_rows: list[dict]) -> list[str]:
    with open(manifest_path, newline="", encoding="utf-8") as f:
        frozen = {(row["split"], row["image"]): row for row in csv.DictReader(f)}
    current = {(row["split"], row["image"]): row for row in current_rows}
    errors = [f"{split}/{name}: in frozen manifest but no longer in split" for split, name in frozen.keys() - current.keys()]
    errors += [f"{split}/{name}: in split but not in frozen manifest" for split, name in current.keys() - frozen.keys()]
    for key in frozen.keys() & current.keys():
        for field in ("image_sha256", "label_sha256"):
            if frozen[key][field] != current[key][field]:
                errors.append(f"{key[0]}/{key[1]}: {field.split('_')[0]} file changed since freeze")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true", help="(re)write the frozen split manifest")
    args = parser.parse_args()

    report = validate(SPLITS_DIR, LABELS_DIR, OUT_DIR, freeze=args.freeze)
    summary = {k: v for k, v in report.items() if k not in ("errors", "per_split_source_counts")}
    print(json.dumps(summary, indent=2))
    for error in report["errors"][:20]:
        print(f"  ERROR {error}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
