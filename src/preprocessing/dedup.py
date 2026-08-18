"""Perceptual-hash deduplication across the merged image pool.

RDD2022, the Roboflow pothole set, and Indian Roads were sourced independently and are
known to overlap (Roboflow/Kaggle pothole sets are frequently re-exports of the same
underlying photos). This scans a directory of images, groups near-duplicates by phash
Hamming distance, and keeps exactly one image per cluster.

Survivor priority within a cluster (highest first):
  1. source with an unambiguous, confirmed-open license (see docs/phase1/05_dataset_verification.md)
  2. higher resolution (more useful for area/depth severity cues)
  3. otherwise, alphabetically first path (deterministic, reproducible output)
"""

from __future__ import annotations

import csv
from pathlib import Path

import imagehash
from PIL import Image

# Matches the confirmed-license ordering from docs/phase1/05_dataset_verification.md —
# sources not listed here are assumed lower priority than any listed source.
SOURCE_PRIORITY = {
    "rdd2022": 0,
    "roboflow_pothole": 1,
    "indian_roads": 2,
    "pothole600": 3,
}

HAMMING_THRESHOLD = 5  # empirically standard for phash near-duplicate detection


def _priority(path: Path, source: str) -> tuple:
    with Image.open(path) as im:
        resolution = im.size[0] * im.size[1]
    return (SOURCE_PRIORITY.get(source, 99), -resolution, str(path))


def find_duplicate_clusters(image_paths: list[tuple[Path, str]]) -> list[list[tuple[Path, str]]]:
    """image_paths: list of (path, source_name). Returns clusters of size >= 2."""
    hashes: list[tuple[Path, str, imagehash.ImageHash]] = []
    for path, source in image_paths:
        try:
            with Image.open(path) as im:
                hashes.append((path, source, imagehash.phash(im)))
        except Exception as exc:  # corrupt/unreadable image — surfaced, not silently dropped
            print(f"[dedup] could not hash {path}: {exc}")

    clusters: list[list[tuple[Path, str]]] = []
    assigned: set[int] = set()

    for i in range(len(hashes)):
        if i in assigned:
            continue
        cluster = [(hashes[i][0], hashes[i][1])]
        assigned.add(i)
        for j in range(i + 1, len(hashes)):
            if j in assigned:
                continue
            if hashes[i][2] - hashes[j][2] <= HAMMING_THRESHOLD:
                cluster.append((hashes[j][0], hashes[j][1]))
                assigned.add(j)
        if len(cluster) > 1:
            clusters.append(cluster)

    return clusters


def deduplicate(image_paths: list[tuple[Path, str]], report_csv: Path) -> list[tuple[Path, str]]:
    """Returns the survivor list (one per near-duplicate cluster, all unique images kept
    as-is) and writes a CSV report of every dropped image and which survivor replaced it."""
    clusters = find_duplicate_clusters(image_paths)
    dropped: set[Path] = set()
    rows = []

    for cluster in clusters:
        survivor_path, survivor_source = min(cluster, key=lambda ps: _priority(*ps))
        for path, source in cluster:
            if path != survivor_path:
                dropped.add(path)
                rows.append(
                    {
                        "dropped_path": str(path),
                        "dropped_source": source,
                        "kept_path": str(survivor_path),
                        "kept_source": survivor_source,
                    }
                )

    Path(report_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dropped_path", "dropped_source", "kept_path", "kept_source"])
        writer.writeheader()
        writer.writerows(rows)

    survivors = [(p, s) for p, s in image_paths if p not in dropped]
    print(f"[dedup] {len(image_paths)} images -> {len(survivors)} survivors, {len(dropped)} dropped as duplicates")
    return survivors
