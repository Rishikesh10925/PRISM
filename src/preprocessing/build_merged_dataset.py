"""Driver: run every available per-source converter, auto-clean, dedup, and write the
unified YOLO-seg dataset to data/merged/ + data/annotations/.

Only sources actually present under data/raw/ are processed — this is intentionally
tolerant of a partial data/raw/ (see docs/phase2/01_dataset_download_status.md for
which sources are currently blocked vs downloaded). Re-run any time data/raw/ gains a
new source; already-processed sources are simply re-converted, it's idempotent.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from clean_annotations import auto_clean, sample_for_manual_review
from convert_pothole600 import convert_pothole600_split
from dedup import deduplicate
from schema import ImageAnnotation
from write_yolo_seg import write_data_yaml, write_yolo_seg

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
MERGED_DIR = REPO_ROOT / "data" / "merged"
ANNOTATIONS_DIR = REPO_ROOT / "data" / "annotations"


def collect_all_annotations() -> list[ImageAnnotation]:
    annotations: list[ImageAnnotation] = []

    pothole600_dir = RAW_DIR / "pothole600"
    if pothole600_dir.exists():
        for split in ("training", "validation", "testing"):
            split_dir = pothole600_dir / split
            if split_dir.exists():
                anns = convert_pothole600_split(split_dir)
                for a in anns:
                    a.source = f"pothole600_{split}"  # keep source-native split visible downstream
                annotations.extend(anns)
                print(f"[build_merged_dataset] pothole600/{split}: {len(anns)} images")

    # Additional sources (RDD2022, Roboflow, Indian Roads) plug in here the same way
    # once they're downloaded — see docs/phase2/01_dataset_download_status.md.

    return annotations


def main() -> None:
    annotations = collect_all_annotations()
    if not annotations:
        print("[build_merged_dataset] no sources found under data/raw/ — nothing to do")
        return

    cleaned, dropped = auto_clean(annotations)
    print(f"[build_merged_dataset] auto_clean dropped {dropped} degenerate instances")

    image_paths = [(Path(a.image_path), a.source) for a in cleaned]
    survivors = deduplicate(image_paths, report_csv=MERGED_DIR / "dedup_report.csv")
    survivor_set = {str(p) for p, _ in survivors}
    cleaned = [a for a in cleaned if a.image_path in survivor_set]

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    (MERGED_DIR / "images").mkdir(exist_ok=True)
    for ann in cleaned:
        dest = MERGED_DIR / "images" / Path(ann.image_path).name
        if not dest.exists():
            shutil.copy2(ann.image_path, dest)
        ann.image_path = str(dest)

    write_yolo_seg(cleaned, ANNOTATIONS_DIR)
    write_data_yaml(ANNOTATIONS_DIR, ANNOTATIONS_DIR / "data.yaml")

    sample_for_manual_review(cleaned, MERGED_DIR / "review_sample", fraction=0.12)

    total_instances = sum(len(a.instances) for a in cleaned)
    print(f"[build_merged_dataset] done: {len(cleaned)} images, {total_instances} instances -> {MERGED_DIR}")


if __name__ == "__main__":
    main()
