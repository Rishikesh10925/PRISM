"""Driver: run every available per-source converter, auto-clean, dedup, and write the
unified YOLO-seg dataset to data/merged/ + data/annotations/.

Only sources actually present under data/raw/ are processed — this is intentionally
tolerant of a partial data/raw/ (see docs/phase2/01_dataset_download_status.md for
which sources are currently blocked vs downloaded). Re-run any time data/raw/ gains a
new source; already-processed sources are simply re-converted, it's idempotent.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from box_to_mask_sam import convert_boxes_to_masks
from clean_annotations import auto_clean, sample_for_manual_review
from convert_pothole600 import convert_pothole600_split
from convert_voc import convert_voc_dir
from convert_yolo_box import convert_yolo_box_dir
from dedup import deduplicate
from schema import ImageAnnotation
from write_yolo_seg import write_data_yaml, write_yolo_seg

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
MERGED_DIR = REPO_ROOT / "data" / "merged"
ANNOTATIONS_DIR = REPO_ROOT / "data" / "annotations"


def _subsample(anns: list[ImageAnnotation], max_count: int | None, seed: int = 0) -> list[ImageAnnotation]:
    """Box-only sources need a slow SAM pass (see box_to_mask_sam.py) before they're
    usable for training -- at ~2s/image, running all ~7800 box-only images downloaded
    this round would take hours. max_count caps each new source to a random subsample
    (seeded, so it's reproducible) for a faster first integration pass; pass
    max_per_new_source=None to process everything on a later, longer run."""
    if max_count is None or len(anns) <= max_count:
        return anns
    return random.Random(seed).sample(anns, max_count)


def collect_all_annotations(max_per_new_source: int | None = 500) -> list[ImageAnnotation]:
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

    # kaggle_annotated_potholes (D2): images + VOC XML together in one folder
    d2_dir = RAW_DIR / "kaggle_annotated_potholes" / "annotated-images"
    if d2_dir.exists():
        anns = _subsample(convert_voc_dir(d2_dir, d2_dir, source="kaggle_annotated_potholes"), max_per_new_source)
        annotations.extend(anns)
        print(f"[build_merged_dataset] kaggle_annotated_potholes: {len(anns)} images")

    # kaggle_potholes_yolov8 (D3): its own train/valid folders, merged into one pool
    # (not treated as a leakage-sensitive native split like Pothole-600's video frames —
    # these are independently curated stills, not sequential frames)
    d3_dir = RAW_DIR / "kaggle_potholes_yolov8"
    if d3_dir.exists():
        d3_anns: list[ImageAnnotation] = []
        for split in ("train", "valid"):
            split_dir = d3_dir / split
            if (split_dir / "images").exists():
                d3_anns.extend(
                    convert_yolo_box_dir(
                        split_dir / "images", split_dir / "labels", source="kaggle_potholes_yolov8", source_classes=["pothole"]
                    )
                )
        d3_anns = _subsample(d3_anns, max_per_new_source)
        annotations.extend(d3_anns)
        print(f"[build_merged_dataset] kaggle_potholes_yolov8: {len(d3_anns)} images")

    # kaggle_severity_levels: images + VOC XML in separate folders; severity labels
    # (minor/medium/major) all map to "pothole" for detection purposes here -- the
    # severity values themselves are used separately by src/severity/, not through this
    # detection-training pipeline.
    severity_dir = RAW_DIR / "kaggle_severity_levels"
    if (severity_dir / "images").exists():
        anns = _subsample(
            convert_voc_dir(severity_dir / "images", severity_dir / "annotations", source="kaggle_severity_levels"),
            max_per_new_source,
        )
        annotations.extend(anns)
        print(f"[build_merged_dataset] kaggle_severity_levels: {len(anns)} images")

    # kaggle_indian_roads: flat folder, images + YOLO txt together, 3 classes (only
    # "pothole" mapped in -- see docs/phase2/03_kaggle_indian_roads_class_mapping.md)
    indian_roads_dir = RAW_DIR / "kaggle_indian_roads" / "Dataset3Class"
    if indian_roads_dir.exists():
        indian_roads_anns = convert_yolo_box_dir(
            indian_roads_dir,
            indian_roads_dir,
            source="kaggle_indian_roads",
            source_classes=["speed_breaker", "pothole", "unpaved_road"],
        )
        # most images here also contain speed_breaker/unpaved_road boxes, which are
        # dropped (unmapped) -- filter to images with a mapped instance left BEFORE
        # subsampling, or a chunk of the subsample would be wasted on empty images
        indian_roads_anns = [a for a in indian_roads_anns if a.instances]
        anns = _subsample(indian_roads_anns, max_per_new_source)
        annotations.extend(anns)
        print(f"[build_merged_dataset] kaggle_indian_roads: {len(anns)} images")

    # RDD2022 and Roboflow Universe remain blocked -- see
    # docs/phase2/01_dataset_download_status.md. Plug in the same way once downloaded.

    return annotations


def main() -> None:
    annotations = collect_all_annotations()
    if not annotations:
        print("[build_merged_dataset] no sources found under data/raw/ — nothing to do")
        return

    cleaned, dropped = auto_clean(annotations)
    print(f"[build_merged_dataset] auto_clean dropped {dropped} degenerate instances")

    needs_mask_count = sum(1 for a in cleaned for inst in a.instances if inst.needs_mask)
    if needs_mask_count:
        print(f"[build_merged_dataset] running SAM on {needs_mask_count} box-only instances...")
        cleaned = convert_boxes_to_masks(cleaned, manifest_csv=MERGED_DIR / "sam_conversion_manifest.csv")

    image_paths = [(Path(a.image_path), a.source) for a in cleaned]
    survivors = deduplicate(image_paths, report_csv=MERGED_DIR / "dedup_report.csv")
    survivor_set = {str(p) for p, _ in survivors}
    cleaned = [a for a in cleaned if a.image_path in survivor_set]

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    (MERGED_DIR / "images").mkdir(exist_ok=True)
    for ann in cleaned:
        # Prefix with source: several sources (e.g. Pothole-600's training/validation/
        # testing folders) reuse the same bare filenames like 0000.png, and a flat
        # merge without this prefix silently collides them (one image's pixels ending
        # up paired with a different image's label file).
        original = Path(ann.image_path)
        dest = MERGED_DIR / "images" / f"{ann.source}_{original.name}"
        if not dest.exists():
            shutil.copy2(original, dest)
        ann.image_path = str(dest)

    # Labels go in data/merged/labels/, a sibling of data/merged/images/ — Ultralytics
    # locates an image's label by textually swapping "/images/" for "/labels/" in its
    # path, so the two must live side by side under the same parent (data/annotations/
    # holds the split manifests and data.yaml instead, see prepare_splits.py).
    write_yolo_seg(cleaned, MERGED_DIR)
    # data.yaml gets its real train/val/test manifest paths from
    # src/detection/prepare_splits.py, which runs after this and knows the splits;
    # this placeholder just means "not yet split" if someone inspects it in between.
    write_data_yaml(ANNOTATIONS_DIR / "data.yaml")

    sample_for_manual_review(cleaned, MERGED_DIR / "review_sample", fraction=0.12)

    total_instances = sum(len(a.instances) for a in cleaned)
    print(f"[build_merged_dataset] done: {len(cleaned)} images, {total_instances} instances -> {MERGED_DIR}")


if __name__ == "__main__":
    main()
