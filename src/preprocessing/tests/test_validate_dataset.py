from pathlib import Path

import cv2
import numpy as np

from validate_dataset import (
    find_split_overlaps,
    source_of,
    validate,
    validate_label_text,
)

SQUARE = "0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"


def test_valid_polygon_passes():
    errors, n, tiny = validate_label_text(SQUARE, 100, 100)
    assert errors == [] and n == 1 and tiny == 0


def test_label_errors_are_reported():
    text = "\n".join(
        [
            "9 0.2 0.2 0.8 0.2 0.8 0.8",  # unknown class
            "0 0.2 0.2 0.8 0.2",  # only 2 points
            "0 0.2 0.2 1.3 0.2 0.8 0.8",  # out of range
            "0 0.5 0.5 0.5 0.5 0.5 0.5",  # zero area
            "0 a b c d e f",  # non-numeric
        ]
    )
    errors, n, _ = validate_label_text(text, 100, 100)
    assert len(errors) == 5 and n == 0


def test_tiny_instance_flagged_not_rejected():
    errors, n, tiny = validate_label_text("0 0.50 0.50 0.51 0.50 0.51 0.51 0.50 0.51", 1000, 1000)
    assert errors == [] and n == 1 and tiny == 1


def test_empty_label_is_valid():
    assert validate_label_text("", 100, 100) == ([], 0, 0)


def test_split_overlap_detected():
    splits = {"train": [Path("a.jpg"), Path("b.jpg")], "val": [Path("c.jpg")], "test": [Path("b.jpg")]}
    assert find_split_overlaps(splits) == [("b.jpg", "train", "test")]


def test_source_of():
    assert source_of("kaggle_indian_roads_PotHoles_12.jpg") == "kaggle_indian_roads"
    assert source_of("pothole600_testing_0001.png") == "pothole600"
    assert source_of("mystery.jpg") == "unknown"


def _make_dataset(root: Path) -> tuple[Path, Path]:
    images, labels, splits = root / "images", root / "labels", root / "splits"
    for d in (images, labels, splits):
        d.mkdir()
    rng = np.random.default_rng(0)
    manifests = {"train": [], "val": [], "test": []}
    for i, split in enumerate(["train", "train", "val", "test"]):
        path = images / f"pothole600_x_{i}.png"
        cv2.imwrite(str(path), rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        (labels / f"{path.stem}.txt").write_text(SQUARE)
        manifests[split].append(str(path))
    for split, paths in manifests.items():
        (splits / f"{split}.txt").write_text("\n".join(paths))
    return splits, labels


def test_freeze_then_verify_detects_changed_label(tmp_path):
    splits, labels = _make_dataset(tmp_path)
    out = tmp_path / "out"

    assert validate(splits, labels, out, freeze=True)["passed"]
    assert validate(splits, labels, out, freeze=False)["passed"]

    (labels / "pothole600_x_3.txt").write_text("0 0.1 0.1 0.9 0.1 0.9 0.9")
    report = validate(splits, labels, out, freeze=False)
    assert not report["passed"]
    assert any("test/pothole600_x_3.png: label file changed" in e for e in report["errors"])


def test_verify_without_frozen_manifest_fails(tmp_path):
    splits, labels = _make_dataset(tmp_path)
    report = validate(splits, labels, tmp_path / "out", freeze=False)
    assert not report["passed"]
