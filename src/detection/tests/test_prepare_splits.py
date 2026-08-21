from pathlib import Path

from prepare_splits import _source_and_native_split, build_splits


def test_source_and_native_split_pothole600():
    assert _source_and_native_split(Path("pothole600_training_0000.png")) == ("pothole600_training", "train")
    assert _source_and_native_split(Path("pothole600_validation_0071.png")) == ("pothole600_validation", "val")
    assert _source_and_native_split(Path("pothole600_testing_0044.png")) == ("pothole600_testing", "test")


def test_source_and_native_split_no_native_split():
    source, split = _source_and_native_split(Path("rdd2022_Japan_000123.jpg"))
    assert split is None
    assert source == "rdd2022_Japan"  # trailing "_<original filename>" token stripped off as the source name


def test_build_splits_honors_native_split_and_stratifies_unassigned(tmp_path, monkeypatch):
    import prepare_splits as ps

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for i in range(24):
        (images_dir / f"pothole600_training_{i:04d}.png").touch()
    for i in range(6):
        (images_dir / f"pothole600_validation_{i:04d}.png").touch()
    for i in range(6):
        (images_dir / f"pothole600_testing_{i:04d}.png").touch()
    for i in range(20):
        (images_dir / f"othersource_{i:04d}.jpg").touch()

    splits_dir = tmp_path / "splits"
    monkeypatch.setattr(ps, "MERGED_IMAGES_DIR", images_dir)
    monkeypatch.setattr(ps, "SPLITS_DIR", splits_dir)

    manifest = build_splits(seed=1)

    # native-split sources: exact counts preserved regardless of train/val/test fractions
    assert sum(1 for p in manifest["train"] if "pothole600_training" in p) == 24
    assert sum(1 for p in manifest["val"] if "pothole600_validation" in p) == 6
    assert sum(1 for p in manifest["test"] if "pothole600_testing" in p) == 6

    # unassigned source: randomly split ~70/15/15 of its own 20 images
    other_total = sum(1 for split in manifest.values() for p in split if "othersource" in p)
    assert other_total == 20
    other_train = sum(1 for p in manifest["train"] if "othersource" in p)
    assert other_train == 14  # round(20 * 0.70)

    assert (splits_dir / "split_summary.csv").exists()
