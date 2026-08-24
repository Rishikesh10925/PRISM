from pathlib import Path

import numpy as np
from PIL import Image

from dedup import deduplicate


def _save_noise_image(path: Path, seed: int, size=(64, 64)):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_deduplicate_keeps_one_per_cluster_and_prefers_priority_source(tmp_path: Path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"  # near-identical copy of a (same content -> same phash)
    c = tmp_path / "c.jpg"  # unrelated image

    _save_noise_image(a, seed=1)
    Image.open(a).save(b)  # exact duplicate bytes/content, different path
    _save_noise_image(c, seed=2)

    image_paths = [
        (a, "roboflow_pothole"),
        (b, "rdd2022"),  # higher priority source, should survive over `a`
        (c, "roboflow_pothole"),
    ]

    survivors = deduplicate(image_paths, report_csv=tmp_path / "report.csv")
    survivor_paths = {p for p, _ in survivors}

    assert c in survivor_paths  # unrelated image always kept
    assert len(survivors) == 2  # one of {a, b} dropped as a duplicate of the other
    assert b in survivor_paths  # rdd2022 (higher priority) wins over roboflow_pothole
    assert a not in survivor_paths

    report_text = (tmp_path / "report.csv").read_text(encoding="utf-8")
    assert "a.jpg" in report_text and "b.jpg" in report_text


def test_deduplicate_excludes_unreadable_paths_from_survivors(tmp_path: Path):
    # regression test: a path that fails to open (missing file, or -- as happened with
    # the real kaggle_severity_levels source -- a VOC XML <filename> that doesn't match
    # what's actually on disk) must never end up in the survivor list. It was never
    # added to any duplicate cluster, so the old code let it pass through untouched,
    # and a downstream shutil.copy2() on that path crashed the whole merge.
    real = tmp_path / "real.jpg"
    _save_noise_image(real, seed=1)
    missing = tmp_path / "does_not_exist.jpg"

    image_paths = [(real, "kaggle_severity_levels"), (missing, "kaggle_severity_levels")]

    survivors = deduplicate(image_paths, report_csv=tmp_path / "report.csv")
    survivor_paths = {p for p, _ in survivors}

    assert real in survivor_paths
    assert missing not in survivor_paths

    report_text = (tmp_path / "report.csv").read_text(encoding="utf-8")
    assert "does_not_exist.jpg" in report_text
    assert "UNREADABLE" in report_text
