"""fit_cue_bounds.py end-to-end on 3 synthetic images in tmp_path (use_midas=False)."""

import json

import cv2
import numpy as np
import pytest

import fit_cue_bounds
from fusion import SeverityConfig
from pipeline import compute_severity


def _write_sample(images_dir, labels_dir, stem, polygons, size=(120, 160)):
    h, w = size
    image = np.full((h, w, 3), 190, dtype=np.uint8)
    lines = []
    for poly in polygons:
        pts = (np.array(poly) * [w, h]).astype(np.int32)
        cv2.fillPoly(image, [pts], (60, 60, 60))
        lines.append("0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in poly))
    cv2.imwrite(str(images_dir / f"{stem}.png"), image)
    (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return images_dir / f"{stem}.png"


@pytest.fixture
def dataset(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    paths = [
        _write_sample(images, labels, "a", [[(0.3, 0.5), (0.5, 0.5), (0.5, 0.8), (0.3, 0.8)]]),
        _write_sample(
            images,
            labels,
            "b",
            [[(0.1, 0.6), (0.4, 0.55), (0.3, 0.9)], [(0.6, 0.6), (0.9, 0.7), (0.7, 0.95), (0.65, 0.8)]],
        ),
        _write_sample(images, labels, "c", [[(0.2, 0.4), (0.8, 0.45), (0.7, 0.9), (0.25, 0.85)]]),
    ]
    manifest = tmp_path / "train.txt"
    manifest.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
    return tmp_path, manifest, labels


def test_instance_masks_from_yolo_label(dataset):
    _, _, labels = dataset
    masks = fit_cue_bounds.instance_masks_from_yolo_label(labels / "b.txt", 120, 160)
    assert len(masks) == 2 and all(m.sum() > 0 for m in masks)


def test_cli_writes_loadable_bounds(dataset):
    tmp_path, manifest, labels = dataset
    out = tmp_path / "evaluation" / "severity" / "cue_bounds.json"
    rc = fit_cue_bounds.main(
        [
            "--manifest", str(manifest), "--labels-dir", str(labels), "--out", str(out),
            "--no-midas", "--percentile", "90", "--min-samples", "2",
        ]
    )
    assert rc == 0 and out.exists()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["metadata"]["n_images"] == 3
    assert payload["metadata"]["n_instances"] == 4
    assert payload["metadata"]["depth_sources"] == ["shadow_heuristic"]

    cfg = SeverityConfig.from_json(out)
    default = SeverityConfig()
    assert cfg.area_max < default.area_max  # fitted to the small synthetic potholes
    assert cfg.shadow_depth_max != default.shadow_depth_max
    assert cfg.depth_max == default.depth_max  # no MiDaS values -> kept

    image = cv2.imread(str(tmp_path / "images" / "a.png"))
    mask = fit_cue_bounds.instance_masks_from_yolo_label(labels / "a.txt", *image.shape[:2])[0]
    score_from_path, _, _ = compute_severity(image, mask, use_midas=False, config=out)
    score_from_cfg, _, _ = compute_severity(image, mask, use_midas=False, config=cfg)
    score_default, _, _ = compute_severity(image, mask, use_midas=False)
    assert score_from_path == score_from_cfg
    assert 0.0 <= score_from_path <= 100.0
    assert score_from_path > score_default


def test_csv_manifest_with_split_filter(dataset, tmp_path):
    root, _, _ = dataset
    csv_path = tmp_path / "split_manifest.csv"
    csv_path.write_text("split,image\ntrain,a.png\ntrain,b.png\ntest,c.png\n", encoding="utf-8")
    paths = fit_cue_bounds.read_manifest(csv_path, root / "images", split="train")
    assert [p.name for p in paths] == ["a.png", "b.png"]


def test_empty_manifest_returns_error(tmp_path):
    manifest = tmp_path / "empty.txt"
    manifest.write_text("", encoding="utf-8")
    assert fit_cue_bounds.main(["--manifest", str(manifest), "--out", str(tmp_path / "x.json"), "--no-midas"]) == 1
    assert not (tmp_path / "x.json").exists()
