from pathlib import Path

from build_experiment_manifests import filter_manifest, max_instance_fraction

SMALL = "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2"  # 1% of the image
BIG = "0 0.0 0.0 0.6 0.0 0.6 0.6 0.0 0.6"  # 36% of the image


def test_max_instance_fraction():
    assert abs(max_instance_fraction(SMALL) - 0.01) < 1e-6
    assert abs(max_instance_fraction(SMALL + "\n" + BIG) - 0.36) < 1e-6
    assert max_instance_fraction("") == 0.0


def test_filter_drops_oversized_indian_roads_and_their_aug_copies(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "kaggle_indian_roads_A_1.txt").write_text(BIG)
    (labels / "kaggle_indian_roads_A_2.txt").write_text(SMALL)
    (labels / "kaggle_potholes_yolov8_x.txt").write_text(BIG)  # other sources are never dropped

    paths = [
        Path("img/kaggle_indian_roads_A_1.jpg"),
        Path("img/kaggle_indian_roads_A_1_aug0.jpg"),
        Path("img/kaggle_indian_roads_A_2.jpg"),
        Path("img/kaggle_potholes_yolov8_x.jpg"),
    ]
    kept, dropped = filter_manifest(paths, labels)
    assert [p.name for p in dropped] == ["kaggle_indian_roads_A_1.jpg", "kaggle_indian_roads_A_1_aug0.jpg"]
    assert [p.name for p in kept] == ["kaggle_indian_roads_A_2.jpg", "kaggle_potholes_yolov8_x.jpg"]
