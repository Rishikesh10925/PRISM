from pathlib import Path

from clean_annotations import auto_clean, sample_for_manual_review
from schema import ImageAnnotation, Instance


def test_auto_clean_drops_degenerate_and_out_of_bounds_instances():
    ann = ImageAnnotation(
        image_path="img.jpg",
        width=100,
        height=100,
        source="test",
        instances=[
            Instance(class_name="pothole", polygon=[(10, 10), (30, 10), (30, 30), (10, 30)]),  # valid
            Instance(class_name="pothole", polygon=[(5, 5), (5, 5)]),  # <3 points
            Instance(class_name="pothole", polygon=[(5, 5), (5.1, 5), (5, 5.1)]),  # near-zero area
            Instance(class_name="pothole", polygon=[(200, 200), (210, 200), (210, 210), (200, 210)]),  # out of bounds
        ],
    )

    cleaned, dropped = auto_clean([ann])

    assert dropped == 3
    assert len(cleaned[0].instances) == 1
    assert cleaned[0].instances[0].class_name == "pothole"


def test_sample_for_manual_review_writes_overlays_and_checklist(tmp_path: Path):
    import cv2
    import numpy as np

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    anns = []
    for i in range(10):
        path = images_dir / f"img{i}.jpg"
        cv2.imwrite(str(path), np.zeros((50, 50, 3), dtype=np.uint8))
        anns.append(
            ImageAnnotation(
                image_path=str(path),
                width=50,
                height=50,
                source="rdd2022",
                instances=[Instance(class_name="pothole", polygon=[(5, 5), (20, 5), (20, 20), (5, 20)])],
            )
        )

    out_dir = tmp_path / "review"
    sample_for_manual_review(anns, out_dir, fraction=0.3, seed=42)

    checklist = (out_dir / "review_checklist.csv").read_text(encoding="utf-8")
    lines = [l for l in checklist.splitlines() if l.strip()]
    assert len(lines) == 4  # header + 3 sampled (30% of 10, rounded)
    assert (out_dir / "overlays").exists()
    assert len(list((out_dir / "overlays").glob("*.jpg"))) == 3
