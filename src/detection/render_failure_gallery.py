"""Renders annotated prediction images for the specific failure cases picked by
find_failure_cases.py, for docs/phase6's failure-case gallery."""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = REPO_ROOT / "data" / "merged" / "images"
MODEL_PATH = REPO_ROOT / "models" / "yolov8n_seg_multisource.pt"
OUT_DIR = REPO_ROOT / "evaluation" / "failure_gallery"

CASES = [
    "kaggle_potholes_yolov8_pothole_103.jpg",  # severe under-detection in dense clutter (55 labels, 25 preds)
    "kaggle_annotated_potholes_img-428.jpg",  # over-detection / duplicate boxes (6 labels, 15 preds)
    "kaggle_indian_roads_PotHoles_205.jpg",  # darkest test image (brightness 15.3)
    "kaggle_indian_roads_PotHoles_699.jpg",  # lowest-confidence low-light detection (mean_conf 0.343)
]


def main() -> None:
    model = YOLO(str(MODEL_PATH))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename in CASES:
        image_path = IMAGES_DIR / filename
        result = model.predict(str(image_path), verbose=False)[0]
        out_path = OUT_DIR / filename
        result.save(filename=str(out_path))
        n = len(result.boxes) if result.boxes is not None else 0
        print(f"[render_failure_gallery] {filename}: {n} detections -> {out_path}")


if __name__ == "__main__":
    main()
