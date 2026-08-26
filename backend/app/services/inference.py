"""Wires the real trained pipeline (src/detection, src/severity, src/prioritization)
into the backend -- this is the same pipeline demo/app.py uses, just returning
structured data for persistence instead of rendering it directly.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
for sub in ("severity", "prioritization"):
    path = str(REPO_ROOT / "src" / sub)
    if path not in sys.path:
        sys.path.insert(0, path)

from ultralytics import YOLO  # noqa: E402

from pipeline import compute_severity  # noqa: E402
from formula import priority_category, priority_score  # noqa: E402
from priority_schema import PriorityInputs, PriorityWeights  # noqa: E402
from road_type import road_type_weight  # noqa: E402
from traffic_recurrence import recurrence_factor, traffic_proxy  # noqa: E402

from app.config import USE_MIDAS, YOLO_MODEL_PATH


@dataclass
class DetectionResult:
    class_name: str
    confidence: float
    mask_polygon: list[list[float]]
    bbox: list[float]
    severity_score: float
    severity_category: str
    area_ratio: float
    depth_value: float
    depth_source: str
    irregularity: float
    priority_score: float
    priority_category: str
    recurrence_factor: float
    weights: PriorityWeights


@lru_cache(maxsize=1)
def _load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def _mask_to_polygon(mask: np.ndarray) -> list[list[float]] | None:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return [[float(p[0][0]), float(p[0][1])] for p in largest]


def analyze_report_image(
    image_bgr: np.ndarray,
    latitude: float,
    longitude: float,
    report_hour: int | None = None,
    report_count: int = 1,
    weights: PriorityWeights | None = None,
) -> tuple[list[DetectionResult], float, float]:
    """Runs detection -> severity -> priority on one uploaded photo. Returns
    (detections, road_type_weight, traffic_proxy) -- the latter two are stored on the
    Report row since they're shared context for every detection in the same photo."""
    weights = weights or PriorityWeights()
    hour = report_hour if report_hour is not None else datetime.now().hour

    model = _load_model(YOLO_MODEL_PATH)
    results = model.predict(image_bgr, verbose=False)[0]

    rt_weight = road_type_weight(latitude, longitude)
    t_proxy = traffic_proxy(rt_weight, hour)
    rec_factor = recurrence_factor(report_count)

    detections: list[DetectionResult] = []
    if results.masks is not None:
        for i, mask_tensor in enumerate(results.masks.data):
            mask = cv2.resize(
                mask_tensor.cpu().numpy(),
                (image_bgr.shape[1], image_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            polygon = _mask_to_polygon(mask)
            if polygon is None:
                continue

            score, category, cues = compute_severity(image_bgr, mask, use_midas=USE_MIDAS)
            confidence = float(results.boxes.conf[i]) if results.boxes is not None else 0.0
            box = results.boxes.xyxy[i].tolist() if results.boxes is not None else [0, 0, 0, 0]

            p_score = priority_score(
                PriorityInputs(
                    severity_score=score,
                    road_type_weight=rt_weight,
                    traffic_proxy=t_proxy,
                    recurrence_factor=rec_factor,
                ),
                weights,
            )

            detections.append(
                DetectionResult(
                    class_name="pothole",
                    confidence=confidence,
                    mask_polygon=polygon,
                    bbox=[float(v) for v in box],
                    severity_score=score,
                    severity_category=category,
                    area_ratio=cues.area_ratio,
                    depth_value=cues.depth,
                    depth_source=cues.depth_source,
                    irregularity=cues.irregularity,
                    priority_score=p_score,
                    priority_category=priority_category(p_score),
                    recurrence_factor=rec_factor,
                    weights=weights,
                )
            )

    return detections, rt_weight, t_proxy
