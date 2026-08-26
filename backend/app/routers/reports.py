"""POST /api/report (Phase 5 Task 4): accept image + GPS, run the real detect ->
severity -> priority pipeline, persist and return the result.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from geoalchemy2.shape import from_shape
from shapely.geometry import Point
from sqlalchemy.orm import Session

from app.config import UPLOAD_DIR
from app.database import get_db
from app.models import Detection, PriorityScoreRecord, Report, SeverityScore
from app.schemas import ReportOut
from app.services.inference import analyze_report_image

router = APIRouter(prefix="/api", tags=["reports"])


@router.post("/report", response_model=ReportOut)
def create_report(
    db: Session = Depends(get_db),
    image: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
) -> Report:
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise HTTPException(status_code=422, detail="latitude/longitude out of range")

    image_bytes = image.file.read()
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=422, detail="could not decode image")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}_{Path(image.filename or 'upload.jpg').name}"
    stored_path = UPLOAD_DIR / stored_name
    stored_path.write_bytes(image_bytes)

    detections, rt_weight, t_proxy = analyze_report_image(image_bgr, latitude, longitude)

    report = Report(
        image_path=str(stored_path),
        location=from_shape(Point(longitude, latitude), srid=4326),
        submitted_at=datetime.now(timezone.utc),
        processed_at=datetime.now(timezone.utc),
        road_type_weight=rt_weight,
        traffic_proxy=t_proxy,
    )
    db.add(report)
    db.flush()  # assigns report.id without committing yet

    for det in detections:
        detection_row = Detection(
            report_id=report.id,
            class_name=det.class_name,
            confidence=det.confidence,
            mask_polygon=det.mask_polygon,
            bbox=det.bbox,
        )
        db.add(detection_row)
        db.flush()

        db.add(
            SeverityScore(
                detection_id=detection_row.id,
                score=det.severity_score,
                category=det.severity_category,
                area_ratio=det.area_ratio,
                depth_value=det.depth_value,
                depth_source=det.depth_source,
                irregularity=det.irregularity,
            )
        )
        db.add(
            PriorityScoreRecord(
                detection_id=detection_row.id,
                score=det.priority_score,
                category=det.priority_category,
                recurrence_factor=det.recurrence_factor,
                alpha=det.weights.alpha,
                beta=det.weights.beta,
                gamma=det.weights.gamma,
                delta=det.weights.delta,
            )
        )

    db.commit()
    db.refresh(report)
    return report
