"""POST /api/report (Phase 5 Task 4): accept image + GPS, run the real detect ->
severity -> priority pipeline, persist and return the result.

Hardening: the upload is validated by decoded content (size -> 413, type -> 415,
dimensions -> 422; app/services/uploads.py), the pipeline runs BEFORE anything is
written, and the photo is only stored once processing succeeded (and removed again if
the DB commit fails), so a failure never leaves a half-written report or orphaned file.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from geoalchemy2 import Geography
from geoalchemy2.shape import from_shape
from shapely.geometry import Point
from sqlalchemy import cast, func, select
from sqlalchemy.orm import Session

from app import config
from app.database import get_db
from app.models import Detection, PriorityScoreRecord, RepairStatus, Report, SeverityScore
from app.schemas import ReportOut
from app.services.inference import analyze_report_image
from app.services.uploads import UploadRejected, exif_capture_hour, safe_stored_name, validate_image_bytes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["reports"])

# Detection statuses that make an earlier report count as "the pothole is still there"
# for recurrence. RepairStatus has no "rejected" value; "repaired" is excluded, since a
# new report at a fixed spot is a new pothole, not a repeat of the old one.
RECURRENCE_ACTIVE_STATUSES = (RepairStatus.reported, RepairStatus.in_progress)


def count_earlier_reports_nearby(db: Session, latitude: float, longitude: float, radius_m: float) -> int:
    """Number of already-stored reports within radius_m metres (PostGIS ST_DWithin on
    geography -- Report.location is a geography column -- so the distance is true metres)
    that have at least one detection still reported/in_progress. Reports with no
    detections, or whose detections are all repaired, don't count. Called before the new
    report is inserted, so the current report is never part of its own count."""
    here = cast(func.ST_SetSRID(func.ST_MakePoint(longitude, latitude), 4326), Geography(srid=4326))
    stmt = (
        select(func.count(func.distinct(Report.id)))
        .join(Report.detections)
        .where(Detection.status.in_(RECURRENCE_ACTIVE_STATUSES))
        .where(func.ST_DWithin(Report.location, here, radius_m))
    )
    return int(db.execute(stmt).scalar_one())


@router.post("/report", response_model=ReportOut)
def create_report(
    db: Session = Depends(get_db),
    image: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
) -> Report:
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise HTTPException(status_code=422, detail="latitude/longitude out of range")

    image_bytes = image.file.read(config.MAX_UPLOAD_BYTES + 1)
    try:
        pil_image = validate_image_bytes(image_bytes, config.MAX_UPLOAD_BYTES)
    except UploadRejected as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    # hour for the (heuristic) traffic proxy: the photo's own capture time if it has a
    # plausible one, else the server's current local hour
    report_hour = exif_capture_hour(pil_image)
    time_source = "exif" if report_hour is not None else "server"
    if report_hour is None:
        report_hour = datetime.now().hour

    image_bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=422, detail="could not decode image")

    earlier_reports = count_earlier_reports_nearby(db, latitude, longitude, config.RECURRENCE_RADIUS_M)

    try:
        detections, rt_weight, rt_source, t_proxy = analyze_report_image(
            image_bgr, latitude, longitude, report_hour=report_hour, report_count=earlier_reports + 1
        )
    except Exception:
        logger.exception("detection/severity pipeline failed")
        raise HTTPException(status_code=500, detail="image analysis failed; the report was not saved")

    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored_path = config.UPLOAD_DIR / safe_stored_name(uuid.uuid4().hex, image.filename, pil_image.format)

    try:
        report = Report(
            image_path=str(stored_path),
            location=from_shape(Point(longitude, latitude), srid=4326),
            submitted_at=datetime.now(timezone.utc),
            processed_at=datetime.now(timezone.utc),
            road_type_weight=rt_weight,
            road_type_source=rt_source,
            traffic_proxy=t_proxy,
            time_source=time_source,
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

        stored_path.write_bytes(image_bytes)
        db.commit()
    except Exception:
        db.rollback()
        stored_path.unlink(missing_ok=True)
        logger.exception("failed to persist report")
        raise HTTPException(status_code=500, detail="could not save the report")

    db.refresh(report)
    logger.info(
        "report %s stored: %d detection(s), %d earlier nearby report(s), road_type_source=%s, time_source=%s",
        report.id, len(detections), earlier_reports, rt_source, time_source,
    )
    return report
