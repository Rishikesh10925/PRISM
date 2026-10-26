"""GET /api/potholes, GET /api/priority-list (Phase 5 Task 5),
PATCH /api/potholes/{id}/status (Phase 5 Task 6), and the admin detail view's
GET /api/potholes/{id} and GET /api/reports/{id}/image.
"""

from __future__ import annotations

import mimetypes
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app import config
from app.auth import get_current_admin
from app.database import get_db
from app.models import Detection, RepairStatus, Report
from app.schemas import DetectionDetailOut, DetectionOut, PriorityListItem, ReportSummaryOut, StatusUpdate

router = APIRouter(prefix="/api", tags=["potholes"])


@router.get("/potholes", response_model=list[DetectionOut])
def list_potholes(
    status: RepairStatus | None = None,
    min_severity: float | None = None,
    db: Session = Depends(get_db),
    admin: str = Depends(get_current_admin),
) -> list[Detection]:
    """List/filter detections by status and/or minimum severity score."""
    query = select(Detection).options(joinedload(Detection.severity_score), joinedload(Detection.priority_score))
    if status is not None:
        query = query.where(Detection.status == status)
    detections = list(db.execute(query).unique().scalars().all())

    if min_severity is not None:
        detections = [d for d in detections if d.severity_score and d.severity_score.score >= min_severity]

    return detections


@router.get("/priority-list", response_model=list[PriorityListItem])
def priority_list(
    status: RepairStatus | None = None,
    db: Session = Depends(get_db),
    admin: str = Depends(get_current_admin),
) -> list[PriorityListItem]:
    """The ranked repair worklist -- every detection with a priority score, sorted
    highest priority first."""
    query = (
        select(Detection)
        .join(Detection.report)
        .options(
            joinedload(Detection.severity_score), joinedload(Detection.priority_score), joinedload(Detection.report)
        )
    )
    if status is not None:
        query = query.where(Detection.status == status)

    detections = list(db.execute(query).unique().scalars().all())
    rows = [
        PriorityListItem(
            detection_id=d.id,
            report_id=d.report_id,
            latitude=d.report.latitude,
            longitude=d.report.longitude,
            image_path=d.report.image_path,
            severity_score=d.severity_score.score,
            severity_category=d.severity_score.category,
            priority_score=d.priority_score.score,
            priority_category=d.priority_score.category,
            status=d.status,
            submitted_at=d.report.submitted_at,
            road_type_weight=d.report.road_type_weight or 0.0,
            traffic_proxy=d.report.traffic_proxy or 0.0,
            recurrence_factor=d.priority_score.recurrence_factor,
        )
        for d in detections
        if d.severity_score and d.priority_score
    ]
    rows.sort(key=lambda r: r.priority_score, reverse=True)
    return rows


@router.patch("/potholes/{detection_id}/status", response_model=DetectionOut)
def update_status(
    detection_id: int,
    body: StatusUpdate,
    db: Session = Depends(get_db),
    admin: str = Depends(get_current_admin),
) -> Detection:
    detection = db.get(Detection, detection_id)
    if detection is None:
        raise HTTPException(status_code=404, detail="detection not found")

    detection.status = body.status
    detection.status_updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(detection)
    return detection


def _stored_image_path(report: Report) -> Path | None:
    """The report's photo, only if it exists inside UPLOAD_DIR -- image_path comes from
    the database, so never serve a path that resolves anywhere else."""
    path = Path(report.image_path).resolve()
    if not path.is_file() or not path.is_relative_to(Path(config.UPLOAD_DIR).resolve()):
        return None
    return path


@router.get("/potholes/{detection_id}", response_model=DetectionDetailOut)
def get_pothole(
    detection_id: int,
    db: Session = Depends(get_db),
    admin: str = Depends(get_current_admin),
) -> DetectionDetailOut:
    """One detection with its scores and parent report, for the admin detail view."""
    detection = db.get(
        Detection,
        detection_id,
        options=[joinedload(Detection.severity_score), joinedload(Detection.priority_score), joinedload(Detection.report)],
    )
    if detection is None:
        raise HTTPException(status_code=404, detail="detection not found")

    report = detection.report
    width = height = None
    image_path = _stored_image_path(report)
    if image_path is not None:
        with Image.open(image_path) as image:
            width, height = image.size

    return DetectionDetailOut(
        **DetectionOut.model_validate(detection).model_dump(),
        report=ReportSummaryOut(
            id=report.id,
            latitude=report.latitude,
            longitude=report.longitude,
            submitted_at=report.submitted_at,
            road_type_weight=report.road_type_weight,
            road_type_source=report.road_type_source,
            traffic_proxy=report.traffic_proxy,
            time_source=report.time_source,
            image_width=width,
            image_height=height,
        ),
    )


@router.get("/reports/{report_id}/image")
def get_report_image(
    report_id: int,
    db: Session = Depends(get_db),
    admin: str = Depends(get_current_admin),
) -> FileResponse:
    report = db.get(Report, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="report not found")
    image_path = _stored_image_path(report)
    if image_path is None:
        raise HTTPException(status_code=404, detail="report image not available")
    media_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    return FileResponse(image_path, media_type=media_type)
