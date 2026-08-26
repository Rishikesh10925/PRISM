"""GET /api/potholes, GET /api/priority-list (Phase 5 Task 5), and
PATCH /api/potholes/{id}/status (Phase 5 Task 6).
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.database import get_db
from app.models import Detection, RepairStatus, Report
from app.schemas import DetectionOut, PriorityListItem, StatusUpdate

router = APIRouter(prefix="/api", tags=["potholes"])


@router.get("/potholes", response_model=list[DetectionOut])
def list_potholes(
    status: RepairStatus | None = None,
    min_severity: float | None = None,
    db: Session = Depends(get_db),
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
        )
        for d in detections
        if d.severity_score and d.priority_score
    ]
    rows.sort(key=lambda r: r.priority_score, reverse=True)
    return rows


@router.patch("/potholes/{detection_id}/status", response_model=DetectionOut)
def update_status(detection_id: int, body: StatusUpdate, db: Session = Depends(get_db)) -> Detection:
    detection = db.get(Detection, detection_id)
    if detection is None:
        raise HTTPException(status_code=404, detail="detection not found")

    detection.status = body.status
    detection.status_updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(detection)
    return detection
