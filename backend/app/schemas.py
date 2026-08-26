"""Pydantic request/response models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models import RepairStatus


class SeverityScoreOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    score: float
    category: str
    area_ratio: float
    depth_value: float
    depth_source: str
    irregularity: float


class PriorityScoreOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    score: float
    category: str
    recurrence_factor: float
    alpha: float
    beta: float
    gamma: float
    delta: float
    computed_at: datetime


class DetectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    class_name: str
    confidence: float
    mask_polygon: list
    bbox: list
    status: RepairStatus
    severity_score: SeverityScoreOut | None = None
    priority_score: PriorityScoreOut | None = None


class ReportOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    image_path: str
    latitude: float
    longitude: float
    submitted_at: datetime
    processed_at: datetime | None
    road_type_weight: float | None
    traffic_proxy: float | None
    detections: list[DetectionOut] = []


class PriorityListItem(BaseModel):
    """One row of GET /api/priority-list -- flattened for a worklist table, not the
    full nested report/detection structure."""

    detection_id: int
    report_id: int
    latitude: float
    longitude: float
    image_path: str
    severity_score: float
    severity_category: str
    priority_score: float
    priority_category: str
    status: RepairStatus
    submitted_at: datetime
    # raw components behind priority_score, so a client (e.g. the admin dashboard's
    # weight sliders, Phase 5 Task 10) can recompute the weighted combination locally
    # and re-rank instantly, without a round trip or re-running detection.
    road_type_weight: float
    traffic_proxy: float
    recurrence_factor: float


class StatusUpdate(BaseModel):
    status: RepairStatus
