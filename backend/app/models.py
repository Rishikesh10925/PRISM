"""Database schema (Phase 5 Task 1): reports, detections, severity scores, priority
scores, and repair status.

A citizen "report" is one photo + GPS submission. Detection runs YOLOv8-seg on it,
producing zero or more "detections" (one row per pothole found in that photo). Each
detection gets exactly one severity score and one priority score. Repair status lives
on the detection, since a single photo can contain multiple potholes at different
stages of repair.
"""

from __future__ import annotations

import enum
from datetime import datetime

from geoalchemy2 import Geography
from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class RepairStatus(str, enum.Enum):
    reported = "reported"
    in_progress = "in_progress"
    repaired = "repaired"


class Report(Base):
    """One citizen photo+GPS submission."""

    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_path: Mapped[str] = mapped_column(String, nullable=False)
    location: Mapped[str] = mapped_column(Geography(geometry_type="POINT", srid=4326), nullable=False)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    road_type_weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    traffic_proxy: Mapped[float | None] = mapped_column(Float, nullable=True)

    detections: Mapped[list["Detection"]] = relationship(back_populates="report", cascade="all, delete-orphan")


class Detection(Base):
    """One detected pothole within a report's photo."""

    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_id: Mapped[int] = mapped_column(ForeignKey("reports.id"), nullable=False)
    class_name: Mapped[str] = mapped_column(String, default="pothole")
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    mask_polygon: Mapped[list] = mapped_column(JSON, nullable=False)  # [[x, y], ...] in image pixel space
    bbox: Mapped[list] = mapped_column(JSON, nullable=False)  # [xmin, ymin, xmax, ymax]
    status: Mapped[RepairStatus] = mapped_column(Enum(RepairStatus), default=RepairStatus.reported)
    status_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    report: Mapped["Report"] = relationship(back_populates="detections")
    severity_score: Mapped["SeverityScore | None"] = relationship(
        back_populates="detection", uselist=False, cascade="all, delete-orphan"
    )
    priority_score: Mapped["PriorityScoreRecord | None"] = relationship(
        back_populates="detection", uselist=False, cascade="all, delete-orphan"
    )


class SeverityScore(Base):
    """The output of src/severity/pipeline.py for one detection -- see that module for
    how these values are actually computed; this table just stores the result."""

    __tablename__ = "severity_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    detection_id: Mapped[int] = mapped_column(ForeignKey("detections.id"), unique=True, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)
    area_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    depth_value: Mapped[float] = mapped_column(Float, nullable=False)
    depth_source: Mapped[str] = mapped_column(String, nullable=False)
    irregularity: Mapped[float] = mapped_column(Float, nullable=False)

    detection: Mapped["Detection"] = relationship(back_populates="severity_score")


class PriorityScoreRecord(Base):
    """The output of src/prioritization/formula.py for one detection."""

    __tablename__ = "priority_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    detection_id: Mapped[int] = mapped_column(ForeignKey("detections.id"), unique=True, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)
    recurrence_factor: Mapped[float] = mapped_column(Float, nullable=False)
    alpha: Mapped[float] = mapped_column(Float, nullable=False)
    beta: Mapped[float] = mapped_column(Float, nullable=False)
    gamma: Mapped[float] = mapped_column(Float, nullable=False)
    delta: Mapped[float] = mapped_column(Float, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    detection: Mapped["Detection"] = relationship(back_populates="priority_score")
