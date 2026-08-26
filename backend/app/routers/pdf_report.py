"""GET /api/report/pdf (Phase 5 Task 7): auto-generated summary PDF of the current
pothole worklist, for a municipal authority to print/share.

The blueprint's plan names this endpoint "per-region"; since there's no administrative
region field in the schema yet (no reverse-geocoding wired in), "region" here is an
optional GPS bounding box instead -- the same practical idea (a filtered area), without
depending on a geocoding service this project doesn't have.
"""

from __future__ import annotations

import io
from collections import Counter
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.database import get_db
from app.models import Detection

router = APIRouter(prefix="/api", tags=["reports"])


@router.get("/report/pdf")
def report_pdf(
    min_lat: float | None = None,
    max_lat: float | None = None,
    min_lon: float | None = None,
    max_lon: float | None = None,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    query = select(Detection).options(
        joinedload(Detection.severity_score), joinedload(Detection.priority_score), joinedload(Detection.report)
    )
    detections = list(db.execute(query).unique().scalars().all())

    if min_lat is not None:
        detections = [d for d in detections if d.report.latitude >= min_lat]
    if max_lat is not None:
        detections = [d for d in detections if d.report.latitude <= max_lat]
    if min_lon is not None:
        detections = [d for d in detections if d.report.longitude >= min_lon]
    if max_lon is not None:
        detections = [d for d in detections if d.report.longitude <= max_lon]

    detections = [d for d in detections if d.severity_score and d.priority_score]
    detections.sort(key=lambda d: d.priority_score.score, reverse=True)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("PRISM — Pothole Repair Worklist", styles["Title"]),
        Paragraph(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]),
        Spacer(1, 12),
    ]

    severity_counts = Counter(d.severity_score.category for d in detections)
    priority_counts = Counter(d.priority_score.category for d in detections)
    story.append(Paragraph(f"Total open reports: {len(detections)}", styles["Heading2"]))
    story.append(
        Paragraph(
            "By severity: " + ", ".join(f"{k}={v}" for k, v in severity_counts.items()) or "none", styles["Normal"]
        )
    )
    story.append(
        Paragraph(
            "By priority: " + ", ".join(f"{k}={v}" for k, v in priority_counts.items()) or "none", styles["Normal"]
        )
    )
    story.append(Spacer(1, 16))

    table_data = [["#", "Location (lat, lon)", "Severity", "Priority", "Status"]]
    for i, d in enumerate(detections[:100], start=1):
        table_data.append(
            [
                str(i),
                f"{d.report.latitude:.4f}, {d.report.longitude:.4f}",
                f"{d.severity_score.category} ({d.severity_score.score:.0f})",
                f"{d.priority_score.category} ({d.priority_score.score:.0f})",
                d.status.value,
            ]
        )

    table = Table(table_data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e2e2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
            ]
        )
    )
    story.append(table)

    doc.build(story)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=prism_worklist.pdf"},
    )
