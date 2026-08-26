from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import pdf_report, potholes, reports

app = FastAPI(title="PRISM API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tightened once a real frontend origin exists (Phase 5 Tasks 8-9)
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(reports.router)
app.include_router(potholes.router)
app.include_router(pdf_report.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
