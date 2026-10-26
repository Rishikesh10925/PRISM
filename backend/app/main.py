from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import APP_ENV, CORS_ORIGINS
from app.routers import auth, pdf_report, potholes, reports

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger(__name__).info("PRISM API starting (APP_ENV=%s, CORS origins=%s)", APP_ENV, CORS_ORIGINS)

app = FastAPI(title="PRISM API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # env CORS_ORIGINS, default = Vite dev server (app/config.py)
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(reports.router)
app.include_router(potholes.router)
app.include_router(pdf_report.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
