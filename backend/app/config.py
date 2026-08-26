"""App configuration, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg://prism:prism_dev_password@localhost:5433/prism"
)

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", str(REPO_ROOT / "data" / "reports")))

YOLO_MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH", str(REPO_ROOT / "models" / "yolov8n_seg_multisource.pt")
)
USE_MIDAS = os.environ.get("USE_MIDAS", "true").lower() == "true"
