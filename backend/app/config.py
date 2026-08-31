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

# Admin auth (see app/auth.py). The default hash/secret below are for local dev only --
# ADMIN_PASSWORD_HASH is the bcrypt hash of "PrismAdmin#2026" (see .env.example); both
# must be overridden via real env vars before this is ever exposed beyond localhost.
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = os.environ.get(
    "ADMIN_PASSWORD_HASH", "$2b$12$siL63kAfXPBGRMDb3.zebu42dZYYDa6sPR1xarp72HVFhH/FaPSRW"
)
JWT_SECRET = os.environ.get("JWT_SECRET", "dev-only-insecure-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "480"))
