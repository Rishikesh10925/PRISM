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
# Chosen on the VALIDATION split (evaluation/detection/conf_threshold_choice.json): the
# lowest threshold within 0.005 of max instance-level F1.
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.35"))

# Severity cue bounds fitted on the TRAIN split (src/severity/fit_cue_bounds.py); falls
# back to SeverityConfig() defaults if the file is missing.
SEVERITY_CONFIG_PATH = Path(
    os.environ.get("SEVERITY_CONFIG_PATH", str(REPO_ROOT / "evaluation" / "severity" / "cue_bounds.json"))
)

# Admin auth (see app/auth.py). The default hash/secret below are for local dev only --
# ADMIN_PASSWORD_HASH is the bcrypt hash of "PrismAdmin#2026" (see .env.example); both
# must be overridden via real env vars before this is ever exposed beyond localhost.
DEV_ADMIN_PASSWORD_HASH = "$2b$12$siL63kAfXPBGRMDb3.zebu42dZYYDa6sPR1xarp72HVFhH/FaPSRW"
DEV_JWT_SECRET = "dev-only-insecure-secret-change-me"
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = os.environ.get("ADMIN_PASSWORD_HASH", DEV_ADMIN_PASSWORD_HASH)
JWT_SECRET = os.environ.get("JWT_SECRET", DEV_JWT_SECRET)
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "480"))

# "development" (default, zero-config) or "production". In production the app refuses
# to start while JWT_SECRET / ADMIN_PASSWORD_HASH are still the dev defaults above.
APP_ENV = os.environ.get("APP_ENV", "development").strip().lower()

# Browser origins allowed by CORS (comma-separated). Default = the Vite dev server
# (frontend/vite.config.js sets no port, so Vite's default 5173).
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if o.strip()
]

# POST /api/report upload limits.
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "15"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)

# Earlier reports within this many metres of a new one count towards its recurrence
# factor (same default as traffic_recurrence.cluster_recurrence_counts).
RECURRENCE_RADIUS_M = float(os.environ.get("RECURRENCE_RADIUS_M", "25"))


def check_production_secrets(app_env: str, jwt_secret: str, admin_password_hash: str) -> None:
    """Raises RuntimeError if app_env is production and either secret is a dev default
    (or empty). Never logs/echoes the secret values themselves."""
    if app_env != "production":
        return
    insecure = []
    if not jwt_secret or jwt_secret == DEV_JWT_SECRET:
        insecure.append("JWT_SECRET")
    if not admin_password_hash or admin_password_hash == DEV_ADMIN_PASSWORD_HASH:
        insecure.append("ADMIN_PASSWORD_HASH")
    if insecure:
        raise RuntimeError(
            f"APP_ENV=production but {', '.join(insecure)} still use the dev defaults -- "
            "set real values (see backend/.env.example) before starting."
        )


check_production_secrets(APP_ENV, JWT_SECRET, ADMIN_PASSWORD_HASH)
