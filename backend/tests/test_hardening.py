"""Phase 6 hardening: upload validation, CORS, production-secret check, and clean
failure handling of POST /api/report."""

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from app import config
from app.models import Report

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _image_bytes(fmt="PNG", size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, (90, 90, 90)).save(buf, format=fmt)
    return buf.getvalue()


def _post(client, data, filename="photo.jpg"):
    return client.post(
        "/api/report",
        files={"image": (filename, io.BytesIO(data), "image/jpeg")},
        data={"latitude": 17.4, "longitude": 78.4},
    )


# --- upload validation ------------------------------------------------------------


def test_oversize_upload_rejected_413(client, monkeypatch, upload_dir):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1024)
    resp = _post(client, _image_bytes(size=(400, 400)) + b"\0" * 2048)
    assert resp.status_code == 413
    assert not upload_dir.exists() or not any(upload_dir.iterdir())


def test_non_image_rejected_415_even_with_image_filename(client, upload_dir):
    resp = _post(client, b"#!/bin/sh\necho definitely not a jpeg\n", filename="photo.jpg")
    assert resp.status_code == 415
    assert not upload_dir.exists() or not any(upload_dir.iterdir())


def test_disallowed_image_format_rejected_415(client):
    assert _post(client, _image_bytes(fmt="GIF"), filename="photo.png").status_code == 415


def test_absurd_dimensions_rejected_422(client):
    assert _post(client, _image_bytes(size=(20_000, 40))).status_code == 422  # too long
    assert _post(client, _image_bytes(size=(8, 8))).status_code == 422  # too small


def test_safe_stored_name_ignores_client_extension():
    from app.services.uploads import safe_stored_name

    name = safe_stored_name("abc", "../../evil photo.php", "JPEG")
    assert name == "abc_evil_photo.jpg"
    assert "/" not in name and ".." not in name


# --- failure handling -------------------------------------------------------------


def test_pipeline_failure_returns_clean_500_and_leaves_nothing(client, db_session, monkeypatch, upload_dir):
    from app.routers import reports

    def boom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory (simulated)")

    monkeypatch.setattr(reports, "analyze_report_image", boom)
    before = db_session.query(Report).count()

    resp = _post(client, _image_bytes(), filename="photo.png")

    assert resp.status_code == 500
    assert resp.json() == {"detail": "image analysis failed; the report was not saved"}
    assert "CUDA" not in resp.text  # internals stay in the server log only
    assert db_session.query(Report).count() == before
    assert not upload_dir.exists() or not any(upload_dir.iterdir())


# --- CORS -------------------------------------------------------------------------


def test_cors_allows_configured_origin(client):
    resp = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_rejects_other_origin(client):
    resp = client.get("/api/health", headers={"Origin": "https://evil.example.com"})
    assert "access-control-allow-origin" not in resp.headers

    preflight = client.options(
        "/api/report",
        headers={"Origin": "https://evil.example.com", "Access-Control-Request-Method": "POST"},
    )
    assert preflight.status_code == 400
    assert "access-control-allow-origin" not in preflight.headers


# --- production secrets -----------------------------------------------------------


def test_check_production_secrets():
    check = config.check_production_secrets
    check("development", config.DEV_JWT_SECRET, config.DEV_ADMIN_PASSWORD_HASH)  # dev: fine

    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        check("production", config.DEV_JWT_SECRET, "$2b$12$real-hash")
    with pytest.raises(RuntimeError, match="ADMIN_PASSWORD_HASH"):
        check("production", "a-real-secret", config.DEV_ADMIN_PASSWORD_HASH)
    check("production", "a-real-secret", "$2b$12$real-hash")


def _import_config(**env_overrides):
    env = {k: v for k, v in os.environ.items() if k not in ("APP_ENV", "JWT_SECRET", "ADMIN_PASSWORD_HASH")}
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", "import app.config"], cwd=BACKEND_ROOT, env=env, capture_output=True, text=True
    )


def test_production_refuses_to_start_with_dev_secrets():
    result = _import_config(APP_ENV="production")
    assert result.returncode != 0
    assert "still use the dev defaults" in result.stderr
    assert config.DEV_JWT_SECRET not in result.stderr  # never echo the secret itself


def test_production_starts_with_real_secrets():
    result = _import_config(APP_ENV="production", JWT_SECRET="x" * 40, ADMIN_PASSWORD_HASH="$2b$12$" + "a" * 53)
    assert result.returncode == 0, result.stderr


# --- photo orientation ------------------------------------------------------------


def _rotated_phone_jpeg(width=60, height=40, orientation=6):
    """A JPEG stored landscape but tagged 'rotate 90 degrees' -- how phones save portrait shots."""
    exif = Image.Exif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (90, 90, 90)).save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def test_upright_image_bytes_applies_exif_rotation():
    from app.services.uploads import upright_image_bytes

    data = _rotated_phone_jpeg()
    upright = Image.open(io.BytesIO(upright_image_bytes(Image.open(io.BytesIO(data)), data)))
    assert upright.size == (40, 60)
    assert upright.getexif().get(0x0112, 1) == 1


def test_upright_image_bytes_leaves_untagged_photos_untouched():
    from app.services.uploads import upright_image_bytes

    data = _image_bytes(fmt="JPEG", size=(60, 40))
    assert upright_image_bytes(Image.open(io.BytesIO(data)), data) is data


def test_rotated_phone_photo_is_analysed_and_stored_upright(client, monkeypatch, upload_dir):
    from app.routers import reports

    seen = {}

    def fake_analyze(image_bgr, *args, **kwargs):
        seen["shape"] = image_bgr.shape
        return [], 0.4, "osm", 0.2

    monkeypatch.setattr(reports, "analyze_report_image", fake_analyze)
    resp = _post(client, _rotated_phone_jpeg())

    assert resp.status_code == 200, resp.text
    assert seen["shape"][:2] == (60, 40)  # height x width: upright portrait
    stored = Image.open(next(upload_dir.iterdir()))
    assert stored.size == (40, 60)
