"""Validation of citizen photo uploads (POST /api/report) and EXIF capture-time lookup.

The file's real type is decided from its decoded content (Pillow), never from the
client-supplied filename or Content-Type header.
"""

from __future__ import annotations

import io
import re
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}  # Pillow format -> stored extension
MIN_IMAGE_SIDE = 32  # px -- nothing smaller can contain a usable pothole
MAX_IMAGE_SIDE = 12_000  # px -- ~ a 100+ MP phone sensor; beyond that is junk or a decompression bomb
MAX_IMAGE_PIXELS = 50_000_000

EXIF_IFD_POINTER = 0x8769
EXIF_DATETIME_ORIGINAL = 36867


class UploadRejected(Exception):
    """An upload failed validation; status_code/detail map straight onto an HTTPException."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def validate_image_bytes(image_bytes: bytes, max_bytes: int) -> Image.Image:
    """Returns the opened (header-parsed, not fully decoded) Pillow image, or raises
    UploadRejected: 413 too large, 415 not JPEG/PNG/WebP, 422 empty or absurd dimensions."""
    if len(image_bytes) > max_bytes:
        raise UploadRejected(413, f"image exceeds the {max_bytes / (1024 * 1024):g} MB upload limit")
    if not image_bytes:
        raise UploadRejected(422, "empty upload")

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Image.DecompressionBombError:
        raise UploadRejected(422, "image dimensions too large")
    except (UnidentifiedImageError, OSError):
        raise UploadRejected(415, "unsupported file type: upload a JPEG, PNG or WebP image")

    if img.format not in ALLOWED_FORMATS:
        raise UploadRejected(415, "unsupported image format: upload a JPEG, PNG or WebP image")

    width, height = img.size
    if (
        min(width, height) < MIN_IMAGE_SIDE
        or max(width, height) > MAX_IMAGE_SIDE
        or width * height > MAX_IMAGE_PIXELS
    ):
        raise UploadRejected(422, f"image dimensions {width}x{height} out of the accepted range")
    return img


def safe_stored_name(unique_prefix: str, client_filename: str | None, image_format: str) -> str:
    """{prefix}_{sanitized client stem}{extension from the DECODED format} -- the client's
    extension is discarded, and the stem is reduced to [A-Za-z0-9_-] (max 40 chars)."""
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(client_filename or "upload").stem)[:40] or "upload"
    return f"{unique_prefix}_{stem}{ALLOWED_FORMATS[image_format]}"


def exif_capture_hour(img: Image.Image, now: datetime | None = None) -> int | None:
    """Hour (0-23) from the photo's EXIF DateTimeOriginal, or None if absent, unparsable
    or implausible (before 2000, or more than a day after `now`, default server-local
    time). EXIF times are camera-local with no timezone, which is what we want for an
    hour-of-day traffic heuristic."""
    now = now or datetime.now()
    try:
        raw = img.getexif().get_ifd(EXIF_IFD_POINTER).get(EXIF_DATETIME_ORIGINAL)
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("ascii", errors="ignore")
        taken = datetime.strptime(str(raw).strip().strip("\x00"), "%Y:%m:%d %H:%M:%S")
    except Exception:  # malformed EXIF must never fail the upload
        return None
    if taken.year < 2000 or taken > now + timedelta(days=1):
        return None
    return taken.hour


EXIF_ORIENTATION = 0x0112


def upright_image_bytes(img: Image.Image, image_bytes: bytes) -> bytes:
    """The upload re-encoded upright if its EXIF Orientation tag says it's rotated or
    mirrored (common for phone photos), else the original bytes unchanged.

    OpenCV ignores the tag, so without this the model would analyse a sideways image and
    the stored mask polygons wouldn't match the photo browsers display (which honour the
    tag). Analysis and storage both use the returned bytes, so they always agree."""
    if img.getexif().get(EXIF_ORIENTATION, 1) in (0, 1):
        return image_bytes
    upright = ImageOps.exif_transpose(img)
    if img.format == "JPEG" and upright.mode not in ("RGB", "L"):
        upright = upright.convert("RGB")
    out = io.BytesIO()
    upright.save(out, format=img.format, **({"quality": 95} if img.format in ("JPEG", "WEBP") else {}))
    return out.getvalue()
