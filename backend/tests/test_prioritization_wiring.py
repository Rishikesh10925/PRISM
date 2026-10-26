"""Phase 5 prioritization wiring: recurrence counting via PostGIS, road-type source
recording, EXIF report time, and backend/frontend priority-formula parity.

Report-submitting tests run the real pipeline (like test_api.py); only the Overpass
API is faked (conftest.fake_overpass) so nothing depends on a live external service.
"""

import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import requests
from PIL import Image

from app.models import Detection, RepairStatus

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_IMAGE = REPO_ROOT / "data" / "merged" / "images" / "pothole600_testing_0007.png"
PRIORITY_FORMULA_JS = REPO_ROOT / "frontend" / "src" / "priorityFormula.js"

needs_sample = pytest.mark.skipif(not SAMPLE_IMAGE.exists(), reason="real merged dataset image not present")

LAT, LON = 17.4239, 78.4738


def _submit(client, lat=LAT, lon=LON, image_bytes=None, filename="photo.png"):
    data = image_bytes if image_bytes is not None else SAMPLE_IMAGE.read_bytes()
    return client.post(
        "/api/report",
        files={"image": (filename, io.BytesIO(data), "application/octet-stream")},
        data={"latitude": lat, "longitude": lon},
    )


def _recurrence(body):
    return [d["priority_score"]["recurrence_factor"] for d in body["detections"]]


# --- recurrence -------------------------------------------------------------------


@needs_sample
def test_second_report_at_same_spot_gets_recurrence(client):
    first = _submit(client)
    assert first.status_code == 200, first.text
    assert first.json()["detections"], "sample image should yield at least one detection"
    assert all(r == 0.0 for r in _recurrence(first.json()))

    # ~5 m north of the first report -- inside the default 25 m radius
    second = _submit(client, lat=LAT + 0.000045)
    assert second.status_code == 200, second.text
    # count = 1 earlier + this one = 2 -> recurrence_factor(2) = (2-1)/(5-1) = 0.25
    assert _recurrence(second.json()) and all(r == pytest.approx(0.25) for r in _recurrence(second.json()))


@needs_sample
def test_far_away_report_does_not_count(client):
    assert _submit(client).status_code == 200
    # ~110 m north -- outside the 25 m radius
    far = _submit(client, lat=LAT + 0.001)
    assert far.status_code == 200, far.text
    assert all(r == 0.0 for r in _recurrence(far.json()))


@needs_sample
def test_repaired_report_does_not_count(client, db_session):
    first = _submit(client)
    assert first.status_code == 200
    for det in first.json()["detections"]:
        db_session.get(Detection, det["id"]).status = RepairStatus.repaired
    db_session.flush()

    second = _submit(client)
    assert second.status_code == 200, second.text
    assert all(r == 0.0 for r in _recurrence(second.json()))


# --- road-type source -------------------------------------------------------------


@needs_sample
@pytest.mark.parametrize(
    "overpass_result, expected_weight, expected_source",
    [
        ([{"tags": {"highway": "primary"}}], 1.0, "osm"),
        ([], 0.4, "default_no_road"),
        (requests.ConnectionError("overpass down"), 0.4, "default_api_error"),
    ],
)
def test_road_type_source_is_recorded(client, fake_overpass, overpass_result, expected_weight, expected_source):
    fake_overpass.result = overpass_result
    resp = _submit(client)
    assert resp.status_code == 200, resp.text
    assert resp.json()["road_type_weight"] == pytest.approx(expected_weight)
    assert resp.json()["road_type_source"] == expected_source


# --- report time ------------------------------------------------------------------


def _with_exif_datetime(png_path: Path, value: str) -> bytes:
    img = Image.open(png_path).convert("RGB")
    exif = Image.Exif()
    exif.get_ifd(0x8769)[36867] = value  # DateTimeOriginal
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def test_exif_capture_hour_plausibility():
    from datetime import datetime

    from app.services.uploads import exif_capture_hour

    def hour_for(value):
        img = Image.new("RGB", (64, 64))
        exif = Image.Exif()
        exif.get_ifd(0x8769)[36867] = value
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif)
        return exif_capture_hour(Image.open(io.BytesIO(buf.getvalue())), now=datetime(2026, 10, 26, 12))

    assert hour_for("2026:10:20 08:15:00") == 8
    assert hour_for("1999:01:01 08:00:00") is None  # implausibly old
    assert hour_for("2027:01:01 08:00:00") is None  # in the future
    assert hour_for("garbage") is None
    assert exif_capture_hour(Image.new("RGB", (64, 64))) is None  # no EXIF at all


@needs_sample
def test_report_uses_exif_hour_when_present(client):
    # residential road (0.4) at 03:00 -> night multiplier 0.2 -> traffic 0.08
    resp = _submit(client, image_bytes=_with_exif_datetime(SAMPLE_IMAGE, "2026:01:15 03:10:00"), filename="p.jpg")
    assert resp.status_code == 200, resp.text
    assert resp.json()["time_source"] == "exif"
    assert resp.json()["traffic_proxy"] == pytest.approx(0.08)

    no_exif = _submit(client)
    assert no_exif.json()["time_source"] == "server"


# --- formula parity (backend formula.py vs frontend priorityFormula.js) ------------

# (weights, inputs, hand-computed P)
PARITY_CASES = [
    # default weights: 0.4*0.5 + 0.3*0.4 + 0.2*0.28 + 0.1*0 = 0.376
    ((0.4, 0.3, 0.2, 0.1), (50, 0.4, 0.28, 0.0), 37.6),
    ((0.4, 0.3, 0.2, 0.1), (100, 1.0, 1.0, 1.0), 100.0),
    ((0.4, 0.3, 0.2, 0.1), (0, 0.0, 0.0, 0.0), 0.0),
    # unnormalized equal weights: (0.8 + 0.7 + 0.35 + 0.25) / 4 = 0.525
    ((1, 1, 1, 1), (80, 0.7, 0.35, 0.25), 52.5),
    # severity only
    ((2, 0, 0, 0), (63, 1.0, 1.0, 1.0), 63.0),
]


@pytest.mark.parametrize("weights, inputs, expected", PARITY_CASES)
def test_backend_formula_matches_hand_computed(weights, inputs, expected):
    from formula import priority_score
    from priority_schema import PriorityInputs, PriorityWeights

    assert priority_score(PriorityInputs(*inputs), PriorityWeights(*weights)) == pytest.approx(expected)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_frontend_formula_matches_backend():
    from formula import priority_category, priority_score
    from priority_schema import PriorityInputs, PriorityWeights

    cases = [
        {
            "weights": dict(zip("alpha beta gamma delta".split(), w)),
            "item": dict(zip(["severity_score", "road_type_weight", "traffic_proxy", "recurrence_factor"], i)),
        }
        for w, i, _ in PARITY_CASES
    ]
    script = (
        f"import {{ recomputePriority, priorityCategory }} from {json.dumps(PRIORITY_FORMULA_JS.as_uri())};"
        f"const cases = {json.dumps(cases)};"
        "console.log(JSON.stringify(cases.map(c => { const s = recomputePriority(c.item, c.weights);"
        " return [s, priorityCategory(s)]; })));"
    )
    out = subprocess.run(
        ["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=30, check=True
    ).stdout
    js_results = json.loads(out)

    for (w, i, _), (js_score, js_category) in zip(PARITY_CASES, js_results):
        py_score = priority_score(PriorityInputs(*i), PriorityWeights(*w))
        assert js_score == pytest.approx(py_score)
        assert js_category == priority_category(py_score)
