"""Integration tests against the real pipeline and a real (isolated-schema) Postgres+
PostGIS database -- no mocks for detection/severity/priority, since the whole point is
verifying the real wiring works, matching how the rest of this project is tested.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_IMAGE = REPO_ROOT / "data" / "merged" / "images" / "pothole600_testing_0007.png"

pytestmark = pytest.mark.skipif(
    not SAMPLE_IMAGE.exists(), reason="real merged dataset image not present"
)


def _submit_report(client, image_path=SAMPLE_IMAGE, lat=17.4239, lon=78.4738):
    with open(image_path, "rb") as f:
        return client.post(
            "/api/report",
            files={"image": ("photo.png", f, "image/png")},
            data={"latitude": lat, "longitude": lon},
        )


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_create_report_runs_real_pipeline_and_persists(client):
    resp = _submit_report(client)
    assert resp.status_code == 200
    body = resp.json()

    assert body["latitude"] == pytest.approx(17.4239)
    assert body["longitude"] == pytest.approx(78.4738)
    assert len(body["detections"]) >= 1

    detection = body["detections"][0]
    assert detection["status"] == "reported"
    assert 0 <= detection["severity_score"]["score"] <= 100
    assert detection["severity_score"]["category"] in (
        "Very Low", "Low", "Medium", "High", "Critical",
    )
    assert 0 <= detection["priority_score"]["score"] <= 100


def test_create_report_rejects_invalid_coordinates(client):
    resp = _submit_report(client, lat=200, lon=0)
    assert resp.status_code == 422


def test_list_and_filter_potholes(client):
    _submit_report(client)

    resp = client.get("/api/potholes")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1

    resp_filtered = client.get("/api/potholes", params={"status": "repaired"})
    assert resp_filtered.status_code == 200
    assert resp_filtered.json() == []  # nothing marked repaired yet


def test_priority_list_is_sorted_descending(client):
    _submit_report(client)
    _submit_report(client, lat=17.5, lon=78.5)

    resp = client.get("/api/priority-list")
    assert resp.status_code == 200
    scores = [row["priority_score"] for row in resp.json()]
    assert scores == sorted(scores, reverse=True)


def test_update_status(client):
    create_resp = _submit_report(client)
    detection_id = create_resp.json()["detections"][0]["id"]

    resp = client.patch(f"/api/potholes/{detection_id}/status", json={"status": "repaired"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "repaired"


def test_update_status_404_for_unknown_detection(client):
    resp = client.patch("/api/potholes/999999/status", json={"status": "repaired"})
    assert resp.status_code == 404


def test_pdf_report_endpoint(client):
    _submit_report(client)
    resp = client.get("/api/report/pdf")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content[:4] == b"%PDF"
