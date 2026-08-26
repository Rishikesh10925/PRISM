# Phase 5 — Application Integration: Status

## Done and verified

- **Database.** PostgreSQL 16 + PostGIS 3.4 running via [docker-compose.yml](../../docker-compose.yml)
  (host port 5433, to avoid clashing with an unrelated container already using 5432 on this machine).
  Schema in [backend/app/models.py](../../backend/app/models.py), applied through Alembic
  (`backend/alembic/versions/1c446b0a5a73_initial_schema.py`). Details in
  [02_database_schema.md](02_database_schema.md).
- **Backend API**, [backend/app/](../../backend/app/), FastAPI + SQLAlchemy 2.0:
  - `POST /api/report` — accepts a photo + GPS coordinates, runs the real detection →
    severity → priority pipeline (`app/services/inference.py`, calling `src/severity` and
    `src/prioritization` directly, not a re-implementation), persists Report/Detection/
    SeverityScore/PriorityScoreRecord.
  - `GET /api/potholes` — list with status/min-severity filters.
  - `GET /api/priority-list` — flattened, sorted worklist; includes the raw
    `road_type_weight`/`traffic_proxy`/`recurrence_factor` components so a client can re-rank
    locally without another round trip.
  - `PATCH /api/potholes/{id}/status` — reported / in_progress / repaired.
  - `GET /api/report/pdf` — real ReportLab-generated PDF, optional bbox filter.
  - Covered by 8 integration tests in `backend/tests/test_api.py`, run against the real
    Postgres instance (isolated schema + per-test rollback), all passing.
- **Citizen portal**, [frontend/src/pages/CitizenUpload.jsx](../../frontend/src/pages/CitizenUpload.jsx):
  mobile-first upload flow using real `navigator.geolocation`, submits to `POST /api/report`,
  shows a plain-language result (severity/priority badges + scores only — no technical
  internals, consistent with the demo's UI redesign). Verified end-to-end in a real browser:
  real image upload → real inference → correct result matching the API response exactly.
- **Admin dashboard**, [frontend/src/pages/AdminDashboard.jsx](../../frontend/src/pages/AdminDashboard.jsx):
  Leaflet map with colour-coded markers, worklist table, per-row status control, status
  filter, PDF download link, and four weight sliders (Pothole Severity / Road Importance /
  Traffic Level / Previous Reports) that re-rank the table instantly on the client using
  [frontend/src/priorityFormula.js](../../frontend/src/priorityFormula.js) — a direct port of
  `src/prioritization/formula.py`'s formula, checked by hand against the backend's own output
  (default weights: 19; severity weight maxed to 1.00: 14 — matches the formula exactly).
  Verified in a real browser: map renders, marker popups show correct per-pothole detail,
  status filter correctly narrows the row count, and the PDF link resolves to a real
  `application/pdf` response.
- **End-to-end wiring.** Citizen upload → backend inference → Postgres → admin dashboard is a
  real, working loop, not mocked at any layer. One real bug was found and fixed along the way:
  the backend process had been started without `--reload`, so it kept serving stale API
  responses after a schema/router change — see the "stale server" note below.

## Known limitation: Task 12 (real-world field testing)

The work plan calls for testing the system on photos/video captured by walking around real
roads. I can't do that myself — I have no camera or physical presence. What's been done
instead, honestly: the citizen upload flow has been exercised with real (not synthetic) sample
images from the existing dataset, through the actual browser UI, with real GPS coordinates
supplied via the browser's geolocation API. That confirms the pipeline works end-to-end on real
photographs, but it is not a substitute for a genuine field-capture test. If you can walk a
stretch of road and upload a few photos through the running citizen portal
(`http://localhost:5173`), that would close this task out for real — happy to help review the
results.

## Recurring bug pattern worth remembering

Twice this session, code on disk was correct but a long-running process kept serving the old
version: `st.cache_resource` silently holding a stale model object in the Streamlit demo, and
`uvicorn` without `--reload` silently serving an old schema in the FastAPI backend. Both looked
like logic bugs (wrong severity, `NaN` priority) until traced back to "the process didn't
reload." Worth checking first whenever a fix doesn't take effect.
