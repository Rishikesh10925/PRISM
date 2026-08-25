# PRISM demo dashboard

A local Streamlit interface wired directly to the real trained pipeline (YOLOv8n-seg
detection → severity scoring → prioritization), for presenting/demoing the working
system before the planned production web app exists.

**This is not the production frontend.** The blueprint's real architecture is React +
Leaflet + FastAPI + PostgreSQL/PostGIS (see the root [README.md](../README.md)) — none
of that has been built yet (Phase 5). This is a single-process stand-in that lets you
show the actual detect → severity → priority pipeline running on real images, including
the live weight-slider re-ranking demo moment described in the blueprint (Section 10),
without waiting on the full web app.

## Run it

```
pip install streamlit
streamlit run demo/app.py
```

Opens at `http://localhost:8501`. Requires a trained checkpoint at
`models/yolov8n_seg_augmented.pt` (see [src/detection/train_yolo.py](../src/detection/train_yolo.py)).

## What it does

1. Pick an uploaded photo or a real test-set sample image.
2. Runs the actual trained YOLOv8n-seg model — no mock data.
3. For each detected pothole, computes the real severity pipeline (area ratio, MiDaS
   depth-proxy, contour irregularity → fused Severity Score).
4. Computes a real Priority Score using a live OpenStreetMap road-type lookup for the
   given coordinates, plus the traffic/recurrence heuristics.
5. α/β/γ/δ priority weight sliders re-rank instantly — the expensive detection/MiDaS
   step is cached per image, so only the fast formula recomputes on slider movement.

## Who sees what

The main screen is built for a non-technical viewer — a citizen or an authority
official, not a developer. Per pothole it shows a **Severity Score (0–100)** with a
plain level (Very Low/Low/Medium/High/Critical), a **Repair Priority Score (0–100)**
with a plain level (Less Important/Moderate/Important/Very Important), a one-line
recommended action, and four plain-language factors (Pothole Severity, Road Importance,
Traffic Level, Previous Reports) instead of raw weights or model internals.

All the underlying technical numbers (area ratio, MiDaS depth value, contour
irregularity, detection confidence, the raw OSM road-type weight, the actual α/β/γ/δ
weights) are still there, just tucked into a collapsed **"🔧 Admin / developer
details"** section per result card, plus an **"⚙️ Admin / developer details"** section
in the sidebar for adjusting the priority weights or overriding the location manually.
See [docs/phase5/01_demo_ui_notes.md](../docs/phase5/01_demo_ui_notes.md) for exactly
what changed and why — the detection/severity/priority calculation logic itself is
untouched, only what's displayed by default.

Location and time are auto-detected where a genuine automatic source exists (the
system clock for time; real browser geolocation, with a documented fallback location if
denied/unavailable, for GPS) — "Previous Reports" defaults to "None" rather than
fabricating a number, since there's no live citizen-report database yet (Phase 5).

## Verified working

Manually QA'd via browser automation on 2026-08-25: page loads with zero console
errors, a real sample image runs through the full pipeline (detection mask overlay,
plain-language result card, and the Admin details expander all render correctly), and
the displayed severity/priority levels were checked against their score thresholds
(e.g. a 10/100 severity score correctly shows "Very Low", a 24/100 priority score
correctly shows "Less Important").
