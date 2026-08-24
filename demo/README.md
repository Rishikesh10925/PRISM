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

## Verified working

Manually QA'd via browser automation on 2026-08-25: page loads with zero console
errors, a real sample image runs through the full pipeline (detection mask overlay +
severity sub-scores + priority score all render correctly), and moving the severity
slider to its maximum changed the displayed Priority Score from 22.2 to 15.9 — which
matches `formula.priority_score()`'s output for that exact weight change by hand
calculation, confirming the live re-rank is wired correctly, not just visually
plausible.
