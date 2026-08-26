# PRISM — Pothole Repair Intelligence & Severity Mapping

Academic capstone project (NMIMS Hyderabad) building a computer-vision system that goes beyond pothole
*detection* to quantify pothole *severity* and produce a multi-criteria *repair-priority* ranking for
municipal road maintenance.

Full plans:
- [docs/Pothole_Severity_Detection_Blueprint.docx](docs/Pothole_Severity_Detection_Blueprint.docx) — research + engineering blueprint
- [docs/Pothole_Project_Full_WorkPlan.docx](docs/Pothole_Project_Full_WorkPlan.docx) — phase-by-phase task plan
- [docs/phase1/](docs/phase1/) — Phase 1 (literature review & planning) deliverables

## Project layout

```
data/                 raw, merged, and annotated datasets (not versioned; see data/*/README)
notebooks/             exploratory notebooks
src/
  preprocessing/        dataset cleaning, format conversion, dedup
  detection/             YOLOv8-seg / Mask R-CNN training & inference
  severity/              area / depth-proxy / irregularity modules + fusion
  prioritization/        priority scoring engine
  utils/                 shared helpers
models/                trained weights (not versioned)
backend/               FastAPI service + PostgreSQL/PostGIS models, migrations, tests
frontend/              React + Leaflet citizen upload portal and admin dashboard
demo/                  local Streamlit demo dashboard (real pipeline, presentation use only — see demo/README.md)
evaluation/            detection / severity / prioritization metric scripts
paper/                 paper source and figures
```

## Status

- **Phase 1** (literature review & planning) — done, see [docs/phase1/](docs/phase1/).
- **Phase 2** (dataset preparation) — partial, see [docs/phase2/](docs/phase2/). Pothole-600 plus four Kaggle
  sources are downloaded and converted; RDD2022 and Roboflow Universe remain access-blocked; the severity
  validation subset (human ratings) hasn't been built yet.
- **Phase 3** (detection model) — done, see [docs/phase3/](docs/phase3/). YOLOv8n-seg and a Mask R-CNN
  baseline are both trained and evaluated.
- **Phase 4** (severity & prioritization modules) — done except fusion-weight calibration (blocked on the
  Phase 2 human-ratings gap), see [docs/phase4/](docs/phase4/).
- **Phase 5** (application integration: backend/frontend/database) — done except real-world field
  testing (walking around and photographing actual roads, which needs a person on site), see
  [docs/phase5/](docs/phase5/). PostgreSQL+PostGIS database, FastAPI backend (5 endpoints, PDF
  report generation, 8 passing integration tests), and a React citizen upload portal + admin
  dashboard (Leaflet map, live-adjustable priority weights) are all built and verified working
  end-to-end against the real detection/severity/prioritization pipeline.
