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
backend/               FastAPI service
frontend/              React + Leaflet dashboard
evaluation/            detection / severity / prioritization metric scripts
paper/                 paper source and figures
```

## Status

Phase 1 (literature review & planning) in progress — see [docs/phase1/](docs/phase1/).
