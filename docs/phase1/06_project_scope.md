# Phase 1 / Task 6 — Project Scope Decisions (Signed Off)

Locked in from the blueprint's Feasibility section (Section 11), cross-checked against what Phase 1 actually confirmed.

## Primary detector: YOLOv8-seg

Locked as the primary detection/segmentation model. Justification from the literature review:
instance segmentation (not bounding boxes) is required because the severity module's area-ratio and
irregularity cues both need a pixel mask, and YOLOv8-seg variants are the most consistently used and
best-documented choice in recent pothole literature (P01, P04, P05), with reported mAP@0.5 in the 90%+
range on comparable data. Mask R-CNN (Detectron2) stays in scope only as the accuracy/speed comparison
baseline (per P07), not as a candidate for the production model.

## Stretch goal: Open-vocabulary detection (Grounding DINO)

Confirmed as **stretch, not core** — matches the blueprint's Contribution 3 framing. Task 1's search found
no prior application of Grounding DINO (or OWL-ViT) to road damage specifically (P14), which is exactly
why it's a genuine novelty add-on if time allows, but the paper is fully publishable on Contributions 1+2
(severity scoring + prioritization) alone if Phase 6/13 time runs out.

## What gets cut first if time runs short (from Feasibility section, reaffirmed)

In priority order, these are dropped before anything in the core scope:

1. Open-vocabulary comparison module (Contribution 3) — bonus only.
2. Live traffic API integration — falls back to the OSM road-type + time-of-day heuristic, stated as a limitation.
3. Real depth sensors/stereo capture for our own data — stays with the monocular MiDaS/DPT proxy; Pothole-600 is used only to validate this proxy on a subset, not to change the core approach.
4. Citizen-report recurrence tracking — falls back to synthetic repeated-report simulation if real usage data isn't available by submission time.

## Core scope (must-do, not cuttable without weakening the paper)

- Fine-tune YOLOv8-seg on the merged, confirmed-license dataset pool (D1 RDD2022, D5 Roboflow pothole set, D7 Indian Roads; Pothole-600 for depth-proxy validation only, not merged/redistributed — per [05_dataset_verification.md](05_dataset_verification.md)).
- Three-cue severity fusion formula (area, depth-proxy, irregularity) calibrated against a 150-250 image human-rated validation subset.
- Prioritization engine (severity + road-type + traffic proxy + recurrence) with configurable weights.
- Full-stack dashboard (FastAPI + PostgreSQL/PostGIS + React/Leaflet).
- Complete evaluation suite: detection metrics, severity correlation (target Spearman's ρ ≥ 0.7), prioritization ranking (target Kendall's τ ≥ 0.6), and the four required ablations (fusion, weight-sensitivity, augmentation, model choice).

## Dataset licensing constraint carried into scope (new, from Task 5)

Because Pothole-600's license is unconfirmed for redistribution, the project's **released/public dataset
artifact** (if any is published alongside the paper, e.g. the severity validation subset) will be built only
from RDD2022 (CC BY-SA 4.0), the Roboflow pothole set (ODbL v1.0), and Indian Roads (GPL 2.0) content, or
from newly/self-captured images. Pothole-600 remains an internal-only validation aid. This is a direct,
concrete consequence of Task 5's findings and is now part of the locked scope, not an open question.

## Sign-off

All six Phase 1 deliverables are complete as of 2026-08-18:

| # | Task | Deliverable | Status |
|---|---|---|---|
| 1 | Survey existing literature | [01_annotated_bibliography.csv](01_annotated_bibliography.csv) / [.md](01_annotated_bibliography.md) | Done — 16 sources |
| 2 | Identify the research gap | [02_research_gap_statement.md](02_research_gap_statement.md) | Done |
| 3 | Draft Related Work (rough) | [03_related_work_draft_v0.md](03_related_work_draft_v0.md) | Done — v0 |
| 4 | Shortlist candidate datasets | [04_dataset_shortlist.md](04_dataset_shortlist.md) | Done — 7 candidates (D1-D7) |
| 5 | Verify dataset access/licensing | [05_dataset_verification.md](05_dataset_verification.md) | Done — 3 fully confirmed + 1 conditional |
| 6 | Finalize project scope decisions | this document | Done |

Phase 1 is complete. Next: Phase 2 — Dataset Preparation (download D1/D5/D7, standardize to YOLO-seg format, build the severity validation subset and rubric).
