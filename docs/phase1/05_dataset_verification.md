# Phase 1 / Task 5 — Dataset Access & Licensing Verification

Each shortlisted source from [04_dataset_shortlist.md](04_dataset_shortlist.md) was visited directly. Results below reflect what was actually confirmed today (2026-08-18), not what the shortlist assumed.

| # | Dataset | Access check | License confirmed | Verdict |
|---|---|---|---|---|
| D1 | RDD2022 | arXiv paper page loads and is readable; figshare record returned HTTP 403 to automated fetch (likely bot-blocking, not an access restriction — figshare listings are normally public) | **CC BY-SA 4.0**, confirmed directly on the arXiv abstract page (links to `creativecommons.org/licenses/by-sa/4.0`) | ✅ Confirmed. Academic use and redistribution of derived/merged sets is permitted under CC BY-SA 4.0, provided attribution and share-alike terms are honored in our merged dataset's license/README |
| D2 | Annotated Potholes Image Dataset (Kaggle) | Page is JavaScript-rendered; automated fetch only returns the title, not license/size metadata | **Not confirmed** — Kaggle's dataset-info panel did not load via automated fetch | ⚠️ Needs manual browser check (log into Kaggle, open dataset page, read the "License" field directly) before relying on it for redistribution |
| D3 | Potholes-Detection-YOLOv8 (Kaggle) | Same JS-rendering limitation as D2 | **Not confirmed** by direct fetch; search-result snippet indicated CC0: Public Domain, but this was not independently verified on-page | ⚠️ Needs the same manual check as D2 |
| D4 | Potholes and Roads Instance Segmentation (Roboflow Universe project page) | Automated fetch returned HTTP 403 (Roboflow Universe project pages appear to require a session/login to render fully) | **Not confirmed** | ⚠️ Dropped from the confirmed list; would need a Roboflow account to verify before use |
| D5 | Pothole Object Detection Dataset (public.roboflow.com) | Fetched successfully | **ODbL v1.0** (Open Data Commons Open Database License) — confirmed on-page, 665 images | ✅ Confirmed. Note: this corrects the shortlist draft, which had guessed "Public Domain" from a search snippet — the actual license is ODbL v1.0, which permits reuse/redistribution with attribution and share-alike on the database itself |
| D6 | Pothole-600 | Project site loads; direct download links present (Google Drive + Kaggle mirror) | **No explicit license stated** — page only asks users to cite the associated papers when used for research | ⚠️ Confirmed downloadable, but licensing is ambiguous: citation-only norms are common in academic CV datasets but do not equal a redistribution license. **Decision: use for internal validation of the depth-proxy module only; do not redistribute Pothole-600 imagery as part of any merged/released dataset** without contacting the authors for explicit permission |
| D7 | Indian Roads (semantic segmentation) | Dataset Ninja mirror page fetched successfully | **GNU GPL 2.0**, confirmed on-page; downloadable via Supervisely format or Kaggle mirror, 3,227 images / 8,129 objects | ✅ Confirmed |

## Confirmed dataset list (for Phase 2 download)

Sources with a directly confirmed, unambiguous open license, ready to use for training and for a
redistributable merged/derived set:

1. **RDD2022** — CC BY-SA 4.0
2. **Pothole Object Detection Dataset (Roboflow public)** — ODbL v1.0
3. **Indian Roads (semantic segmentation)** — GNU GPL 2.0

Plus one conditionally-confirmed source, valid for internal validation but restricted for redistribution:

4. **Pothole-600** — downloadable, citation-required, no redistribution license confirmed; use only to validate the depth-proxy module against real disparity, keep out of any released merged dataset.

This satisfies the "Confirmed dataset list (3-4 sources)" deliverable. D2, D3, and D4 remain on the broader
shortlist as candidates for extra volume, but are held out of the confirmed/actionable list until their
license fields are checked manually in a logged-in browser session — automated fetching could not read
Kaggle's or Roboflow Universe's client-rendered license metadata (Kaggle returned only page titles;
Roboflow Universe project pages 403'd without a session). This is a concrete to-do for whoever starts
Phase 2 Task 1 (dataset download), not a blocker for finishing Phase 1 planning.

## Action items carried into Phase 2

- [ ] Manually confirm D2 and D3 licenses by logging into Kaggle and reading each dataset's "License" field.
- [ ] Create/log into a Roboflow account to re-check D4's license and export options.
- [ ] Before merging RDD2022 (CC BY-SA 4.0) with GPL 2.0 (Indian Roads) and ODbL (Roboflow) content, confirm
      with a project mentor/advisor that a merged derivative dataset for internal training use (not public
      redistribution) does not create a license conflict; if public release of the merged set is planned
      later, revisit share-alike compatibility between CC BY-SA, ODbL, and GPL 2.0 explicitly.
- [ ] Never include Pothole-600 images in any dataset export shared outside the team without direct author permission.
