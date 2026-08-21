# Phase 2 / Task 1 — Dataset Download Status

Real download attempts against every source confirmed in [docs/phase1/05_dataset_verification.md](../phase1/05_dataset_verification.md), run on 2026-08-21.

| Source | Attempted | Result |
|---|---|---|
| **Pothole-600** | Direct Google Drive fetch via `gdown` | ✅ **Downloaded** — 223 MB, 600 images (240 train / 180 val / 180 test) with RGB + binary pothole mask + transformed-disparity map per image, into `data/raw/pothole600/` |
| **RDD2022** | Direct S3 links from the official `sekilab/RoadDamageDetector` README (both the full `RDD2022.zip` and the per-country `RDD2022_India.zip`) | ❌ **Blocked** — the entire `bigdatacup.s3.ap-northeast-1.amazonaws.com` bucket now returns `403 Forbidden` to anonymous requests (verified with multiple objects, including files the README describes as directly linkable). The FigShare mirror sits behind an AWS WAF bot challenge that only a real browser can pass. Not fixable from this environment — needs a person to open the FigShare page in a browser (`https://figshare.com/articles/dataset/RDD2022...`) and download manually, or an S3 access request to the sekilab team |
| **Roboflow public pothole set (D5)** | Direct GET on the dataset's `/download/yolov8` URL | ❌ **Blocked** — returns the Roboflow app's HTML shell, not a zip; Roboflow's actual export endpoint requires a signed URL obtained via a logged-in session or an API key |
| **Indian Roads segmentation (D7)** | Not yet attempted this session | ⏳ Pending — needs the same manual/account-based check as above |
| **Kaggle sources (D2, D3)** | `kaggle` CLI installed, but no `kaggle.json` API token present at `~/.kaggle/` | ⏳ **Pending on you** — place your Kaggle API token at `~/.kaggle/kaggle.json` (Kaggle account → Settings → API → Create New Token) and the existing `kaggle` CLI can pull these directly |

## What this means for the rest of Phase 2/3

Every dataset host that isn't a plain public file link (Kaggle, Roboflow, and — as of now — even RDD2022's
S3 bucket and FigShare mirror) gates bulk/programmatic downloads behind an account, even when the dataset's
*license* is fully open. This is a real, external limitation of this environment, not a project-code issue.

Rather than block Phase 3 entirely, the pipeline was run on the one source that did download —
**Pothole-600** (582 images after cleaning/dedup, single class: `pothole`, real human-verified segmentation
masks). Everything downstream (`build_merged_dataset.py`, the YOLO-seg conversion, splits, training, and
evaluation) is written to be source-agnostic: re-running `python src/preprocessing/build_merged_dataset.py`
after any of RDD2022 / Roboflow / Indian Roads / Kaggle sources land in `data/raw/` picks them up
automatically (`convert_voc.py`, `convert_coco.py`, `convert_supervisely.py`, `convert_yolo_box.py` are
already written and unit-tested for exactly this — see [docs/phase1/06_project_scope.md](../phase1/06_project_scope.md)).

**Practical consequence:** the Phase 3 detection metrics in this round are computed on a 582-image,
single-source, single-class dataset — a real but much smaller and less diverse corpus than the multi-source,
multi-class merge the blueprint scoped. Treat current numbers as a pipeline validation baseline, not the
paper's final reported metrics. Re-run Phase 3 once the blocked sources are unblocked (Kaggle token supplied,
or RDD2022/Roboflow downloaded manually through a browser and dropped into `data/raw/`).

## Note on a dedup edge case found while merging

`dedup.py`'s perceptual-hash pass found 18 near-duplicate pairs inside Pothole-600 (582 of 600 images kept);
2 of those pairs spanned across the dataset's own train/validation/testing split (see
`data/merged/dedup_report.csv`). Dropping one image from a cross-split near-duplicate pair does not leak
labels between splits — the two images just stayed in their original splits, one was removed — but it's
worth a human glance at those 2 pairs to confirm they're truly redundant frames and not two genuinely
different potholes that happen to look similar at phash resolution.
