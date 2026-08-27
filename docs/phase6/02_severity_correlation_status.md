# Phase 6 / Task 2 — Severity Correlation Evaluation: Blocked, Not Skipped

The plan asks for Spearman's rho and MAE between predicted Severity Score and human ratings on
the validation subset. **This cannot be computed yet because that validation subset does not
exist.**

This is the same blocker already documented in
[docs/phase4/01_severity_modules.md](../phase4/01_severity_modules.md) (Phase 4 Task 6) and
[docs/phase2/01_dataset_download_status.md](../phase2/01_dataset_download_status.md) (Phase 2
Tasks 7-9): building `data/severity_val_subset/` requires selecting 150-250 images and recruiting
2-3 real human raters to independently score them — recruiting real people and collecting real
subjective judgments isn't something that can be done or simulated from here. `data/severity_val_subset/`
is still empty (`.gitkeep` only) as of this phase; nothing has changed on that front since Phase 4.

**What was not done instead:** fabricating placeholder ratings to produce a rho/MAE number. That
would misrepresent the project's central severity-validation claim with a number that looks like
evidence but isn't.

**What was done instead:** [04_severity_fusion_ablation.md](04_severity_fusion_ablation.md)
(Phase 6 Task 4) evaluates how the fusion weights affect scores and rankings *relative to each
other* on real images, which doesn't require ground truth and is a legitimate, honest
substitute for "which config is right" until real ratings exist.

**To finish Task 2 for real:** collect the human ratings (Phase 2 Tasks 7-9), then run
[`calibrate.py`](../../src/severity/calibrate.py) — already implemented and verified correct
against synthetic data — against the resulting `(predicted_score, human_rating)` pairs. No further
code changes should be needed.
