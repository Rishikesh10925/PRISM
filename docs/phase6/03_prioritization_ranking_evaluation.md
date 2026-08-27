# Phase 6 / Task 3 — Prioritization Ranking Evaluation

Implemented by extending [src/prioritization/ablation.py](../../src/prioritization/ablation.py)
(originally Phase 4 Task 10's weight-sensitivity ablation). Run on 60 real Pothole-600 test
images with real pothole masks; severity comes from the real pipeline (MiDaS included), road-type
weight from real live OSM Overpass lookups. What's *not* real, same as Phase 4: this project has
no live citizen-report GPS/timestamp data yet, so which real Hyderabad coordinate each sampled
pothole "sits at" and its report-hour/recurrence-count are assigned illustratively (seeded,
documented, not fabricated as if collected) — see the module docstring.

## Kendall's tau vs. a simulated "ideal" order

No real domain expert was available to produce a genuine ideal repair order. Rather than invent
one, the severity-only ranking (rank purely by how bad the pothole is, ignoring road/traffic/
recurrence context) is used as the simulated reference — the simplest defensible notion of repair
urgency absent human judgment. The table below reports Kendall's tau and top-5 overlap between
each weighting config and the **default** config; the `severity_only` row's numbers are exactly
that tau/overlap against the simulated reference order.

| Config | Kendall's tau vs. default | Top-5 overlap vs. default |
|---|---|---|
| default | 1.000 | 5/5 |
| severity_only (≈ simulated ideal order) | 0.400 | 2/5 |
| traffic_heavy | 0.790 | 5/5 |

Reading this honestly: the default weighting (severity 40% / road-type 30% / traffic 20% /
recurrence 10%) reorders the worklist substantially relative to ranking by severity alone
(tau=0.400, only 2 of the top 5 agree) — context genuinely changes which pothole gets fixed
first, which is the entire point of the priority formula rather than just triaging by severity.
`traffic_heavy` stays closer to `default` (tau=0.790) since it shares more of default's context
weighting.

## Top-K precision for "Critical" severity flags

Of the top-K potholes by priority score, what fraction are severity-category "Critical"?

| Config | Top-5 critical precision | Top-10 critical precision |
|---|---|---|
| default | 0.000 | 0.000 |
| severity_only | 0.000 | 0.000 |
| traffic_heavy | 0.000 | 0.000 |

**This is 0.000 across the board because zero of the 60 sampled real instances were classified
"Critical"** under the current fusion thresholds (28 Very Low, 26 Low, 3 Medium, 3 High, 0
Critical) — not a ranking failure. This is a direct, honest consequence of Task 2's blocker: the
severity fusion's normalization bounds (`DEFAULT_DEPTH_MAX=30.0`,
`DEFAULT_IRREGULARITY_MAX=5.0` in [fusion.py](../../src/severity/fusion.py)) are uncalibrated
placeholders, not tuned against real ratings — so real category thresholds may currently sit
higher than they should. Re-run this metric once Task 6 calibration is unblocked; it may look
very different once the thresholds are real.

Raw per-image output: `evaluation/prioritization_ablation.csv` /
`evaluation/prioritization_ablation_summary.csv`.
