# Phase 4 / Tasks 7-10 — Prioritization Engine

| Task | Module | What it computes |
|---|---|---|
| 7 | [formula.py](../../src/prioritization/formula.py) | `P = alpha*S + beta*RoadTypeWeight + gamma*TrafficProxy + delta*RecurrenceFactor`, rescaled to 0-100 (see the module docstring for why S/100 is used internally); `rank_potholes()` produces the sorted worklist |
| 8 | [road_type.py](../../src/prioritization/road_type.py) | Real OSM Overpass API lookup: nearest tagged road within a radius of a GPS point → `highway` tag → weight (primary/trunk/motorway 1.0, secondary/tertiary 0.7, residential/unclassified/service 0.4, default 0.4 for anything else) |
| 9 | [traffic_recurrence.py](../../src/prioritization/traffic_recurrence.py) | `TrafficProxy` = road-type weight × an hour-of-day rush-hour multiplier (no live traffic API, per blueprint's explicit fallback); `RecurrenceFactor` = saturating count of other reports within a real haversine-distance radius |
| 10 | [ablation.py](../../src/prioritization/ablation.py) | Compares default / severity-only / traffic-heavy weightings on real data (below) |

## Verifying Task 8 against a real, live API

`road_type.py`'s first version failed silently against the real Overpass API — `requests`' default
`python-requests/x.x` User-Agent gets a `406 Not Acceptable` from `overpass-api.de` (their fair-use policy
expects an identifying client string). Fixed by sending a real `User-Agent` header; verified against known
Manhattan coordinates in `tests/test_road_type.py`, which hit the live API. That endpoint is also rate-limited
under back-to-back test-suite load — `tests/test_road_type.py` retries with backoff, and `road_type_weight()`
itself is `lru_cache`d and never raises (falls back to `DEFAULT_WEIGHT` on any failure), so a slow/unavailable
OSM response degrades the live dashboard gracefully instead of blocking a report submission.

## Task 10 — Ablation: what's real and what's illustrative

Ran on 20 real Pothole-600 test images: severity scores are genuine `severity.pipeline.compute_severity()`
output (MiDaS depth proxy included), and road-type weights are genuine live OSM lookups against 5 real
Hyderabad-area coordinates. What's **not** real: this project has no live citizen-report GPS/timestamp data
yet (Pothole-600 images carry no geotags), so which of the 5 real coordinates each pothole is assigned to,
its report hour, and its recurrence count are seeded/deterministic assignments made purely to exercise the
formula across a range of contexts — not measured data. Re-run `ablation.py` once real citizen reports exist.

Full per-image rankings: [evaluation/prioritization_ablation.csv](../../evaluation/prioritization_ablation.csv).
Summary: [evaluation/prioritization_ablation_summary.csv](../../evaluation/prioritization_ablation_summary.csv).

| Weighting | Kendall's τ vs. default | Top-5 overlap with default |
|---|---|---|
| default (α .4 β .3 γ .2 δ .1) | 1.000 | 5/5 |
| severity-only (α 1, rest 0) | 0.821 | 5/5 |
| traffic-heavy (α .2 β .2 γ .5 δ .1) | 0.832 | 2/5 |

**The interesting finding:** severity-only and traffic-heavy get nearly identical τ against default (0.821 vs.
0.832) — so as a single summary number, they look equally "different" from default. But top-5 overlap tells a
different story: severity-only keeps all 5 of default's top potholes (just reorders them), while traffic-heavy
swaps 3 of the top 5 out entirely. τ, being a whole-ranking correlation, is dominated by agreement across the
long tail of lower-priority potholes and can mask exactly the reordering that matters most in practice — which
potholes an authority's crew sees at the top of today's worklist. This is why the ablation reports both
metrics rather than τ alone, and it's a concrete demonstration that the prioritization framework does genuine
multi-criteria reasoning rather than just re-sorting by severity (the demo moment described in blueprint
Section 10).

Concretely: `pothole600_testing_0163` is #1 in all three configs (high severity *and* high-priority context).
But traffic-heavy promotes `pothole600_testing_0131`, `_0116`, and `_0035` into the top 5 — each scored lower
on severity alone but sits at a busier assigned context — while default/severity-only's #3-4
(`pothole600_testing_0081`, `_0112`) drop out.
