# Phase 6 / Task 4 — Severity Fusion Ablation

The most important ablation per the work plan. Implemented in
[src/severity/fusion_ablation.py](../../src/severity/fusion_ablation.py): real (area, depth, irregularity) cues
computed once per real pothole instance (100 real test-set images, MiDaS included), then re-fused
under four weight configurations so the comparison isolates "which cues are included" from any
randomness in cue extraction itself.

Because no severity validation subset exists yet (Task 2), this can't say which config is more
*accurate* — only how much each cue moves the score/category relative to the full model, on real
images. Full results: `evaluation/severity_fusion_ablation.csv` /
`evaluation/severity_fusion_ablation_summary.csv`.

| Config | Spearman's rho vs. full | Kendall's tau vs. full | Mean abs. score diff | Category agreement |
|---|---|---|---|---|
| full (area+depth+irregularity) | 1.000 | 1.000 | 0.00 | 1.000 |
| area_only | 0.584 | 0.406 | 11.19 | 0.580 |
| area+depth | 0.805 | 0.628 | 8.64 | 0.710 |
| area+irregularity | 0.817 | 0.686 | 8.37 | 0.540 |

## Reading this

- **Area alone diverges most from the full model** (rho=0.584, only 58% category agreement) —
  area by itself misses over 40% of what depth and irregularity together contribute to the
  ranking. This is the clearest evidence in the project so far that the multi-cue fusion is doing
  real work, not just area with two redundant extra terms.
- **Adding either depth or irregularity to area** closes most of the gap (rho 0.805 / 0.817 vs.
  0.584 for area alone) — the two three-cue-vs-two-cue configs are much closer to the full model
  than area-only is to either of them.
- **area+irregularity has the highest rank correlation (rho=0.817) but the lowest category
  agreement (0.540)** of the two-cue configs — it tracks the full model's *ordering* well but
  crosses category-band boundaries more often, i.e. it agrees on relative severity more than on
  absolute severity tier. area+depth is the more balanced of the two on this small sample.
- No config perfectly reproduces the full model except the full model itself (as expected — this
  is a sanity check the unit tests also verify: `src/severity/tests/test_fusion_ablation.py`).
