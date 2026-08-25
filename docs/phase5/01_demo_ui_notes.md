# Demo Dashboard — Non-Technical UI Redesign

The first version of [demo/app.py](../../demo/app.py) exposed raw technical values directly (α/β/γ/δ
weight sliders and percentages, area ratio, MiDaS depth numbers, contour irregularity, detection
confidence, raw OSM road-type weight, recurrence factor) — fine for a developer, unreadable for anyone
else. Redesigned so a non-technical viewer only ever sees plain language, while every number is still
computed exactly the same way underneath.

## What changed (display layer only — no calculation logic changed)

- **Severity is never a user input.** It's shown as a read-only `Severity Score (0–100)` plus a level —
  `Very Low / Low / Medium / High / Critical` — computed by `severity.fusion.severity_category()`, extended
  from 4 to 5 evenly-spaced bands (0-20/20-40/40-60/60-80/80-100) so there's room for a genuinely mild case
  below "Low". The score itself (`severity_score()`) is untouched.
- **Priority Score (0–100)** gets its own plain level — `Less Important / Moderate / Important / Very
  Important` — via the new `prioritization.formula.priority_category()`. The formula (`priority_score()`)
  is untouched.
- **α/β/γ/δ and their percentages are gone from the normal view.** The four factors that feed priority are
  shown instead as plain qualitative rows: **Pothole Severity**, **Road Importance**, **Traffic Level**,
  **Previous Reports** — each mapped from the real underlying number via a new small labeling function
  (`road_type.road_importance_label()`, `traffic_recurrence.traffic_level_label()`,
  `traffic_recurrence.recurrence_level_label()`). The numeric weights (default α=0.4, β=0.3, γ=0.2, δ=0.1,
  matching the blueprint) are fixed internally and no longer presented as adjustable percentages on the main
  screen.
- **Context is auto-detected where a real automatic source exists, not faked where one doesn't:**
  - *Time* — the system clock, genuinely automatic.
  - *Location* — real browser geolocation (`navigator.geolocation`), requested once per page load via a
    small injected script that writes the coordinates into the URL and reloads; if the user's browser
    doesn't support it or they decline the permission prompt, the page falls back to a documented default
    location rather than blocking. See `resolve_location()`.
  - *Road importance* — a real live OpenStreetMap lookup at whatever location was resolved (unchanged from
    before).
  - *Previous reports* — **cannot** be genuinely automatic yet: there is no live citizen-report database
    (Phase 5's backend/DB don't exist). Defaults to "1 report (this one)" = **None** rather than fabricating
    a number; a manual override slider is available under Admin for demonstrating the ranking's sensitivity
    to recurrence.
- **All raw technical values are still there**, just moved into a collapsed "🔧 Admin / developer details"
  expander per result card (and a separate "⚙️ Admin / developer details" expander in the sidebar for the
  weight sliders, the depth-estimation toggle, and the manual location override) — nothing was deleted, only
  hidden by default.

## What did not change

`severity_score()`, `priority_score()`, the fusion weights/normalization bounds, and the detection model
itself are byte-for-byte the same as before this UI pass. The five/four-level category thresholds are a new
display-layer mapping on top of the existing continuous scores, verified against their boundaries in
`src/severity/tests/test_fusion.py::test_severity_category_boundaries_and_full_range` and
`src/prioritization/tests/test_display_labels.py`.

## A real bug this session: stale cached model after switching checkpoints

After retraining on the multi-source dataset, `demo/app.py`'s `MODEL_PATH` constant was updated to point at
the new checkpoint — but the dashboard kept detecting almost nothing on any of the new Kaggle-source sample
photos (Pothole-600 samples still worked). Root cause: `load_model()` was decorated `@st.cache_resource` with
**no arguments**, so Streamlit's resource cache key never changed when `MODEL_PATH` did — the app kept
serving the *old*, already-cached `YOLO` object (trained on Pothole-600 only, which never saw the Kaggle
sources' distribution) even though the footer text correctly printed the new checkpoint's filename (that's
just string interpolation re-evaluated fresh each render, unrelated to which cached model object was
actually running inference). A live 10-sample sweep through the real browser UI showed 6/10 "not detected"
before the fix, 9/10 correct (the 10th being a normal model miss, matching the ~93% hit rate measured via
direct inference) after. **Fix:** pass the model path as an explicit argument to the cached function
(`load_model(model_path: str)`), so the cache key changes whenever the checkpoint does — and restarting the
Streamlit process (not just letting its file-watcher hot-reload `app.py`) is required after this kind of
change, since `st.cache_resource` values persist across script reloads within the same process.
