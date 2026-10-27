// Display helpers for scores: category -> tone, plain-language cue explanations,
// priority-term breakdowns and date formatting. Pure functions, no React.

const SEVERITY_TONES = {
  "Very Low": "low",
  Low: "low",
  Medium: "mid",
  High: "high",
  Critical: "crit",
};

const PRIORITY_TONES = {
  "Less Important": "low",
  Moderate: "mid",
  Important: "high",
  "Very Important": "crit",
};

export function severityTone(category) {
  return SEVERITY_TONES[category] || "none";
}

export function priorityTone(category) {
  return PRIORITY_TONES[category] || "none";
}

// Marker fills for the map (Leaflet needs real colour values, not CSS classes). Kept in
// step with the --tone-*-dot tokens in index.css.
export const TONE_COLORS = {
  low: "#4f8a6b",
  mid: "#b7832a",
  high: "#d0632a",
  crit: "#c2342f",
  none: "#7b8798",
};

// Severity cue bounds. The backend scores with evaluation/severity/cue_bounds.json
// (fitted on the train split; see src/severity/fusion.py): each cue is clipped-linear
// to [0, 1] and the three are averaged with equal weight, then x100. Reports stored
// before that change used the legacy bounds (raw-unit depth). explainSeverity tries
// both and uses whichever reproduces the stored score.
const CUE_CONFIGS = [
  { name: "fitted", areaMax: 0.4087675589767157, depthMax: 0.05084861595137472, shadowDepthMax: 0.25, irregularityMax: 3.5462374546254494 },
  { name: "legacy", areaMax: 1.0, depthMax: 30, shadowDepthMax: 30, irregularityMax: 5.0 },
];
const CUE_WEIGHT = 1 / 3;

const clamp01 = (x) => (Number.isFinite(x) ? Math.min(1, Math.max(0, x)) : 0);

function normaliseCues(sev, cfg) {
  const depthMax = sev.depth_source === "shadow_heuristic" ? cfg.shadowDepthMax : cfg.depthMax;
  const area = clamp01(sev.area_ratio / cfg.areaMax);
  const depth = sev.depth_source === "none" ? 0 : clamp01(sev.depth_value / depthMax);
  const irregularity = clamp01((sev.irregularity - 1) / (cfg.irregularityMax - 1));
  return { area, depth, irregularity, total: 100 * CUE_WEIGHT * (area + depth + irregularity) };
}

function levelWord(n) {
  if (n < 0.15) return "minimal";
  if (n < 0.4) return "modest";
  if (n < 0.7) return "notable";
  return "pronounced";
}

function depthAdverb(n) {
  if (n < 0.15) return "slightly";
  if (n < 0.4) return "moderately";
  if (n < 0.7) return "noticeably";
  return "much";
}

const DEPTH_SOURCE_LABELS = {
  midas: "MiDaS monocular depth model",
  shadow_heuristic: "shadow-contrast fallback",
  none: "not available for this detection",
};

/**
 * Breaks a severity_score object into the three cues, each with a normalised level
 * (0..1), its share of the 0-100 score, and a plain-language sentence. `consistent`
 * says whether the recomputed total matches the stored score -- if the backend ever
 * scores with different bounds, the UI shows the levels but not the point split.
 */
export function explainSeverity(sev) {
  if (!sev) return null;
  const candidates = CUE_CONFIGS.map((cfg) => normaliseCues(sev, cfg));
  const match = candidates.find((c) => Math.abs(c.total - sev.score) < 1.5);
  const { area: areaN, depth: depthN, irregularity: irregN } = match || candidates[0];
  const pct = (sev.area_ratio * 100).toFixed(sev.area_ratio < 0.01 ? 2 : 1);
  const rawDepth = Math.abs(sev.depth_value) >= 1 ? sev.depth_value.toFixed(1) : sev.depth_value.toFixed(3);

  const cues = [
    {
      key: "area",
      label: "Size in the photo",
      level: areaN,
      points: 100 * CUE_WEIGHT * areaN,
      raw: `${pct}% of frame`,
      text: `The outlined area covers ${pct}% of the photo — ${levelWord(areaN)} relative to the potholes PRISM was calibrated on.`,
    },
    {
      key: "depth",
      label: "Estimated depth",
      level: depthN,
      points: 100 * CUE_WEIGHT * depthN,
      raw: sev.depth_source === "none" ? "n/a" : rawDepth,
      text:
        sev.depth_source === "none"
          ? "Depth could not be estimated, so this cue adds nothing to the score."
          : `The pothole reads ${depthN <= 0 ? "no deeper than" : `${depthAdverb(depthN)} deeper than`} the surrounding road (estimated by the ${DEPTH_SOURCE_LABELS[sev.depth_source] || sev.depth_source}).`,
    },
    {
      key: "irregularity",
      label: "Edge irregularity",
      level: irregN,
      points: 100 * CUE_WEIGHT * irregN,
      raw: sev.irregularity.toFixed(2),
      text: `Edge roughness ${sev.irregularity.toFixed(2)} (1.00 is a smooth circle) — ${levelWord(irregN)} breakup, which often signals crumbling edges.`,
    },
  ];
  return { cues, consistent: Boolean(match) };
}

/**
 * Each weighted term's contribution to a priority score, in points out of 100. Uses
 * the same arithmetic as priorityFormula.recomputePriority so the parts sum to it.
 */
export function priorityTerms(item, weights) {
  const { alpha, beta, gamma, delta } = weights;
  const total = alpha + beta + gamma + delta;
  const terms = [
    { key: "alpha", label: "Severity", weight: alpha, value: item.severity_score / 100, display: `${Math.round(item.severity_score)}/100` },
    { key: "beta", label: "Road importance", weight: beta, value: item.road_type_weight, display: fmt2(item.road_type_weight) },
    { key: "gamma", label: "Traffic level", weight: gamma, value: item.traffic_proxy, display: fmt2(item.traffic_proxy) },
    { key: "delta", label: "Previous reports", weight: delta, value: item.recurrence_factor, display: fmt2(item.recurrence_factor) },
  ];
  return terms.map((t) => ({
    ...t,
    points: total > 0 ? (100 * t.weight * (Number.isFinite(t.value) ? t.value : 0)) / total : 0,
    share: total > 0 ? t.weight / total : 0,
  }));
}

function fmt2(x) {
  return Number.isFinite(x) ? x.toFixed(2) : "–";
}

export const ROAD_SOURCE_LABELS = {
  osm: "Road type looked up on OpenStreetMap",
  default_no_road: "No mapped road nearby — default road weight used",
  default_api_error: "OpenStreetMap lookup failed — default road weight used",
};

export const TIME_SOURCE_LABELS = {
  exif: "Traffic estimated from the photo's capture time (EXIF)",
  server: "Traffic estimated from the upload time (photo had no capture time)",
};

export const STATUS_LABELS = {
  reported: "Reported",
  in_progress: "In progress",
  repaired: "Repaired",
};

const rtf = typeof Intl !== "undefined" && Intl.RelativeTimeFormat ? new Intl.RelativeTimeFormat("en", { numeric: "auto" }) : null;

export function relativeTime(iso, now = Date.now()) {
  const t = new Date(iso).getTime();
  if (!Number.isFinite(t)) return "–";
  const diff = (t - now) / 1000;
  const abs = Math.abs(diff);
  const steps = [
    [60, "second", 1],
    [3600, "minute", 60],
    [86400, "hour", 3600],
    [86400 * 30, "day", 86400],
    [86400 * 365, "month", 86400 * 30],
    [Infinity, "year", 86400 * 365],
  ];
  for (const [limit, unit, div] of steps) {
    if (abs < limit) {
      if (unit === "second") return "just now";
      const v = Math.round(diff / div);
      return rtf ? rtf.format(v, unit) : `${Math.abs(v)} ${unit}s ago`;
    }
  }
  return "–";
}

export function fullDate(iso) {
  const d = new Date(iso);
  return Number.isFinite(d.getTime())
    ? d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })
    : "";
}

export function osmUrl(lat, lon) {
  return `https://www.openstreetmap.org/?mlat=${lat}&mlon=${lon}#map=18/${lat}/${lon}`;
}
