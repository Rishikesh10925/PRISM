import { priorityTone, severityTone } from "../scoring";

// Subtle tinted chip with a coloured dot. Strong colour is reserved (via CSS) for the
// high/crit tones so a worklist full of low-severity rows stays calm.
export function CategoryChip({ category, kind = "severity", size = "md" }) {
  const tone = kind === "priority" ? priorityTone(category) : severityTone(category);
  return (
    <span className={`chip chip-${size} tone-${tone}`}>
      <span className="chip-dot" aria-hidden="true" />
      {category || "Unscored"}
    </span>
  );
}

// Compact 0-100 bar with the numeric score in tabular figures.
export function ScoreBar({ score, category, kind = "severity", showValue = true, label }) {
  const tone = kind === "priority" ? priorityTone(category) : severityTone(category);
  const value = Number.isFinite(score) ? Math.max(0, Math.min(100, score)) : 0;
  return (
    <span className={`score-bar tone-${tone}`}>
      {showValue && <span className="score-bar-value num">{Math.round(value)}</span>}
      <span
        className="score-bar-track"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(value)}
        aria-label={label || `${kind === "priority" ? "Priority" : "Severity"} score`}
      >
        <span className="score-bar-fill" style={{ width: `${value}%` }} />
      </span>
    </span>
  );
}

export default CategoryChip;
