const SEVERITY_COLORS = {
  "Very Low": "#2e7d32",
  Low: "#66bb6a",
  Medium: "#f9a825",
  High: "#ef6c00",
  Critical: "#c62828",
};

const PRIORITY_COLORS = {
  "Less Important": "#2e7d32",
  Moderate: "#f9a825",
  Important: "#ef6c00",
  "Very Important": "#c62828",
};

export function severityColor(category) {
  return SEVERITY_COLORS[category] || "#666";
}

export function priorityColor(category) {
  return PRIORITY_COLORS[category] || "#666";
}

export default function ScoreBadge({ label, colorFn }) {
  return (
    <span
      style={{
        backgroundColor: colorFn(label),
        color: "white",
        padding: "4px 12px",
        borderRadius: 12,
        fontWeight: 600,
        fontSize: "0.85rem",
        display: "inline-block",
      }}
    >
      {label}
    </span>
  );
}
