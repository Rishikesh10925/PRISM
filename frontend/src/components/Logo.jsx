// PRISM mark: a road in perspective (edges converging on the horizon) with a
// dashed centre lane. Uses the accent token so it themes with the app.

export function PrismMark({ size = 28 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      className="prism-mark"
      aria-hidden="true"
      focusable="false"
    >
      <rect width="32" height="32" rx="8" fill="var(--accent)" />
      {/* road edges converging to the horizon, with a dashed centre lane */}
      <path
        d="M8.5 25 14 8M23.5 25 18 8"
        fill="none"
        stroke="var(--on-accent)"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M16 23.5v-3.2M16 17.4v-2.4M16 12.6v-1.6"
        stroke="var(--on-accent)"
        strokeWidth="1.8"
        strokeLinecap="round"
        opacity="0.85"
      />
    </svg>
  );
}

export default function Logo({ tagline = false }) {
  return (
    <span className="logo">
      <PrismMark />
      <span className="logo-text">
        <span className="logo-word">PRISM</span>
        {tagline && <span className="logo-tagline">Pothole Repair Intelligence</span>}
      </span>
    </span>
  );
}
