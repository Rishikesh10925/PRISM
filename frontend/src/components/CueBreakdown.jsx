import { explainSeverity } from "../scoring";

// "Why this score": the three severity cues in plain language, each with a level bar
// and (when the recomputation matches the stored score) its points out of 100.
export default function CueBreakdown({ severity }) {
  const explained = explainSeverity(severity);
  if (!explained) return <p className="muted small">No severity breakdown is available for this detection.</p>;
  const { cues, consistent } = explained;

  return (
    <div className="cues">
      <p className="cues-intro">
        Severity averages three visual cues, each worth up to 33 points.
      </p>
      <ul className="cue-list">
        {cues.map((c) => (
          <li key={c.key} className="cue">
            <div className="cue-head">
              <span className="cue-label">{c.label}</span>
              <span className="cue-raw num">{c.raw}</span>
              {consistent && <span className="cue-points num">+{c.points.toFixed(1)}</span>}
            </div>
            <span className="cue-track" aria-hidden="true">
              <span className="cue-fill" style={{ width: `${Math.round(c.level * 100)}%` }} />
            </span>
            <p className="cue-text">{c.text}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}
