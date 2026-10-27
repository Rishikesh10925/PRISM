import { ChevronDownIcon } from "./Icons";
import { STATUS_LABELS } from "../scoring";

// Native <select> (keyboard + mobile friendly) dressed as a compact status pill.
export default function StatusSelect({ value, onChange, disabled, label }) {
  return (
    <span className={`status-select status-${value}${disabled ? " is-disabled" : ""}`}>
      <span className="status-dot" aria-hidden="true" />
      <select
        value={value}
        disabled={disabled}
        aria-label={label}
        onChange={(e) => onChange(e.target.value)}
        onClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") e.stopPropagation();
        }}
      >
        {Object.entries(STATUS_LABELS).map(([v, l]) => (
          <option key={v} value={v}>
            {l}
          </option>
        ))}
      </select>
      <ChevronDownIcon size={14} className="status-chevron" />
    </span>
  );
}
