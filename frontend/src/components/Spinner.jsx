export default function Spinner({ label, size = "md" }) {
  return (
    <span className={`spinner-wrap spinner-${size}`}>
      <span className="spinner" aria-hidden="true" />
      {label && <span>{label}</span>}
    </span>
  );
}
