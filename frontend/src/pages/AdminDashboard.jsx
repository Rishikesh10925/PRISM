import { useEffect, useMemo, useState } from "react";
import { CircleMarker, MapContainer, Popup, TileLayer } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { fetchPriorityList, pdfReportUrl, updateStatus } from "../api";
import { priorityCategory, recomputePriority } from "../priorityFormula";
import ScoreBadge, { priorityColor, severityColor } from "../components/ScoreBadge";

const DEFAULT_WEIGHTS = { alpha: 0.4, beta: 0.3, gamma: 0.2, delta: 0.1 };
const DEFAULT_CENTER = [17.4239, 78.4738]; // Hyderabad -- used only when there's no data yet to center on

export default function AdminDashboard() {
  const [items, setItems] = useState([]);
  const [weights, setWeights] = useState(DEFAULT_WEIGHTS);
  const [statusFilter, setStatusFilter] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  async function reload() {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchPriorityList(statusFilter || undefined);
      setItems(data);
    } catch (err) {
      setError("Could not load the worklist. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [statusFilter]);

  // Live re-ranking: recompute priority for every row from its already-fetched raw
  // components as the sliders move -- no network call, no re-running detection.
  const ranked = useMemo(() => {
    return items
      .map((item) => {
        const score = recomputePriority(item, weights);
        return { ...item, live_priority_score: score, live_priority_category: priorityCategory(score) };
      })
      .sort((a, b) => b.live_priority_score - a.live_priority_score);
  }, [items, weights]);

  async function handleStatusChange(detectionId, newStatus) {
    await updateStatus(detectionId, newStatus);
    reload();
  }

  const mapCenter = ranked.length > 0 ? [ranked[0].latitude, ranked[0].longitude] : DEFAULT_CENTER;

  return (
    <div className="page admin-page">
      <h1>🗺️ Admin Repair Worklist</h1>

      {error && <p className="error-text">{error}</p>}

      <div className="admin-layout">
        <aside className="admin-sidebar">
          <h3>Priority weights</h3>
          <p className="hint">Adjust to see the worklist re-rank instantly.</p>
          <WeightSlider label="Pothole Severity" value={weights.alpha} onChange={(v) => setWeights({ ...weights, alpha: v })} />
          <WeightSlider label="Road Importance" value={weights.beta} onChange={(v) => setWeights({ ...weights, beta: v })} />
          <WeightSlider label="Traffic Level" value={weights.gamma} onChange={(v) => setWeights({ ...weights, gamma: v })} />
          <WeightSlider label="Previous Reports" value={weights.delta} onChange={(v) => setWeights({ ...weights, delta: v })} />
          <button className="secondary-button" onClick={() => setWeights(DEFAULT_WEIGHTS)}>Reset to default</button>

          <h3>Filter</h3>
          <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
            <option value="">All statuses</option>
            <option value="reported">Reported</option>
            <option value="in_progress">In progress</option>
            <option value="repaired">Repaired</option>
          </select>

          <a className="secondary-button pdf-link" href={pdfReportUrl()} target="_blank" rel="noreferrer">
            📄 Download PDF report
          </a>
        </aside>

        <main className="admin-main">
          <div className="map-wrap">
            <MapContainer center={mapCenter} zoom={12} style={{ height: 350, width: "100%" }}>
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution="&copy; OpenStreetMap contributors"
              />
              {ranked.map((item) => (
                <CircleMarker
                  key={item.detection_id}
                  center={[item.latitude, item.longitude]}
                  radius={8}
                  pathOptions={{ color: priorityColor(item.live_priority_category), fillOpacity: 0.8 }}
                >
                  <Popup>
                    <strong>Detection #{item.detection_id}</strong>
                    <br />
                    Severity: {item.severity_category} ({Math.round(item.severity_score)})
                    <br />
                    Priority: {item.live_priority_category} ({Math.round(item.live_priority_score)})
                    <br />
                    Status: {item.status}
                  </Popup>
                </CircleMarker>
              ))}
            </MapContainer>
          </div>

          {loading ? (
            <p>Loading worklist...</p>
          ) : (
            <table className="worklist-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Severity</th>
                  <th>Priority</th>
                  <th>Location</th>
                  <th>Reported</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {ranked.map((item, i) => (
                  <tr key={item.detection_id}>
                    <td>{i + 1}</td>
                    <td><ScoreBadge label={item.severity_category} colorFn={severityColor} /></td>
                    <td>
                      <ScoreBadge label={item.live_priority_category} colorFn={priorityColor} />{" "}
                      {Math.round(item.live_priority_score)}
                    </td>
                    <td>{item.latitude.toFixed(4)}, {item.longitude.toFixed(4)}</td>
                    <td>{new Date(item.submitted_at).toLocaleDateString()}</td>
                    <td>
                      <select
                        value={item.status}
                        onChange={(e) => handleStatusChange(item.detection_id, e.target.value)}
                      >
                        <option value="reported">Reported</option>
                        <option value="in_progress">In progress</option>
                        <option value="repaired">Repaired</option>
                      </select>
                    </td>
                  </tr>
                ))}
                {ranked.length === 0 && (
                  <tr>
                    <td colSpan={6}>No reports yet.</td>
                  </tr>
                )}
              </tbody>
            </table>
          )}
        </main>
      </div>
    </div>
  );
}

function WeightSlider({ label, value, onChange }) {
  return (
    <div className="weight-slider">
      <label>
        {label}: {value.toFixed(2)}
      </label>
      <input
        type="range"
        min="0"
        max="1"
        step="0.05"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  );
}
