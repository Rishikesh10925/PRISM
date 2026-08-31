import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { CircleMarker, MapContainer, Popup, TileLayer } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { downloadPdfReport, fetchPriorityList, updateStatus } from "../api";
import { priorityCategory, recomputePriority } from "../priorityFormula";
import { clearToken, getUsername } from "../auth";
import ScoreBadge, { priorityColor, severityColor } from "../components/ScoreBadge";
import Spinner from "../components/Spinner";

const DEFAULT_WEIGHTS = { alpha: 0.4, beta: 0.3, gamma: 0.2, delta: 0.1 };
const DEFAULT_CENTER = [17.4239, 78.4738]; // Hyderabad -- used only when there's no data yet to center on

export default function AdminDashboard() {
  const navigate = useNavigate();
  const [items, setItems] = useState([]);
  const [weights, setWeights] = useState(DEFAULT_WEIGHTS);
  const [statusFilter, setStatusFilter] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [updatingId, setUpdatingId] = useState(null);
  const [downloadingPdf, setDownloadingPdf] = useState(false);

  function handleUnauthorized(err) {
    if (err.response?.status === 401) {
      navigate("/admin/login", { replace: true });
      return true;
    }
    return false;
  }

  async function reload() {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchPriorityList(statusFilter || undefined);
      setItems(data);
    } catch (err) {
      if (!handleUnauthorized(err)) {
        setError("Could not load the worklist. Is the backend running?");
      }
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
    setUpdatingId(detectionId);
    try {
      await updateStatus(detectionId, newStatus);
      await reload();
    } catch (err) {
      if (!handleUnauthorized(err)) {
        setError("Could not update status. Is the backend running?");
      }
    } finally {
      setUpdatingId(null);
    }
  }

  async function handleDownloadPdf() {
    setDownloadingPdf(true);
    setError(null);
    try {
      await downloadPdfReport();
    } catch (err) {
      if (!handleUnauthorized(err)) {
        setError("Could not generate the PDF report.");
      }
    } finally {
      setDownloadingPdf(false);
    }
  }

  function handleLogout() {
    clearToken();
    navigate("/admin/login", { replace: true });
  }

  const mapCenter = ranked.length > 0 ? [ranked[0].latitude, ranked[0].longitude] : DEFAULT_CENTER;

  const criticalCount = ranked.filter((item) => item.severity_category === "Critical").length;
  const urgentCount = ranked.filter(
    (item) => item.live_priority_category === "Important" || item.live_priority_category === "Very Important"
  ).length;

  const username = getUsername() || "admin";

  return (
    <div className="page admin-page">
      <div className="admin-toolbar">
        <div className="admin-toolbar-title">
          <h1>Repair Worklist</h1>
          {!loading && ranked.length > 0 && (
            <div className="stat-row">
              <span className="stat-pill">{ranked.length} report{ranked.length === 1 ? "" : "s"}</span>
              <span className="stat-pill stat-pill-critical">{criticalCount} critical severity</span>
              <span className="stat-pill stat-pill-urgent">{urgentCount} high priority</span>
            </div>
          )}
        </div>
        <div className="admin-user-menu">
          <span className="admin-avatar">{username.charAt(0).toUpperCase()}</span>
          <span className="admin-username">{username}</span>
          <button className="ghost-button" onClick={handleLogout}>
            Sign out
          </button>
        </div>
      </div>

      {error && <p className="error-text">{error}</p>}

      <div className="admin-layout">
        <aside className="admin-sidebar">
          <section>
            <h3>Priority weights</h3>
            <p className="hint">Adjust to see the worklist re-rank instantly.</p>
            <WeightSlider label="Pothole Severity" value={weights.alpha} onChange={(v) => setWeights({ ...weights, alpha: v })} />
            <WeightSlider label="Road Importance" value={weights.beta} onChange={(v) => setWeights({ ...weights, beta: v })} />
            <WeightSlider label="Traffic Level" value={weights.gamma} onChange={(v) => setWeights({ ...weights, gamma: v })} />
            <WeightSlider label="Previous Reports" value={weights.delta} onChange={(v) => setWeights({ ...weights, delta: v })} />
            <button className="secondary-button" onClick={() => setWeights(DEFAULT_WEIGHTS)}>Reset to default</button>
          </section>

          <section>
            <h3>Filter</h3>
            <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
              <option value="">All statuses</option>
              <option value="reported">Reported</option>
              <option value="in_progress">In progress</option>
              <option value="repaired">Repaired</option>
            </select>
          </section>

          <button className="secondary-button pdf-link" onClick={handleDownloadPdf} disabled={downloadingPdf}>
            {downloadingPdf ? (
              <>
                <Spinner size="sm" /> Generating...
              </>
            ) : (
              "📄 Download PDF report"
            )}
          </button>
        </aside>

        <main className="admin-main">
          <div className="map-wrap">
            {loading && (
              <div className="map-loading-overlay">
                <Spinner label="Loading map data..." />
              </div>
            )}
            <MapContainer center={mapCenter} zoom={12} style={{ height: "100%", width: "100%" }}>
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
            <div className="table-skeleton">
              {[0, 1, 2, 3].map((i) => (
                <div className="skeleton-row" key={i} />
              ))}
            </div>
          ) : (
            <div className="table-scroll">
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
                    <tr key={item.detection_id} className={updatingId === item.detection_id ? "row-updating" : ""}>
                      <td>{i + 1}</td>
                      <td><ScoreBadge label={item.severity_category} colorFn={severityColor} /></td>
                      <td>
                        <ScoreBadge label={item.live_priority_category} colorFn={priorityColor} />{" "}
                        {Math.round(item.live_priority_score)}
                      </td>
                      <td>{item.latitude.toFixed(4)}, {item.longitude.toFixed(4)}</td>
                      <td>{new Date(item.submitted_at).toLocaleDateString()}</td>
                      <td className="status-cell">
                        <select
                          value={item.status}
                          disabled={updatingId === item.detection_id}
                          onChange={(e) => handleStatusChange(item.detection_id, e.target.value)}
                        >
                          <option value="reported">Reported</option>
                          <option value="in_progress">In progress</option>
                          <option value="repaired">Repaired</option>
                        </select>
                        {updatingId === item.detection_id && <Spinner size="sm" />}
                      </td>
                    </tr>
                  ))}
                  {ranked.length === 0 && (
                    <tr>
                      <td colSpan={6} className="empty-row">No reports yet — submit one from the citizen upload page.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
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
