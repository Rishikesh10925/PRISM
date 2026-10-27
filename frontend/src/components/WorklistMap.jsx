import { useEffect, useMemo } from "react";
import { CircleMarker, MapContainer, Popup, TileLayer, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { priorityTone, STATUS_LABELS, TONE_COLORS } from "../scoring";

const DEFAULT_CENTER = [17.4239, 78.4738]; // Hyderabad -- only when there's nothing to centre on

function FitBounds({ points, fitKey }) {
  const map = useMap();
  useEffect(() => {
    if (points.length === 0) return;
    if (points.length === 1) {
      map.setView(points[0], 15);
    } else {
      map.fitBounds(points, { padding: [36, 36], maxZoom: 16 });
    }
    // Refit only when the set of markers changes, not on every re-rank.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fitKey, map]);
  return null;
}

function FollowSelection({ selected }) {
  const map = useMap();
  useEffect(() => {
    if (!selected) return;
    const ll = [selected.latitude, selected.longitude];
    if (!map.getBounds().pad(-0.1).contains(ll)) map.panTo(ll, { animate: true });
  }, [selected, map]);
  return null;
}

export default function WorklistMap({ rows, selectedId, onSelect, loading }) {
  const points = useMemo(() => rows.map((r) => [r.latitude, r.longitude]), [rows]);
  const fitKey = useMemo(() => rows.map((r) => r.detection_id).sort((a, b) => a - b).join(","), [rows]);
  const selected = rows.find((r) => r.detection_id === selectedId) || null;

  // Draw the selected marker last so it sits on top of overlapping neighbours.
  const ordered = useMemo(() => {
    const rest = rows.filter((r) => r.detection_id !== selectedId);
    return selected ? [...rest].reverse().concat(selected) : [...rest].reverse();
  }, [rows, selectedId, selected]);

  return (
    <div className="map-wrap">
      {loading && (
        <div className="map-overlay">
          <span className="skeleton-pulse map-overlay-text">Loading map…</span>
        </div>
      )}
      <MapContainer center={DEFAULT_CENTER} zoom={12} className="map" scrollWheelZoom={false}>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution="&copy; OpenStreetMap contributors"
        />
        <FitBounds points={points} fitKey={fitKey} />
        <FollowSelection selected={selected} />
        {ordered.map((item) => {
          const isSel = item.detection_id === selectedId;
          const color = TONE_COLORS[priorityTone(item.live_priority_category)];
          const radius = 6 + (Math.max(0, Math.min(100, item.live_priority_score)) / 100) * 10;
          return (
            <CircleMarker
              key={item.detection_id}
              center={[item.latitude, item.longitude]}
              radius={isSel ? radius + 3 : radius}
              pathOptions={{
                color: isSel ? "#0f1b2d" : "#ffffff",
                weight: isSel ? 3 : 1.5,
                fillColor: color,
                fillOpacity: isSel ? 0.95 : 0.78,
              }}
              eventHandlers={{ click: () => onSelect(item.detection_id) }}
            >
              <Popup className="prism-popup" closeButton={false}>
                <div className="popup">
                  <div className="popup-title">
                    #{item.rank} · Detection {item.detection_id}
                  </div>
                  <div className="popup-row">
                    <span>Priority</span>
                    <strong className="num">{Math.round(item.live_priority_score)}</strong>
                    <span className="popup-cat">{item.live_priority_category}</span>
                  </div>
                  <div className="popup-row">
                    <span>Severity</span>
                    <strong className="num">{Math.round(item.severity_score)}</strong>
                    <span className="popup-cat">{item.severity_category}</span>
                  </div>
                  <div className="popup-row">
                    <span>Status</span>
                    <span className="popup-cat">{STATUS_LABELS[item.status] || item.status}</span>
                  </div>
                </div>
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>
      <div className="map-legend" aria-label="Map legend">
        <span className="map-legend-title">Priority</span>
        {[
          ["crit", "Very important"],
          ["high", "Important"],
          ["mid", "Moderate"],
          ["low", "Less important"],
        ].map(([tone, label]) => (
          <span key={tone} className="map-legend-item">
            <span className="map-legend-dot" style={{ background: TONE_COLORS[tone] }} />
            {label}
          </span>
        ))}
        <span className="map-legend-note">Size = score</span>
      </div>
    </div>
  );
}
