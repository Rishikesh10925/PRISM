import { useState } from "react";
import { submitReport } from "../api";
import ScoreBadge, { severityColor, priorityColor } from "../components/ScoreBadge";
import Spinner from "../components/Spinner";

export default function CitizenUpload() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [coords, setCoords] = useState(null);
  const [geoStatus, setGeoStatus] = useState("idle"); // idle | locating | ok | denied
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  function handleFileChange(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
    captureLocation();
  }

  function captureLocation() {
    if (!navigator.geolocation) {
      setGeoStatus("denied");
      return;
    }
    setGeoStatus("locating");
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setCoords({ lat: pos.coords.latitude, lon: pos.coords.longitude });
        setGeoStatus("ok");
      },
      () => setGeoStatus("denied"),
      { enableHighAccuracy: true, timeout: 10000 }
    );
  }

  async function handleSubmit() {
    if (!file || !coords) return;
    setSubmitting(true);
    setError(null);
    try {
      const data = await submitReport(file, coords.lat, coords.lon);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || "Something went wrong submitting your report. Please try again.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="page citizen-page">
      <h1>🛣️ Report a Pothole</h1>
      <p className="subtitle">Take or upload a photo of the pothole — we'll detect it and assess how urgent it is.</p>

      {!result && (
        <div className="card">
          <label className={`upload-box${submitting ? " upload-box-disabled" : ""}`}>
            <input
              type="file"
              accept="image/*"
              capture="environment"
              onChange={handleFileChange}
              disabled={submitting}
              hidden
            />
            {preview ? (
              <img src={preview} alt="preview" className="preview-img" />
            ) : (
              <span>📷 Tap to take or choose a photo</span>
            )}
          </label>

          <div className="geo-status">
            {geoStatus === "idle" && "Location will be captured automatically when you pick a photo."}
            {geoStatus === "locating" && (
              <span className="geo-locating">
                <Spinner size="sm" /> Getting your location...
              </span>
            )}
            {geoStatus === "ok" && coords && `📍 Location captured: ${coords.lat.toFixed(4)}, ${coords.lon.toFixed(4)}`}
            {geoStatus === "denied" && (
              <span className="geo-warning">
                ⚠️ Couldn't get your location — please enable location access and retry.{" "}
                <button className="link-button" onClick={captureLocation}>Try again</button>
              </span>
            )}
          </div>

          {error && <p className="error-text">{error}</p>}

          <button
            className="primary-button"
            disabled={!file || !coords || submitting}
            onClick={handleSubmit}
          >
            {submitting ? (
              <>
                <Spinner size="sm" /> Analyzing photo...
              </>
            ) : (
              "Submit Report"
            )}
          </button>
        </div>
      )}

      {result && (
        <div className="card">
          <h2>✅ Report submitted</h2>
          <img src={preview} alt="submitted" className="preview-img" />
          {result.detections.length === 0 ? (
            <p>No potholes were detected in this photo.</p>
          ) : (
            result.detections.map((d) => (
              <div key={d.id} className="detection-summary">
                <div className="badge-row">
                  <ScoreBadge label={d.severity_score.category} colorFn={severityColor} />
                  <ScoreBadge label={d.priority_score.category} colorFn={priorityColor} />
                </div>
                <p>Severity score: {Math.round(d.severity_score.score)} / 100</p>
                <p>Priority score: {Math.round(d.priority_score.score)} / 100</p>
              </div>
            ))
          )}
          <button
            className="secondary-button"
            onClick={() => {
              setFile(null);
              setPreview(null);
              setResult(null);
              setGeoStatus("idle");
              setCoords(null);
            }}
          >
            Report another pothole
          </button>
        </div>
      )}
    </div>
  );
}
