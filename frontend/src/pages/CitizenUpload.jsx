import { useEffect, useRef, useState } from "react";
import { submitReport } from "../api";
import Spinner from "../components/Spinner";
import MaskOverlay from "../components/MaskOverlay";
import CueBreakdown from "../components/CueBreakdown";
import { CategoryChip, ScoreBar } from "../components/ScoreBadge";
import {
  AlertCircleIcon,
  AlertTriangleIcon,
  ArrowRightIcon,
  CameraIcon,
  CheckCircleIcon,
  ChevronDownIcon,
  InfoIcon,
  ListOrderedIcon,
  LocateIcon,
  MapPinIcon,
  PlusIcon,
  RefreshIcon,
  ScanIcon,
} from "../components/Icons";

const MAX_MB = 15;

function friendlyError(err) {
  const status = err.response?.status;
  const detail = err.response?.data?.detail;
  if (!err.response) return "We couldn't reach the PRISM server. Check your connection and try again.";
  if (status === 413) return `That photo is larger than ${MAX_MB} MB. Please choose a smaller photo or take a new one.`;
  if (status === 415) return "That file type isn't supported. Please use a JPG, PNG or WebP photo.";
  if (status === 422) {
    if (typeof detail === "string" && /lat|lon/i.test(detail))
      return "Your location looks invalid. Please retry capturing your location.";
    return "We couldn't read that image. Please try a different photo.";
  }
  if (status === 500) return "We couldn't analyse that photo right now, so the report wasn't saved. Please try again.";
  return typeof detail === "string" ? detail : "Something went wrong submitting your report. Please try again.";
}

export default function CitizenUpload() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [coords, setCoords] = useState(null);
  const [geoStatus, setGeoStatus] = useState("idle"); // idle | locating | ok | denied
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);
  const resultRef = useRef(null);

  // Revoke the preview object URL when it's replaced or the page unmounts.
  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  useEffect(() => {
    if (result) resultRef.current?.focus();
  }, [result]);

  function pickFile(f) {
    if (!f) return;
    if (!f.type?.startsWith("image/")) {
      setError("That file isn't an image. Please choose a JPG, PNG or WebP photo.");
      return;
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      setError(`That photo is larger than ${MAX_MB} MB. Please choose a smaller photo.`);
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
    if (geoStatus !== "ok") captureLocation();
  }

  function handleFileChange(e) {
    pickFile(e.target.files?.[0]);
    e.target.value = "";
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragActive(false);
    if (submitting) return;
    pickFile(e.dataTransfer.files?.[0]);
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
      setError(friendlyError(err));
    } finally {
      setSubmitting(false);
    }
  }

  function reset() {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  return (
    <main className="page citizen-page">
      {!result && (
        <>
          <section className="citizen-hero" aria-labelledby="citizen-title">
            <p className="eyebrow">Public reporting</p>
            <h1 id="citizen-title" className="page-title">Report a pothole</h1>
            <p className="lede">
              Send a photo from the spot. PRISM outlines the pothole, estimates how severe it is, and adds it to the
              city's ranked repair list.
            </p>
            <ol className="steps">
              <li className="step">
                <span className="step-icon"><CameraIcon size={18} /></span>
                <span className="step-text"><strong>Photograph</strong><span>Snap it from road level</span></span>
              </li>
              <li className="step">
                <span className="step-icon"><ScanIcon size={18} /></span>
                <span className="step-text"><strong>We detect &amp; assess</strong><span>Outline, size, depth, edges</span></span>
              </li>
              <li className="step">
                <span className="step-icon"><ListOrderedIcon size={18} /></span>
                <span className="step-text"><strong>Crews prioritise</strong><span>Ranked with road &amp; traffic</span></span>
              </li>
            </ol>
          </section>

          <section className="card upload-card" aria-label="Upload a photo">
            <input
              ref={inputRef}
              id="photo-input"
              type="file"
              accept="image/jpeg,image/png,image/webp"
              capture="environment"
              onChange={handleFileChange}
              disabled={submitting}
              className="visually-hidden"
            />

            {preview ? (
              <div className="preview">
                <img src={preview} alt="Selected pothole photo" className="preview-img" />
                <div className="preview-bar">
                  <span className="preview-name" title={file?.name}>{file?.name}</span>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => inputRef.current?.click()}
                    disabled={submitting}
                  >
                    <RefreshIcon size={15} /> Change photo
                  </button>
                </div>
              </div>
            ) : (
              <label
                htmlFor="photo-input"
                className={`dropzone${dragActive ? " is-drag" : ""}${submitting ? " is-disabled" : ""}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  if (!submitting) setDragActive(true);
                }}
                onDragLeave={() => setDragActive(false)}
                onDrop={handleDrop}
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    inputRef.current?.click();
                  }
                }}
              >
                <span className="dropzone-icon"><CameraIcon size={28} strokeWidth={1.5} /></span>
                <span className="dropzone-primary">Take or choose a photo</span>
                <span className="dropzone-secondary">or drag it here</span>
                <span className="dropzone-meta">JPG, PNG or WebP · up to {MAX_MB} MB</span>
              </label>
            )}

            <LocationStatus status={geoStatus} coords={coords} onRetry={captureLocation} />

            <div aria-live="polite">
              {error && (
                <p className="alert alert-error" role="alert">
                  <AlertCircleIcon size={18} />
                  <span>{error}</span>
                </p>
              )}
            </div>

            <button
              type="button"
              className="btn btn-primary btn-block btn-lg"
              disabled={!file || !coords || submitting}
              onClick={handleSubmit}
            >
              {submitting ? (
                <>
                  <Spinner size="sm" /> Analysing photo…
                </>
              ) : (
                <>
                  Submit report <ArrowRightIcon size={16} />
                </>
              )}
            </button>
            {submitting && (
              <p className="muted small center">This usually takes a few seconds — we're outlining and scoring the pothole.</p>
            )}
          </section>
        </>
      )}

      {result && (
        <ResultView result={result} preview={preview} onReset={reset} headingRef={resultRef} />
      )}
    </main>
  );
}

function LocationStatus({ status, coords, onRetry }) {
  if (status === "locating") {
    return (
      <div className="loc loc-pending" role="status">
        <Spinner size="sm" />
        <span>Getting your location…</span>
      </div>
    );
  }
  if (status === "ok" && coords) {
    return (
      <div className="loc loc-ok" role="status">
        <MapPinIcon size={16} />
        <span>
          Location captured <span className="num loc-coords">{coords.lat.toFixed(5)}, {coords.lon.toFixed(5)}</span>
        </span>
      </div>
    );
  }
  if (status === "denied") {
    return (
      <div className="loc loc-warn" role="alert">
        <AlertTriangleIcon size={16} />
        <span>Location unavailable. Allow location access so crews can find it.</span>
        <button type="button" className="btn btn-link btn-sm" onClick={onRetry}>
          <LocateIcon size={15} /> Retry
        </button>
      </div>
    );
  }
  return (
    <div className="loc loc-idle">
      <LocateIcon size={16} />
      <span>Your location is attached automatically when you pick a photo.</span>
    </div>
  );
}

function ResultView({ result, preview, onReset, headingRef }) {
  const detections = result.detections || [];
  const [activeId, setActiveId] = useState(null);
  const count = detections.length;

  return (
    <section className="result" aria-labelledby="result-title">
      <header className="result-head">
        <span className={`result-badge${count === 0 ? " is-neutral" : ""}`}>
          {count === 0 ? <InfoIcon size={20} /> : <CheckCircleIcon size={20} />}
        </span>
        <div>
          <h1 id="result-title" className="page-title" tabIndex={-1} ref={headingRef}>
            {count === 0 ? "No pothole found" : "Report received"}
          </h1>
          <p className="lede">
            {count === 0
              ? "We couldn't spot a pothole in this photo, so nothing was added to the worklist."
              : `We found ${count} pothole${count === 1 ? "" : "s"} and added ${count === 1 ? "it" : "them"} to the repair worklist. Report #${result.id}.`}
          </p>
        </div>
      </header>

      <div className={`result-grid${count === 0 ? " is-empty" : ""}`}>
        <div className="result-photo card card-flush">
          {count > 0 ? (
            <MaskOverlay
              src={preview}
              alt="Your photo with detected potholes outlined"
              activeId={activeId}
              onSelect={(id) => setActiveId((cur) => (cur === id ? null : id))}
              detections={detections.map((d, i) => ({
                id: d.id,
                mask_polygon: d.mask_polygon,
                label: i + 1,
                category: d.severity_score?.category,
              }))}
            />
          ) : (
            <img src={preview} alt="Your photo" className="preview-img" />
          )}
          <p className="photo-caption">
            <MapPinIcon size={14} />
            <span className="num">{result.latitude.toFixed(5)}, {result.longitude.toFixed(5)}</span>
          </p>
        </div>

        <div className="result-side">
          {count === 0 ? (
            <div className="card empty-tips">
              <h2 className="section-title">Try another angle</h2>
              <ul className="tips">
                <li>Stand 1–2 metres back so the whole pothole is in frame.</li>
                <li>Shoot in daylight and avoid strong shadows across the hole.</li>
                <li>Point the camera down at the road, not at the horizon.</li>
                <li>Make sure the pothole is in focus and not covered by water.</li>
              </ul>
            </div>
          ) : (
            detections.map((d, i) => (
              <DetectionCard
                key={d.id}
                index={i + 1}
                detection={d}
                active={activeId === d.id}
                onHover={(on) => setActiveId(on ? d.id : null)}
              />
            ))
          )}

          {count > 0 && (
            <p className="note">
              <InfoIcon size={16} />
              <span>
                Severity is an automated estimate from a single photo; crews verify on site before repairs are
                scheduled.
              </span>
            </p>
          )}

          <button type="button" className="btn btn-secondary btn-block" onClick={onReset}>
            <PlusIcon size={16} /> Report another pothole
          </button>
        </div>
      </div>
    </section>
  );
}

function DetectionCard({ index, detection, active, onHover }) {
  const sev = detection.severity_score;
  const pri = detection.priority_score;
  const [open, setOpen] = useState(index === 1);
  const panelId = `why-${detection.id}`;

  return (
    <article
      className={`card det-card${active ? " is-active" : ""}`}
      onMouseEnter={() => onHover(true)}
      onMouseLeave={() => onHover(false)}
      aria-label={`Pothole ${index}`}
    >
      <header className="det-head">
        <span className="det-index">{index}</span>
        <h2 className="det-title">Pothole {index}</h2>
        <span className="det-conf num" title="How confident the detector is that this is a pothole">
          {Math.round(detection.confidence * 100)}% confidence
        </span>
      </header>

      <div className="det-scores">
        <div className="det-score">
          <span className="label-caps">Severity</span>
          <div className="det-big">
            <span className="det-big-num num">{sev ? Math.round(sev.score) : "–"}</span>
            <span className="det-big-of">/100</span>
          </div>
          {sev && <CategoryChip category={sev.category} kind="severity" />}
          {sev && <ScoreBar score={sev.score} category={sev.category} showValue={false} />}
        </div>
        <div className="det-score">
          <span className="label-caps">Repair priority</span>
          <div className="det-big">
            <span className="det-big-num num">{pri ? Math.round(pri.score) : "–"}</span>
            <span className="det-big-of">/100</span>
          </div>
          {pri && <CategoryChip category={pri.category} kind="priority" />}
          {pri && <ScoreBar score={pri.score} category={pri.category} kind="priority" showValue={false} />}
        </div>
      </div>

      <button
        type="button"
        className="disclosure"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((v) => !v)}
      >
        <span>Why this score</span>
        <ChevronDownIcon size={16} className="disclosure-chevron" />
      </button>
      {open && (
        <div id={panelId} className="disclosure-panel">
          <CueBreakdown severity={sev} />
          <p className="muted small">
            Priority also weighs the road type and expected traffic at this spot, and whether it has been reported
            before.
          </p>
        </div>
      )}
    </article>
  );
}
