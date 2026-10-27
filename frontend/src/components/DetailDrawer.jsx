import { useEffect, useRef, useState } from "react";
import { fetchPothole, fetchReportImage } from "../api";
import { recomputePriority } from "../priorityFormula";
import {
  fullDate,
  osmUrl,
  priorityTerms,
  relativeTime,
  ROAD_SOURCE_LABELS,
  TIME_SOURCE_LABELS,
} from "../scoring";
import MaskOverlay from "./MaskOverlay";
import CueBreakdown from "./CueBreakdown";
import { CategoryChip, ScoreBar } from "./ScoreBadge";
import StatusSelect from "./StatusSelect";
import Spinner from "./Spinner";
import { ClockIcon, ExternalLinkIcon, ImageIcon, InfoIcon, MapPinIcon, XIcon } from "./Icons";

/**
 * Right-side sheet (desktop) / bottom sheet (mobile) with everything about one
 * detection: photo + mask, severity cues, priority-term breakdown under the CURRENT
 * slider weights, provenance notes and the status control. Esc / X / backdrop close.
 */
export default function DetailDrawer({ row, weights, onClose, onStatusChange, updating, onUnauthorized }) {
  const [detail, setDetail] = useState(null);
  const [detailState, setDetailState] = useState("loading"); // loading | ok | missing | error
  const [imageUrl, setImageUrl] = useState(null);
  const [imageState, setImageState] = useState("loading"); // loading | ok | missing
  const closeRef = useRef(null);
  const detectionId = row.detection_id;
  const reportId = row.report_id;

  // Detail record
  useEffect(() => {
    let cancelled = false;
    setDetailState("loading");
    setDetail(null);
    fetchPothole(detectionId)
      .then((d) => {
        if (!cancelled) {
          setDetail(d);
          setDetailState("ok");
        }
      })
      .catch((err) => {
        if (cancelled) return;
        if (onUnauthorized(err)) return;
        setDetailState(err.response?.status === 404 ? "missing" : "error");
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detectionId]);

  // Photo (needs the bearer token, so fetched as a blob -> object URL)
  useEffect(() => {
    let cancelled = false;
    let url = null;
    setImageState("loading");
    setImageUrl(null);
    fetchReportImage(reportId)
      .then((blob) => {
        if (cancelled) return;
        url = URL.createObjectURL(blob);
        setImageUrl(url);
        setImageState("ok");
      })
      .catch((err) => {
        if (cancelled) return;
        if (onUnauthorized(err)) return;
        setImageState("missing");
      });
    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reportId]);

  // Esc to close, focus management, background scroll lock
  useEffect(() => {
    const previouslyFocused = document.activeElement;
    closeRef.current?.focus();
    const onKey = (e) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
      if (previouslyFocused && previouslyFocused.focus) previouslyFocused.focus();
    };
  }, [onClose]);

  const liveScore = recomputePriority(row, weights);
  const terms = priorityTerms(row, weights);
  const report = detail?.report;
  const sev = detail?.severity_score;

  return (
    <div className="drawer-root">
      <div className="drawer-backdrop" onClick={onClose} aria-hidden="true" />
      <aside className="drawer" role="dialog" aria-modal="true" aria-labelledby="drawer-title">
        <header className="drawer-head">
          <div className="drawer-head-text">
            <span className="label-caps">Rank #{row.rank} · Report {row.report_id}</span>
            <h2 id="drawer-title" className="drawer-title">Detection {row.detection_id}</h2>
          </div>
          <button ref={closeRef} type="button" className="btn btn-icon" onClick={onClose} aria-label="Close details">
            <XIcon size={18} />
          </button>
        </header>

        <div className="drawer-body">
          <div className="drawer-photo">
            {imageState === "ok" && imageUrl ? (
              <MaskOverlay
                src={imageUrl}
                alt={`Report ${row.report_id} photo with detection ${row.detection_id} outlined`}
                fallbackSize={report?.image_width ? { w: report.image_width, h: report.image_height } : null}
                detections={
                  detail
                    ? [{ id: detail.id, mask_polygon: detail.mask_polygon, category: row.severity_category }]
                    : []
                }
              />
            ) : (
              <div className="photo-placeholder">
                {imageState === "loading" ? (
                  <Spinner size="sm" label="Loading photo…" />
                ) : (
                  <>
                    <ImageIcon size={28} strokeWidth={1.5} />
                    <span>Photo not available</span>
                  </>
                )}
              </div>
            )}
          </div>

          <section className="drawer-section">
            <div className="kv-grid">
              <div className="kv">
                <span className="label-caps">Priority (current weights)</span>
                <div className="kv-score">
                  <ScoreBar score={liveScore} category={row.live_priority_category} kind="priority" />
                </div>
                <CategoryChip category={row.live_priority_category} kind="priority" size="sm" />
              </div>
              <div className="kv">
                <span className="label-caps">Severity</span>
                <div className="kv-score">
                  <ScoreBar score={row.severity_score} category={row.severity_category} />
                </div>
                <CategoryChip category={row.severity_category} size="sm" />
              </div>
            </div>
          </section>

          <section className="drawer-section">
            <h3 className="section-title">Repair status</h3>
            <div className="drawer-status">
              <StatusSelect
                value={row.status}
                disabled={updating}
                onChange={(s) => onStatusChange(row.detection_id, s)}
                label={`Status for detection ${row.detection_id}`}
              />
              {updating && <Spinner size="sm" label="Saving…" />}
            </div>
          </section>

          <section className="drawer-section">
            <h3 className="section-title">Why this priority</h3>
            <p className="muted small">
              Each factor's contribution under the current weights. Together they make the score of{" "}
              <strong className="num">{liveScore.toFixed(1)}</strong>.
            </p>
            <div className="terms-stack" aria-hidden="true">
              {terms.map((t) => (
                <span key={t.key} className={`terms-seg seg-${t.key}`} style={{ width: `${t.points}%` }} />
              ))}
            </div>
            <ul className="terms">
              {terms.map((t) => (
                <li key={t.key} className="term">
                  <span className={`term-swatch seg-${t.key}`} aria-hidden="true" />
                  <span className="term-label">{t.label}</span>
                  <span className="term-calc num">
                    {(t.share * 100).toFixed(0)}% × {t.display}
                  </span>
                  <span className="term-points num">+{t.points.toFixed(1)}</span>
                </li>
              ))}
            </ul>
          </section>

          <section className="drawer-section">
            <h3 className="section-title">Why this severity</h3>
            {detailState === "loading" && (
              <div className="skeleton-block">
                <div className="skeleton-line" />
                <div className="skeleton-line" />
                <div className="skeleton-line short" />
              </div>
            )}
            {detailState === "ok" && <CueBreakdown severity={sev} />}
            {(detailState === "missing" || detailState === "error") && (
              <p className="muted small">The cue breakdown isn't available for this detection right now.</p>
            )}
          </section>

          <section className="drawer-section">
            <h3 className="section-title">Location &amp; provenance</h3>
            <ul className="facts">
              <li>
                <MapPinIcon size={15} />
                <span className="num">
                  {row.latitude.toFixed(5)}, {row.longitude.toFixed(5)}
                </span>
                <a className="link-inline" href={osmUrl(row.latitude, row.longitude)} target="_blank" rel="noreferrer">
                  Open in OSM <ExternalLinkIcon size={13} />
                </a>
              </li>
              <li>
                <ClockIcon size={15} />
                <span>
                  Reported {relativeTime(row.submitted_at)} · <span className="num">{fullDate(row.submitted_at)}</span>
                </span>
              </li>
              {detail && (
                <li>
                  <InfoIcon size={15} />
                  <span>
                    Detector confidence <span className="num">{Math.round(detail.confidence * 100)}%</span>
                  </span>
                </li>
              )}
              {report?.road_type_source && (
                <li>
                  <InfoIcon size={15} />
                  <span>{ROAD_SOURCE_LABELS[report.road_type_source] || report.road_type_source}</span>
                </li>
              )}
              {report?.time_source && (
                <li>
                  <InfoIcon size={15} />
                  <span>{TIME_SOURCE_LABELS[report.time_source] || report.time_source}</span>
                </li>
              )}
            </ul>
          </section>
        </div>
      </aside>
    </div>
  );
}
