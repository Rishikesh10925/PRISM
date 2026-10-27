import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { downloadPdfReport, fetchPriorityList, updateStatus } from "../api";
import { priorityCategory, recomputePriority } from "../priorityFormula";
import { clearToken, getUsername } from "../auth";
import { CategoryChip, ScoreBar } from "../components/ScoreBadge";
import Spinner from "../components/Spinner";
import WorklistMap from "../components/WorklistMap";
import DetailDrawer from "../components/DetailDrawer";
import StatusSelect from "../components/StatusSelect";
import { fullDate, osmUrl, relativeTime } from "../scoring";
import {
  AlertCircleIcon,
  CheckCircleIcon,
  ChevronDownIcon,
  ExternalLinkIcon,
  FileDownIcon,
  FlagIcon,
  InboxIcon,
  LogOutIcon,
  RefreshIcon,
  SearchIcon,
  SlidersIcon,
  WrenchIcon,
  AlertTriangleIcon,
} from "../components/Icons";

const DEFAULT_WEIGHTS = { alpha: 0.4, beta: 0.3, gamma: 0.2, delta: 0.1 };

const WEIGHT_FIELDS = [
  { key: "alpha", label: "Pothole severity", hint: "Size, depth and edge damage" },
  { key: "beta", label: "Road importance", hint: "Arterial vs. residential" },
  { key: "gamma", label: "Traffic level", hint: "Expected traffic at report time" },
  { key: "delta", label: "Previous reports", hint: "Repeat reports nearby" },
];

const STATUS_FILTERS = [
  { value: "", label: "All" },
  { value: "reported", label: "Reported" },
  { value: "in_progress", label: "In progress" },
  { value: "repaired", label: "Repaired" },
];

export default function AdminDashboard() {
  const navigate = useNavigate();
  const [items, setItems] = useState([]);
  const [weights, setWeights] = useState(DEFAULT_WEIGHTS);
  const [statusFilter, setStatusFilter] = useState("");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [updatingId, setUpdatingId] = useState(null);
  const [downloadingPdf, setDownloadingPdf] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [drawerId, setDrawerId] = useState(null);
  const [panelOpen, setPanelOpen] = useState(false);

  const handleUnauthorized = useCallback(
    (err) => {
      if (err.response?.status === 401) {
        navigate("/admin/login", { replace: true });
        return true;
      }
      return false;
    },
    [navigate]
  );

  // The whole list is fetched once and filtered client-side, so the KPI row always
  // reflects every report regardless of the status filter.
  const reload = useCallback(
    async ({ silent = false } = {}) => {
      if (silent) setRefreshing(true);
      else setLoading(true);
      setError(null);
      try {
        const data = await fetchPriorityList();
        setItems(data);
        setLastUpdated(new Date());
      } catch (err) {
        if (!handleUnauthorized(err)) {
          setError("Could not load the worklist. Is the backend running?");
        }
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    [handleUnauthorized]
  );

  useEffect(() => {
    reload();
  }, [reload]);

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

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return ranked
      .filter((r) => !statusFilter || r.status === statusFilter)
      .filter((r) => {
        if (!q) return true;
        const hay = `${r.detection_id} #${r.detection_id} ${r.report_id} ${r.latitude.toFixed(5)} ${r.longitude.toFixed(5)} ${r.latitude.toFixed(5)}, ${r.longitude.toFixed(5)}`;
        return hay.includes(q.replace(/^#/, "")) || hay.includes(q);
      })
      .map((r, i) => ({ ...r, rank: i + 1 }));
  }, [ranked, statusFilter, query]);

  const kpis = useMemo(() => {
    const open = ranked.filter((r) => r.status !== "repaired").length;
    const inProgress = ranked.filter((r) => r.status === "in_progress").length;
    const repaired = ranked.filter((r) => r.status === "repaired").length;
    const high = ranked.filter(
      (r) =>
        r.status !== "repaired" &&
        (r.live_priority_category === "Important" || r.live_priority_category === "Very Important")
    ).length;
    return { open, inProgress, repaired, high };
  }, [ranked]);

  const drawerRow = drawerId != null ? visible.find((r) => r.detection_id === drawerId) || ranked.find((r) => r.detection_id === drawerId) : null;

  async function handleStatusChange(detectionId, newStatus) {
    setUpdatingId(detectionId);
    setError(null);
    try {
      await updateStatus(detectionId, newStatus);
      await reload({ silent: true });
      setNotice(`Detection ${detectionId} marked as ${newStatus.replace("_", " ")}.`);
    } catch (err) {
      if (!handleUnauthorized(err)) {
        setError("Could not update status. Is the backend running?");
      }
    } finally {
      setUpdatingId(null);
    }
  }

  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setNotice(null), 3500);
    return () => clearTimeout(t);
  }, [notice]);

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

  const openDrawer = useCallback((id) => {
    setSelectedId(id);
    setDrawerId(id);
  }, []);
  const closeDrawer = useCallback(() => setDrawerId(null), []);

  // Marker click -> select + scroll the row into view.
  const tableRef = useRef(null);
  const selectFromMap = useCallback((id) => {
    setSelectedId(id);
    const el = tableRef.current?.querySelector(`[data-row-id="${id}"]`);
    el?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, []);

  const weightTotal = weights.alpha + weights.beta + weights.gamma + weights.delta;
  const weightsChanged = WEIGHT_FIELDS.some((f) => weights[f.key] !== DEFAULT_WEIGHTS[f.key]);
  const username = getUsername() || "admin";

  return (
    <main className="page admin-page">
      <header className="admin-head">
        <div className="admin-head-text">
          <p className="eyebrow">Admin console</p>
          <h1 className="page-title">Repair worklist</h1>
          <p className="admin-updated">
            {lastUpdated ? (
              <>
                Updated <span className="num">{lastUpdated.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
              </>
            ) : (
              "Loading…"
            )}
            <button
              type="button"
              className="btn btn-link btn-sm"
              onClick={() => reload({ silent: true })}
              disabled={loading || refreshing}
            >
              <RefreshIcon size={14} className={refreshing ? "spin" : ""} /> Refresh
            </button>
          </p>
        </div>
        <div className="admin-head-actions">
          <button type="button" className="btn btn-secondary" onClick={handleDownloadPdf} disabled={downloadingPdf}>
            {downloadingPdf ? <Spinner size="sm" /> : <FileDownIcon size={16} />}
            <span>{downloadingPdf ? "Generating…" : "Export PDF"}</span>
          </button>
          <UserMenu username={username} onLogout={handleLogout} />
        </div>
      </header>

      {error && (
        <div className="alert alert-error alert-row" role="alert">
          <AlertCircleIcon size={18} />
          <span>{error}</span>
          <button type="button" className="btn btn-secondary btn-sm" onClick={() => reload()}>
            <RefreshIcon size={14} /> Retry
          </button>
        </div>
      )}

      <section className="kpis" aria-label="Summary">
        <Kpi label="Open reports" value={kpis.open} icon={<InboxIcon size={16} />} loading={loading} hint="Reported or in progress (not yet repaired)" />
        <Kpi label="In progress" value={kpis.inProgress} icon={<WrenchIcon size={16} />} loading={loading} />
        <Kpi label="Repaired" value={kpis.repaired} icon={<CheckCircleIcon size={16} />} loading={loading} />
        <Kpi
          label="High priority"
          value={kpis.high}
          icon={<FlagIcon size={16} />}
          loading={loading}
          emphasis={kpis.high > 0}
          hint="Important + Very important, not yet repaired, under current weights"
        />
      </section>

      <div className="admin-layout">
        <aside className={`panel${panelOpen ? " is-open" : ""}`} aria-label="Filters and weights">
          <button
            type="button"
            className="panel-toggle"
            aria-expanded={panelOpen}
            aria-controls="panel-body"
            onClick={() => setPanelOpen((v) => !v)}
          >
            <SlidersIcon size={16} />
            <span>Filters &amp; weights</span>
            {(statusFilter || query || weightsChanged) && <span className="panel-toggle-dot" aria-label="active" />}
            <ChevronDownIcon size={16} className="panel-toggle-chevron" />
          </button>

          <div id="panel-body" className="panel-body">
            <section className="panel-section">
              <h2 className="label-caps">Status</h2>
              <div className="segmented" role="radiogroup" aria-label="Filter by status">
                {STATUS_FILTERS.map((f) => (
                  <button
                    key={f.value || "all"}
                    type="button"
                    role="radio"
                    aria-checked={statusFilter === f.value}
                    className={`segmented-item${statusFilter === f.value ? " is-on" : ""}`}
                    onClick={() => setStatusFilter(f.value)}
                  >
                    {f.label}
                  </button>
                ))}
              </div>
            </section>

            <section className="panel-section">
              <label className="label-caps" htmlFor="worklist-search">Search</label>
              <div className="search">
                <SearchIcon size={16} />
                <input
                  id="worklist-search"
                  className="input input-sm"
                  type="search"
                  placeholder="Detection ID or coordinates"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
              </div>
            </section>

            <section className="panel-section">
              <div className="panel-section-head">
                <h2 className="label-caps">Priority weights</h2>
                <button
                  type="button"
                  className="btn btn-link btn-sm"
                  onClick={() => setWeights(DEFAULT_WEIGHTS)}
                  disabled={!weightsChanged}
                >
                  Reset
                </button>
              </div>
              <p className="muted small">Drag to re-rank the worklist instantly.</p>
              <div className="share-bar" aria-hidden="true">
                {WEIGHT_FIELDS.map((f) => (
                  <span
                    key={f.key}
                    className={`share-seg seg-${f.key}`}
                    style={{ width: `${weightTotal > 0 ? (100 * weights[f.key]) / weightTotal : 0}%` }}
                  />
                ))}
              </div>
              {WEIGHT_FIELDS.map((f) => (
                <WeightSlider
                  key={f.key}
                  id={`w-${f.key}`}
                  field={f}
                  value={weights[f.key]}
                  share={weightTotal > 0 ? weights[f.key] / weightTotal : 0}
                  onChange={(v) => setWeights((w) => ({ ...w, [f.key]: v }))}
                />
              ))}
              {weightTotal <= 0 && (
                <p className="alert alert-warn small">
                  <AlertTriangleIcon size={16} /> All weights are zero, so every priority is 0.
                </p>
              )}
            </section>
          </div>
        </aside>

        <div className="admin-main">
          <WorklistMap rows={visible} selectedId={selectedId} onSelect={selectFromMap} loading={loading} />

          <section className="card card-flush worklist" aria-labelledby="worklist-title">
            <header className="worklist-head">
              <h2 id="worklist-title" className="section-title">
                Worklist <span className="count num">{loading ? "" : visible.length}</span>
              </h2>
              <span className="muted small">Ranked by priority · click a row for details</span>
            </header>

            {loading ? (
              <div className="table-skeleton" aria-busy="true" aria-label="Loading worklist">
                {[0, 1, 2, 3, 4].map((i) => (
                  <div className="skeleton-row" key={i}>
                    <span className="skeleton-line tiny" />
                    <span className="skeleton-line" />
                    <span className="skeleton-line short" />
                    <span className="skeleton-line" />
                  </div>
                ))}
              </div>
            ) : visible.length === 0 ? (
              <div className="empty">
                <span className="empty-icon"><InboxIcon size={24} strokeWidth={1.5} /></span>
                {ranked.length === 0 ? (
                  <>
                    <p className="empty-title">No reports yet</p>
                    <p className="muted small">Reports submitted from the public page will appear here, ranked by priority.</p>
                  </>
                ) : (
                  <>
                    <p className="empty-title">Nothing matches these filters</p>
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={() => {
                        setStatusFilter("");
                        setQuery("");
                      }}
                    >
                      Clear filters
                    </button>
                  </>
                )}
              </div>
            ) : (
              <div className="table-scroll" ref={tableRef}>
                <table className="worklist-table">
                  <thead>
                    <tr>
                      <th scope="col" className="col-rank">#</th>
                      <th scope="col">Priority</th>
                      <th scope="col">Severity</th>
                      <th scope="col">Location</th>
                      <th scope="col">Reported</th>
                      <th scope="col">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visible.map((item) => {
                      const isSel = item.detection_id === selectedId;
                      return (
                        <tr
                          key={item.detection_id}
                          data-row-id={item.detection_id}
                          className={`${isSel ? "is-selected" : ""}${updatingId === item.detection_id ? " is-updating" : ""}`}
                          tabIndex={0}
                          aria-label={`Rank ${item.rank}, detection ${item.detection_id}, open details`}
                          onClick={() => openDrawer(item.detection_id)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" || e.key === " ") {
                              e.preventDefault();
                              openDrawer(item.detection_id);
                            }
                          }}
                          onMouseEnter={() => setSelectedId(item.detection_id)}
                        >
                          <td className="col-rank num" data-label="Rank">{item.rank}</td>
                          <td data-label="Priority">
                            <div className="cell-priority">
                              <ScoreBar
                                score={item.live_priority_score}
                                category={item.live_priority_category}
                                kind="priority"
                              />
                              <span className="cell-cat">{item.live_priority_category}</span>
                            </div>
                          </td>
                          <td data-label="Severity">
                            <span className="cell-sev">
                              <CategoryChip category={item.severity_category} size="sm" />
                              <span className="muted num">{Math.round(item.severity_score)}</span>
                            </span>
                          </td>
                          <td data-label="Location">
                            <span className="cell-loc">
                              <span className="num">
                                {item.latitude.toFixed(4)}, {item.longitude.toFixed(4)}
                              </span>
                              <a
                                href={osmUrl(item.latitude, item.longitude)}
                                target="_blank"
                                rel="noreferrer"
                                className="icon-link"
                                aria-label="Open in OpenStreetMap"
                                title="Open in OpenStreetMap"
                                onClick={(e) => e.stopPropagation()}
                              >
                                <ExternalLinkIcon size={14} />
                              </a>
                            </span>
                          </td>
                          <td data-label="Reported">
                            <time dateTime={item.submitted_at} title={fullDate(item.submitted_at)}>
                              {relativeTime(item.submitted_at)}
                            </time>
                          </td>
                          <td data-label="Status" className="cell-status">
                            <StatusSelect
                              value={item.status}
                              disabled={updatingId === item.detection_id}
                              onChange={(s) => handleStatusChange(item.detection_id, s)}
                              label={`Status for detection ${item.detection_id}`}
                            />
                            {updatingId === item.detection_id && <Spinner size="sm" />}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </div>
      </div>

      <div className="toast-region" aria-live="polite">
        {notice && (
          <div className="toast">
            <CheckCircleIcon size={16} /> {notice}
          </div>
        )}
      </div>

      {drawerRow && (
        <DetailDrawer
          row={drawerRow}
          weights={weights}
          onClose={closeDrawer}
          onStatusChange={handleStatusChange}
          updating={updatingId === drawerRow.detection_id}
          onUnauthorized={handleUnauthorized}
        />
      )}
    </main>
  );
}

function Kpi({ label, value, icon, loading, emphasis, hint }) {
  return (
    <div className={`kpi${emphasis ? " is-emphasis" : ""}`} title={hint}>
      <span className="kpi-label">
        {icon}
        {label}
      </span>
      {loading ? <span className="skeleton-line kpi-skel" /> : <span className="kpi-value num">{value}</span>}
    </div>
  );
}

function WeightSlider({ id, field, value, share, onChange }) {
  return (
    <div className="weight">
      <div className="weight-head">
        <label htmlFor={id} className="weight-label">
          <span className={`weight-swatch seg-${field.key}`} aria-hidden="true" />
          {field.label}
        </label>
        <span className="weight-value num">{value.toFixed(2)}</span>
        <span className="weight-share num">{Math.round(share * 100)}%</span>
      </div>
      <input
        id={id}
        className="range"
        type="range"
        min="0"
        max="1"
        step="0.05"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ "--fill": `${value * 100}%` }}
        aria-describedby={`${id}-hint`}
      />
      <span id={`${id}-hint`} className="weight-hint">{field.hint}</span>
    </div>
  );
}

function UserMenu({ username, onLogout }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e) => {
      if (!ref.current?.contains(e.target)) setOpen(false);
    };
    const onKey = (e) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="user-menu" ref={ref}>
      <button
        type="button"
        className="user-menu-btn"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <span className="avatar" aria-hidden="true">{username.charAt(0).toUpperCase()}</span>
        <span className="user-name">{username}</span>
        <ChevronDownIcon size={14} />
      </button>
      {open && (
        <div className="menu" role="menu">
          <div className="menu-meta">
            Signed in as <strong>{username}</strong>
          </div>
          <button type="button" role="menuitem" className="menu-item" onClick={onLogout}>
            <LogOutIcon size={16} /> Sign out
          </button>
        </div>
      )}
    </div>
  );
}
