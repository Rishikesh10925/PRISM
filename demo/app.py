"""PRISM demo dashboard — Streamlit, for presentation/demo purposes only.

Not the planned production frontend (blueprint Section 4/7.1 specifies React + Leaflet
+ FastAPI + PostgreSQL/PostGIS, none of which exist yet — see README.md). This is a
single-process, local-only interface wired directly to the real trained pipeline
(YOLOv8n-seg detection -> severity scoring -> prioritization) so the actual working
system can be demonstrated end-to-end without waiting on the full web-app build.

UI design: built for a non-technical viewer (see docs/phase5/01_demo_ui_notes.md).
Every number/label a normal user sees is plain language (Severity Score + a Very
Low..Critical level, Priority Score + a Less Important..Very Important level, and four
named factors instead of raw weights/percentages). All the underlying technical values
(raw cues, OSM weight numbers, detection confidence, the actual alpha/beta/gamma/delta
weights) are still computed exactly the same way as before and remain inspectable in
the collapsed "Admin / developer details" section — nothing about the detection,
severity, or priority calculation logic changed, only what's shown by default.

Run with: streamlit run demo/app.py
"""

from __future__ import annotations

import io
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("severity", "prioritization", "detection"):
    sys.path.insert(0, str(REPO_ROOT / "src" / sub))

from ultralytics import YOLO  # noqa: E402

from pipeline import compute_severity  # noqa: E402
from fusion import SEVERITY_CATEGORIES  # noqa: E402
from formula import PRIORITY_CATEGORIES, priority_category, priority_score  # noqa: E402
from priority_schema import PriorityInputs, PriorityWeights  # noqa: E402
from road_type import road_importance_label, road_type_weight  # noqa: E402
from traffic_recurrence import recurrence_factor, recurrence_level_label, traffic_level_label, traffic_proxy  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "yolov8n_seg_multisource.pt"
SAMPLE_MANIFEST = REPO_ROOT / "data" / "annotations" / "splits" / "test.txt"

# Fixed internally -- the blueprint's default weighting (severity matters most, then
# road type, then traffic, then recurrence). Not exposed to normal users; adjustable in
# the Admin section for anyone who wants to see how the ranking responds.
DEFAULT_WEIGHTS = PriorityWeights(alpha=0.4, beta=0.3, gamma=0.2, delta=0.1)

# Used only when browser geolocation isn't available/granted -- Hyderabad city center,
# matching the coordinates already used elsewhere in this project (see
# src/prioritization/ablation.py). This is a fallback for the demo only; a real
# deployment would require a real GPS reading before accepting a report.
DEFAULT_LAT, DEFAULT_LON = 17.4239, 78.4738

SEVERITY_COLORS = {
    "Very Low": "#2e7d32",
    "Low": "#66bb6a",
    "Medium": "#f9a825",
    "High": "#ef6c00",
    "Critical": "#c62828",
}
PRIORITY_COLORS = {
    "Less Important": "#2e7d32",
    "Moderate": "#f9a825",
    "Important": "#ef6c00",
    "Very Important": "#c62828",
}

RECOMMENDED_ACTION = {
    "Very Important": "Repair as soon as possible — high risk to vehicles and pedestrians.",
    "Important": "Schedule repair soon.",
    "Moderate": "Repair during routine road maintenance.",
    "Less Important": "Monitor — not urgent right now.",
}

st.set_page_config(page_title="PRISM — Pothole Severity & Priority Demo", page_icon="🛣️", layout="wide")


@st.cache_resource
def load_model(model_path: str):
    """model_path is passed explicitly (not read from the module-level MODEL_PATH
    inside the function) so st.cache_resource's cache key actually changes when the
    checkpoint changes -- with a zero-arg cached function, Streamlit can keep serving
    a stale cached model object after the code changes which checkpoint to load,
    silently running inference with the wrong model. Bit us for real: after switching
    MODEL_PATH to the multi-source checkpoint, this cache kept serving the old
    Pothole-600-only model, which explained a real "nothing detected" pattern on
    Kaggle-source sample images that the old model was never trained on."""
    return YOLO(model_path)


@st.cache_data(show_spinner=False)
def list_sample_images(n: int = 12) -> list[str]:
    if not SAMPLE_MANIFEST.exists():
        return []
    paths = [p for p in SAMPLE_MANIFEST.read_text(encoding="utf-8").splitlines() if p.strip()]
    return paths[:: max(1, len(paths) // n)][:n]


def resolve_location(fallback_lat: float, fallback_lon: float) -> tuple[float, float, str]:
    """Best-effort automatic GPS: asks the browser for the device's real location. If
    the browser/user doesn't grant it (or this is a non-browser/automated context), the
    page keeps working with the given fallback location instead of blocking or
    fabricating a fake "detected" coordinate."""
    params = st.query_params
    if "lat" in params and "lon" in params:
        try:
            return float(params["lat"]), float(params["lon"]), "Detected from your browser"
        except ValueError:
            pass

    if "geo_requested" not in st.session_state:
        st.session_state.geo_requested = True
        components.html(
            """
            <script>
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(
                    (pos) => {
                        const params = new URLSearchParams(window.parent.location.search);
                        params.set('lat', pos.coords.latitude);
                        params.set('lon', pos.coords.longitude);
                        window.parent.location.search = params.toString();
                    },
                    (err) => { /* permission denied or unavailable -- keep the fallback */ }
                );
            }
            </script>
            """,
            height=0,
        )
    return fallback_lat, fallback_lon, "Default location (GPS not available/granted)"


@st.cache_data(show_spinner="Analyzing image...")
def run_pipeline(image_bytes: bytes, use_midas: bool) -> dict:
    """Cached on the image content, so interacting with the rest of the page afterward
    doesn't re-run the slow detection/severity model."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    model = load_model(str(MODEL_PATH))
    results = model.predict(image_bgr, verbose=False)[0]

    potholes = []
    overlay = image_bgr.copy()
    if results.masks is not None:
        for i, mask_tensor in enumerate(results.masks.data):
            mask = cv2.resize(
                mask_tensor.cpu().numpy(), (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            score, category, cues = compute_severity(image_bgr, mask, use_midas=use_midas)
            confidence = float(results.boxes.conf[i]) if results.boxes is not None else None

            colored = overlay.copy()
            colored[mask > 0] = (0, 0, 255)
            overlay = cv2.addWeighted(overlay, 0.7, colored, 0.3, 0)
            contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

            potholes.append(
                {
                    "severity_score": score,
                    "severity_category": category,
                    "area_ratio": cues.area_ratio,
                    "depth": cues.depth,
                    "irregularity": cues.irregularity,
                    "depth_source": cues.depth_source,
                    "confidence": confidence,
                }
            )

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return {"overlay": overlay_rgb, "potholes": potholes}


st.title("🛣️ PRISM — Pothole Severity & Priority")
st.caption(
    "Upload a road photo (or try a sample) to see potholes detected automatically, scored for how bad "
    "they are, and ranked for how urgently they should be repaired."
)

if not MODEL_PATH.exists():
    st.error(f"Trained model not found at {MODEL_PATH}. Train it first with src/detection/train_yolo.py.")
    st.stop()

with st.sidebar:
    st.header("📷 Choose a photo")
    uploaded = st.file_uploader("Upload a road photo", type=["jpg", "jpeg", "png"])

    samples = list_sample_images()
    sample_choice = st.selectbox("...or try a sample photo", ["(none)"] + [Path(p).name for p in samples])

    st.divider()
    with st.expander("⚙️ Admin / developer details"):
        use_midas = st.checkbox("Use AI depth estimation (slower, more accurate)", value=True)
        st.caption("Priority weights (advanced — normally fixed):")
        alpha = st.slider("Pothole Severity weight", 0.0, 1.0, DEFAULT_WEIGHTS.alpha, 0.05)
        beta = st.slider("Road Importance weight", 0.0, 1.0, DEFAULT_WEIGHTS.beta, 0.05)
        gamma = st.slider("Traffic Level weight", 0.0, 1.0, DEFAULT_WEIGHTS.gamma, 0.05)
        delta = st.slider("Previous Reports weight", 0.0, 1.0, DEFAULT_WEIGHTS.delta, 0.05)
        st.caption("Manual location override (used only if GPS isn't available):")
        manual_lat = st.number_input("Latitude", value=DEFAULT_LAT, format="%.4f")
        manual_lon = st.number_input("Longitude", value=DEFAULT_LON, format="%.4f")
        manual_reports = st.slider("Simulate: previous reports at this spot", 1, 10, 1)

image_bytes = None
if uploaded is not None:
    image_bytes = uploaded.getvalue()
elif sample_choice != "(none)":
    match = next(p for p in samples if Path(p).name == sample_choice)
    image_bytes = Path(match).read_bytes()

if image_bytes is None:
    st.info("👈 Upload a photo or pick a sample from the sidebar to get started.")
    st.stop()

result = run_pipeline(image_bytes, use_midas)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Your photo")
    st.image(Image.open(io.BytesIO(image_bytes)), use_container_width=True)
with col2:
    st.subheader(f"Potholes found: {len(result['potholes'])}")
    st.image(result["overlay"], use_container_width=True)

if not result["potholes"]:
    st.warning("No potholes detected in this photo.")
    st.stop()

# location / time / traffic-context are resolved once per photo, shared across every
# detected pothole in it (they all come from the same place and moment)
lat, lon, location_source = resolve_location(manual_lat, manual_lon)
hour = datetime.now().hour
rt_weight = road_type_weight(lat, lon)
t_proxy = traffic_proxy(rt_weight, hour)
report_count = manual_reports  # no live citizen-report database yet -- see demo/README.md
rec_factor = recurrence_factor(report_count)
weights = PriorityWeights(alpha=alpha, beta=beta, gamma=gamma, delta=delta)

st.divider()
st.subheader("Results")
st.caption(f"📍 {location_source} · 🕒 detected at {datetime.now().strftime('%H:%M')}")

rows = []
for idx, p in enumerate(result["potholes"]):
    p_score = priority_score(
        PriorityInputs(
            severity_score=p["severity_score"],
            road_type_weight=rt_weight,
            traffic_proxy=t_proxy,
            recurrence_factor=rec_factor,
        ),
        weights,
    )
    rows.append({**p, "priority_score": p_score, "priority_category": priority_category(p_score), "id": f"Pothole {idx + 1}"})

rows.sort(key=lambda r: r["priority_score"], reverse=True)

for r in rows:
    with st.container(border=True):
        top1, top2 = st.columns(2)
        with top1:
            st.markdown(f"#### {r['id']}")
            st.metric("Severity Score", f"{r['severity_score']:.0f} / 100")
            st.markdown(
                f"<span style='background-color:{SEVERITY_COLORS[r['severity_category']]};color:white;"
                f"padding:5px 14px;border-radius:14px;font-weight:600;font-size:0.95rem'>{r['severity_category']} severity</span>",
                unsafe_allow_html=True,
            )
        with top2:
            st.metric("Repair Priority Score", f"{r['priority_score']:.0f} / 100")
            st.markdown(
                f"<span style='background-color:{PRIORITY_COLORS[r['priority_category']]};color:white;"
                f"padding:5px 14px;border-radius:14px;font-weight:600;font-size:0.95rem'>{r['priority_category']}</span>",
                unsafe_allow_html=True,
            )

        st.markdown(f"**Recommended action:** {RECOMMENDED_ACTION[r['priority_category']]}")

        st.markdown("###### Why this ranking:")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Pothole Severity", r["severity_category"])
        f2.metric("Road Importance", road_importance_label(rt_weight))
        f3.metric("Traffic Level", traffic_level_label(t_proxy))
        f4.metric("Previous Reports", recurrence_level_label(report_count))

        with st.expander("🔧 Admin / developer details"):
            st.caption("Raw values behind the scores above — for technical review only.")
            d1, d2 = st.columns(2)
            with d1:
                st.write("**Severity cues**")
                st.write(f"- area ratio: `{r['area_ratio']:.4f}`")
                st.write(f"- depth cue ({r['depth_source']}): `{r['depth']:.3f}`")
                st.write(f"- edge irregularity: `{r['irregularity']:.3f}`")
                if r["confidence"] is not None:
                    st.write(f"- detection confidence: `{r['confidence']:.3f}`")
            with d2:
                st.write("**Priority inputs**")
                st.write(f"- road type weight: `{rt_weight:.3f}`")
                st.write(f"- traffic proxy: `{t_proxy:.3f}`")
                st.write(f"- recurrence factor: `{rec_factor:.3f}`")
                st.write(f"- weights: α={weights.alpha:.2f} β={weights.beta:.2f} γ={weights.gamma:.2f} δ={weights.delta:.2f}")
                st.write(f"- coordinates used: `{lat:.4f}, {lon:.4f}` ({location_source})")

st.divider()
st.caption(
    f"Model: {MODEL_PATH.name} (YOLOv8n-seg) · Severity levels: {', '.join(SEVERITY_CATEGORIES)} · "
    f"Priority levels: {', '.join(PRIORITY_CATEGORIES)} · Run at {time.strftime('%H:%M:%S')}"
)
