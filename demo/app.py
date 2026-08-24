"""PRISM demo dashboard — Streamlit, for presentation/demo purposes only.

Not the planned production frontend (blueprint Section 4/7.1 specifies React + Leaflet
+ FastAPI + PostgreSQL/PostGIS, none of which exist yet — see README.md). This is a
single-process, local-only interface wired directly to the real trained pipeline
(YOLOv8n-seg detection -> severity scoring -> prioritization) so the actual working
system can be demonstrated end-to-end without waiting on the full web-app build.

Run with: streamlit run demo/app.py
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("severity", "prioritization", "detection"):
    sys.path.insert(0, str(REPO_ROOT / "src" / sub))

from ultralytics import YOLO  # noqa: E402

from pipeline import compute_severity  # noqa: E402
from formula import priority_score  # noqa: E402
from priority_schema import PriorityInputs, PriorityWeights  # noqa: E402
from road_type import road_type_weight  # noqa: E402
from traffic_recurrence import recurrence_factor, traffic_proxy  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "yolov8n_seg_augmented.pt"
SAMPLE_MANIFEST = REPO_ROOT / "data" / "annotations" / "splits" / "test.txt"

st.set_page_config(page_title="PRISM — Pothole Severity & Priority Demo", page_icon="🛣️", layout="wide")


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


@st.cache_data(show_spinner=False)
def list_sample_images(n: int = 12) -> list[str]:
    if not SAMPLE_MANIFEST.exists():
        return []
    paths = [p for p in SAMPLE_MANIFEST.read_text(encoding="utf-8").splitlines() if p.strip()]
    return paths[:: max(1, len(paths) // n)][:n]


@st.cache_data(show_spinner="Running detection + severity pipeline...")
def run_pipeline(image_bytes: bytes, use_midas: bool) -> dict:
    """Cached on the image content + midas toggle, so moving priority-weight sliders
    afterward doesn't re-run detection/MiDaS — only the fast formula recomputes."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    model = load_model()
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
                    "category": category,
                    "area_ratio": cues.area_ratio,
                    "depth": cues.depth,
                    "irregularity": cues.irregularity,
                    "depth_source": cues.depth_source,
                    "confidence": confidence,
                }
            )

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return {"overlay": overlay_rgb, "potholes": potholes}


def severity_color(category: str) -> str:
    return {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#ef6c00", "Critical": "#c62828"}.get(category, "#666")


st.title("🛣️ PRISM — Pothole Severity & Priority Demo")
st.caption(
    "Live demo of the real trained pipeline: YOLOv8n-seg detection → 3-cue severity scoring "
    "(area / MiDaS depth / irregularity) → multi-criteria priority ranking. "
    "Local demo interface only — not the planned production web app (see demo/README.md)."
)

if not MODEL_PATH.exists():
    st.error(f"Trained model not found at {MODEL_PATH}. Train it first with src/detection/train_yolo.py.")
    st.stop()

with st.sidebar:
    st.header("Input image")
    uploaded = st.file_uploader("Upload a road photo", type=["jpg", "jpeg", "png"])

    samples = list_sample_images()
    sample_choice = st.selectbox("...or pick a real test-set sample", ["(none)"] + [Path(p).name for p in samples])

    use_midas = st.checkbox("Use MiDaS depth proxy (slower, more accurate)", value=True)

    st.divider()
    st.header("Priority weights")
    st.caption("Live-adjustable — re-ranks instantly without re-running detection.")
    alpha = st.slider("α severity", 0.0, 1.0, 0.4, 0.05)
    beta = st.slider("β road type", 0.0, 1.0, 0.3, 0.05)
    gamma = st.slider("γ traffic", 0.0, 1.0, 0.2, 0.05)
    delta = st.slider("δ recurrence", 0.0, 1.0, 0.1, 0.05)

    st.divider()
    st.header("Context (optional)")
    st.caption("Feeds road-type weight (live OSM lookup) and traffic proxy.")
    lat = st.number_input("Latitude", value=17.4239, format="%.4f")
    lon = st.number_input("Longitude", value=78.4738, format="%.4f")
    hour = st.slider("Report hour", 0, 23, 8)
    report_count = st.slider("Recurrence: reports at this location", 1, 10, 1)

image_bytes = None
if uploaded is not None:
    image_bytes = uploaded.getvalue()
elif sample_choice != "(none)":
    match = next(p for p in samples if Path(p).name == sample_choice)
    image_bytes = Path(match).read_bytes()

if image_bytes is None:
    st.info("Upload an image or pick a sample from the sidebar to run the pipeline.")
    st.stop()

result = run_pipeline(image_bytes, use_midas)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Input")
    st.image(Image.open(io.BytesIO(image_bytes)), use_container_width=True)
with col2:
    st.subheader(f"Detected potholes: {len(result['potholes'])}")
    st.image(result["overlay"], use_container_width=True)

if not result["potholes"]:
    st.warning("No potholes detected in this image.")
    st.stop()

st.divider()
st.subheader("Severity & Priority")

weights = PriorityWeights(alpha=alpha, beta=beta, gamma=gamma, delta=delta)
rt_weight = road_type_weight(lat, lon)
t_proxy = traffic_proxy(rt_weight, hour)
rec_factor = recurrence_factor(report_count)

st.caption(
    f"Context resolved from OSM: road-type weight = **{rt_weight:.2f}**, "
    f"traffic proxy = **{t_proxy:.2f}**, recurrence factor = **{rec_factor:.2f}**"
)

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
    rows.append({**p, "priority_score": p_score, "id": f"pothole_{idx + 1}"})

rows.sort(key=lambda r: r["priority_score"], reverse=True)

for r in rows:
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            st.markdown(f"### {r['id']}")
            st.markdown(
                f"<span style='background-color:{severity_color(r['category'])};color:white;"
                f"padding:4px 10px;border-radius:12px;font-weight:600'>{r['category']}</span>",
                unsafe_allow_html=True,
            )
        with c2:
            st.metric("Severity Score S", f"{r['severity_score']:.1f} / 100")
            st.progress(min(r["area_ratio"], 1.0), text=f"area ratio: {r['area_ratio']:.3f}")
            st.progress(
                min(max(r["depth"] / 30.0, 0.0), 1.0), text=f"depth ({r['depth_source']}): {r['depth']:.2f}"
            )
            st.progress(
                min(max((r["irregularity"] - 1) / 4, 0.0), 1.0), text=f"irregularity: {r['irregularity']:.2f}"
            )
        with c3:
            st.metric("Priority Score P", f"{r['priority_score']:.1f} / 100")
            if r["confidence"] is not None:
                st.caption(f"detection confidence: {r['confidence']:.2f}")

st.divider()
st.caption(
    f"Model: {MODEL_PATH.name} (YOLOv8n-seg, offline-augmented, test set box mAP@0.5=0.891) · "
    f"Pipeline run at {time.strftime('%H:%M:%S')}"
)
