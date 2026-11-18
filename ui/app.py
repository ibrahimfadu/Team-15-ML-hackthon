from pathlib import Path
import math
import requests
import pandas as pd
import streamlit as st
import pydeck as pdk

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "models" / "model_training_deliverables" / "model_training_results"
EVAL_ROOT = ROOT / "data" / "evaluator_test"
API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="PathSnap | Team 15", page_icon="🗺️", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.1rem;max-width:1500px}
.hero{background:linear-gradient(110deg,#10243d,#075b68);padding:22px 28px;border-radius:16px;margin-bottom:18px}
.hero h1{color:white;margin:0;font-size:2.25rem}.hero p{color:#d5edf2;margin:6px 0 0}
[data-testid="stMetric"]{background:rgba(100,120,150,.12);padding:12px;border-radius:12px}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>PathSnap</h1><p>GPS Trajectory Map Matching · Team 15 · Section 5-D</p></div>', unsafe_allow_html=True)


def csv_file(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def load_trip(tid):
    folder = EVAL_ROOT / tid
    track = pd.read_csv(folder / f"{tid}.track", sep=r"\s+", header=None,
                        names=["longitude", "latitude", "timestamp"])
    nodes = pd.read_csv(folder / f"{tid}.nodes", sep=r"\s+", header=None,
                        names=["longitude", "latitude"])
    arcs = pd.read_csv(folder / f"{tid}.arcs", sep=r"\s+", header=None,
                       names=["from_node_id", "to_node_id"])
    return track.dropna(subset=["longitude", "latitude"]), nodes, arcs


def model_table():
    df = csv_file("student_test_model_comparison.csv")
    if df is None:
        df = csv_file("report_model_comparison_table.csv")
    return df


def run_prediction(tid):
    response = requests.post(f"{API_BASE}/predict/{tid}", timeout=300)
    response.raise_for_status()
    return response.json()


with st.sidebar:
    st.title("PathSnap")
    page = st.radio("Dashboard", ["Live map", "Model performance", "Project overview"])
    trip_ids = sorted(p.name for p in EVAL_ROOT.iterdir() if p.is_dir()) if EVAL_ROOT.exists() else []
    trip = st.selectbox("Evaluator trip", trip_ids,
                        index=trip_ids.index("00000015") if "00000015" in trip_ids else 0) if trip_ids else None
    st.caption("Map source: evaluator .track, .nodes and .arcs files. Ground-truth .route is never read by the UI.")

if page == "Live map":
    st.subheader(f"GPS trajectory and road network · {trip or 'No trip'}")
    if not trip:
        st.error("No evaluator trips found in data/evaluator_test.")
    else:
        # Keep prediction results separated by trip, so changing the selection never shows stale metrics.
        prediction_cache = st.session_state.setdefault("prediction_cache", {})
        run_col, status_col = st.columns([1, 3])
        with run_col:
            run_clicked = st.button("Run prediction & score trip", type="primary", use_container_width=True)
        with status_col:
            st.caption("Runs the tuned Random Forest + HMM/Viterbi pipeline, then scores the result against this trip's reference route on the backend.")

        if run_clicked:
            try:
                with st.spinner(f"Running map matching and scoring for trip {trip}..."):
                    prediction_cache[trip] = run_prediction(trip)
                st.success(f"Prediction and per-trip scoring completed for {trip}.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the FastAPI backend. Keep the backend running at http://127.0.0.1:8000.")
            except requests.exceptions.Timeout:
                st.error("Prediction timed out. Check the backend terminal; it may still be processing this trip.")
            except requests.exceptions.HTTPError as e:
                detail = ""
                try:
                    detail = e.response.text
                except Exception:
                    pass
                st.error(f"Backend returned an error: {e}. {detail}")
            except Exception as e:
                st.exception(e)

        result = prediction_cache.get(trip)

        try:
            track, nodes, arcs = load_trip(trip)
            gps = track[["longitude", "latitude"]].astype(float)
            lon_min, lon_max = float(gps.longitude.min()), float(gps.longitude.max())
            lat_min, lat_max = float(gps.latitude.min()), float(gps.latitude.max())
            center_lon, center_lat = (lon_min + lon_max) / 2, (lat_min + lat_max) / 2
            span = max(lon_max - lon_min, lat_max - lat_min, 0.0005)
            zoom = max(9, min(17, int(math.log2(360 / (span * 1.5)))))
            pad_lon = max((lon_max-lon_min)*0.25, 0.001)
            pad_lat = max((lat_max-lat_min)*0.25, 0.001)
            valid = arcs[
                arcs["from_node_id"].between(0, len(nodes)-1) &
                arcs["to_node_id"].between(0, len(nodes)-1)
            ].copy()
            u = valid.from_node_id.astype(int).to_numpy()
            v = valid.to_node_id.astype(int).to_numpy()
            nlons = nodes.longitude.to_numpy(dtype=float)
            nlats = nodes.latitude.to_numpy(dtype=float)
            mask = (
                (nlons[u] >= lon_min-pad_lon) & (nlons[u] <= lon_max+pad_lon) &
                (nlats[u] >= lat_min-pad_lat) & (nlats[u] <= lat_max+pad_lat) &
                (nlons[v] >= lon_min-pad_lon) & (nlons[v] <= lon_max+pad_lon) &
                (nlats[v] >= lat_min-pad_lat) & (nlats[v] <= lat_max+pad_lat)
            )
            local = valid.loc[mask].copy()
            if len(local) > 18000:
                local = local.iloc[::max(1, len(local)//18000)]
            road_paths = []
            for row in local.itertuples(index=False):
                a, b = int(row.from_node_id), int(row.to_node_id)
                road_paths.append({"path": [[float(nlons[a]), float(nlats[a])],
                                             [float(nlons[b]), float(nlats[b])]]})
            gps_path = [{"path": gps[["longitude", "latitude"]].values.tolist()}]
            layers = []
            if road_paths:
                layers.append(pdk.Layer("PathLayer", data=road_paths, get_path="path",
                    get_color=[125, 140, 160, 115], width_min_pixels=1, get_width=2,
                    pickable=False))
            layers.append(pdk.Layer("PathLayer", data=gps_path, get_path="path",
                get_color=[30, 120, 255, 255], width_min_pixels=4, get_width=5,
                pickable=False))

            # API route_coordinates are [latitude, longitude]; PyDeck expects [longitude, latitude].
            predicted_coords = result.get("route_coordinates", []) if result else []
            if predicted_coords:
                predicted_path = [{
                    "path": [[float(point[1]), float(point[0])] for point in predicted_coords
                             if isinstance(point, (list, tuple)) and len(point) >= 2]
                }]
                if predicted_path[0]["path"]:
                    layers.append(pdk.Layer("PathLayer", data=predicted_path, get_path="path",
                        get_color=[20, 175, 105, 255], width_min_pixels=5, get_width=6,
                        pickable=False))

            first, last = gps.iloc[0], gps.iloc[-1]
            markers = pd.DataFrame([
                {"longitude": first.longitude, "latitude": first.latitude, "name": "START"},
                {"longitude": last.longitude, "latitude": last.latitude, "name": "END"},
            ])
            layers.append(pdk.Layer("ScatterplotLayer", data=markers,
                get_position="[longitude, latitude]",
                get_fill_color=[[20,190,100,230],[230,65,65,230]],
                get_radius=35, radius_min_pixels=6, pickable=True))
            view = pdk.ViewState(longitude=center_lon, latitude=center_lat, zoom=zoom,
                                 pitch=0, bearing=0)
            deck = pdk.Deck(layers=layers, initial_view_state=view,
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                tooltip={"text": "{name}"})
            st.pydeck_chart(deck, height=650, width="stretch", key=f"pathsnap_map_{trip}")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("GPS observations", f"{len(track):,}")
            c2.metric("Road nodes", f"{len(nodes):,}")
            c3.metric("Road arcs", f"{len(arcs):,}")
            c4.metric("Road segments drawn", f"{len(road_paths):,}")
            st.caption("Blue: observed GPS trajectory · Grey: nearby road network · Green: predicted matched route · Green start marker · Red end marker")

            st.markdown("### Trip accuracy")
            if result:
                evaluation = result.get("per_trip_evaluation") or {}
                if evaluation:
                    st.caption(f"Metrics calculated for evaluator trip {trip}. Reference route is read and compared server-side only; its arc IDs are not sent to the UI.")
                    def metric_value(key):
                        value = evaluation.get(key)
                        return float(value) if value is not None else None

                    values = [
                        ("Route sequence similarity", "route_accuracy_percent"),
                        ("Segment precision", "segment_precision_percent"),
                        ("Segment recall", "segment_recall_percent"),
                        ("Segment F1", "segment_f1_percent"),
                    ]
                    cols = st.columns(4)
                    for col, (label, key) in zip(cols, values):
                        value = metric_value(key)
                        col.metric(label, f"{value:.2f}%" if value is not None else "N/A")

                    if result.get("inference_time_seconds") is not None:
                        st.caption(f"Prediction time: {float(result['inference_time_seconds']):.1f} seconds")
                    st.caption("Accuracy is reported as route sequence similarity against this trip's reference route.")
                else:
                    st.warning("Prediction succeeded, but the API response did not contain per_trip_evaluation. Confirm that the updated backend is running.")
            else:
                st.info("Select a trip and click **Run prediction & score trip** to calculate metrics for that specific trip. No overall benchmark score is substituted here.")

        except Exception as e:
            st.exception(e)

elif page == "Model performance":
    st.subheader("Held-out student-test performance")
    df = model_table()
    if df is None:
        st.error("Could not find student_test_model_comparison.csv or report_model_comparison_table.csv.")
    else:
        preferred = ["Model", "Candidate Accuracy", "Candidate Precision", "Candidate Recall",
                     "Candidate F1", "Macro F1", "Log Loss", "ROC AUC",
                     "Average Precision", "Observation Top-1 Accuracy", "Inference Time (s)"]
        shown = [c for c in preferred if c in df.columns]
        st.dataframe(df[shown] if shown else df, width="stretch", hide_index=True)
        st.caption("These are saved held-out student-test metrics across the test set, not per-trip live scores.")
        numeric_metrics = [c for c in ["Candidate Accuracy","Candidate Precision","Candidate Recall",
                           "Candidate F1","Macro F1","Observation Top-1 Accuracy"] if c in df.columns]
        if "Model" in df.columns and numeric_metrics:
            st.markdown("### Compare models")
            st.bar_chart(df.set_index("Model")[numeric_metrics])
    final = csv_file("student_test_final_metrics.csv")
    if final is not None:
        st.markdown("### Tuned Random Forest final evaluation")
        st.dataframe(final, width="stretch", hide_index=True)
    st.info("Live prediction uses the tuned Random Forest + HMM/Viterbi. This page shows saved benchmark comparisons; per-trip metrics are calculated on the Live map page.")

else:
    st.subheader("Project overview")
    a,b,c,d = st.columns(4)
    a.metric("ML model artifacts", "6")
    b.metric("Training trips", "80")
    c.metric("Evaluator trips", str(len(trip_ids)))
    d.metric("Input features", "8")
    st.markdown("""
    **PathSnap workflow**

    GPS observations → Candidate road generation → Feature extraction →
    ML candidate scoring → HMM/Viterbi decoding → Reconstructed route

    **Evaluated models:** Random Forest, Tuned Random Forest, Logistic Regression,
    Decision Tree, K-Nearest Neighbors and HistGradientBoosting.
    """)
    fi = csv_file("random_forest_feature_importance.csv")
    if fi is not None:
        st.markdown("### Random Forest feature importance")
        st.dataframe(fi, width="stretch", hide_index=True)
