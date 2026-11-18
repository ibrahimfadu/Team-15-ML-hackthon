"""
ui/app.py
Evaluator-Facing User Interface for Laptop 2.

Runs on Laptop 2:
- Allows evaluator to enter Laptop 1's IP address.
- Fetches the 10 evaluation trips from the FastAPI server.
- Sends the selected trip for inference.
- Displays prediction metrics, route continuity, and interactive Folium map.
"""

import streamlit as st
import requests
import streamlit.components.v1 as components

st.set_page_config(
    page_title="PathSnap Evaluator Interface",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ PathSnap: Map-Matching & Route Reconstruction")
st.markdown("### Laptop 2 — Evaluator Interface")

# Sidebar Configuration
st.sidebar.header("🔌 Network Connection")
api_host = st.sidebar.text_input("Laptop 1 API Base URL:", value="http://localhost:8000")
st.sidebar.caption("Example: http://192.168.1.15:8000")

# Connection Test
server_online = False
trips = []

try:
    health_resp = requests.get(f"{api_host}/health", timeout=2)
    if health_resp.status_code == 200:
        server_online = True
        st.sidebar.success("🟢 Connected to Laptop 1 Server")
        
        # Fetch trips
        trips_resp = requests.get(f"{api_host}/trips", timeout=3)
        if trips_resp.status_code == 200:
            trips = trips_resp.json().get("trips", [])
except Exception as e:
    st.sidebar.error("🔴 Offline: Cannot reach Laptop 1")
    st.sidebar.info("Ensure Laptop 1 is running `uvicorn api.main:app --host 0.0.0.0 --port 8000` and both laptops share the same network.")

# Fallback default trips if server not connected yet
if not trips:
    trips = ["00000015", "00000017", "00000023", "00000030", "00000049", "00000059", "00000067", "00000073", "00000087", "00000094"]

st.write("---")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("📋 Select Evaluation Trip")
    selected_trip = st.selectbox("Choose a reserved evaluation trip:", trips)
    
    run_btn = st.button("🚀 Run Map-Matching & Route Reconstruction", type="primary", use_container_width=True)

if run_btn:
    with st.spinner(f"Requesting inference for Trip [{selected_trip}] from Laptop 1..."):
        try:
            resp = requests.post(f"{api_host}/predict/{selected_trip}", timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                
                st.success(f"Trip [{selected_trip}] successfully reconstructed in {data['latencies']['total_latency_sec']}s!")
                
                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                metrics = data["metrics"]
                obs_acc = metrics.get("observation_top1_accuracy")
                if isinstance(obs_acc, float):
                    obs_acc_str = f"{obs_acc * 100:.2f}%"
                else:
                    obs_acc_str = str(obs_acc)
                    
                cont_score = metrics.get("route_continuity_score", 0.0)
                m1.metric("Observation Accuracy", obs_acc_str)
                m2.metric("Route Continuity", f"{cont_score * 100:.1f}%")
                m3.metric("Reconstructed Route Length", f"{data['reconstructed_route_length']} arcs")
                m4.metric("Inference Latency", f"{data['latencies']['total_latency_sec']}s")
                
                st.write("---")
                
                # Route Arcs Display
                with st.expander("🔍 View Reconstructed Sequence of Arcs (Road IDs)", expanded=False):
                    st.write(data["full_predicted_route"])
                    
                # Interactive Folium Map Display
                st.subheader("🗺️ Reconstructed Route Visualization")
                st.caption("Green Line = Reconstructed Road Route | Blue Dotted Line = Raw GPS Observation Track")
                
                map_html = data.get("map_html", "")
                if map_html:
                    components.html(map_html, height=550, scrolling=True)
                else:
                    st.warning("Map visualization not returned by server.")
                    
            else:
                st.error(f"Server returned error code {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Failed to communicate with API server: {e}")
