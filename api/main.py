"""
api/main.py
FastAPI Backend Server for Laptop 1 (Model and API Host).

Responsibilities:
1. Loads the trained & tuned Random Forest model and HMM Viterbi decoder.
2. Exposes endpoints for the 10 evaluation trips.
3. Performs candidate extraction, Random Forest emission prediction, and HMM decoding.
4. Computes prediction accuracy server-side against internal reference ground truth.
5. Emits live terminal logs on the console for the evaluators to witness.
"""

import sys
from pathlib import Path
import time
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
import folium
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.data_loader import load_single_trip
from src.feature_engineering import extract_trip_features
from src.hmm_decoder import HMMViterbiDecoder

# Configure live console logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [PathSnap-API] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("PathSnap")

app = FastAPI(
    title="PathSnap Map-Matching API",
    description="Distributed GPS Trajectory Map-Matching & Route Reconstruction API",
    version="1.0.0"
)

# Enable CORS for Laptop 2 communication over local network / Wi-Fi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURE_COLS = [
    "dist_to_segment",
    "projection_ratio",
    "heading_diff",
    "segment_length",
    "dist_to_start",
    "dist_to_end",
    "vehicle_speed",
    "candidate_rank"
]

# Locate evaluation dataset folder
EVAL_DIR = Path("evaluator_test (1)/evaluator_test")
if not EVAL_DIR.exists():
    EVAL_DIR = Path("evaluator_test")
if not EVAL_DIR.exists():
    EVAL_DIR = Path("Student folder/student_test")

# Locate Model Artifact (prefer tuned, fallback to base)
MODEL_PATH = Path("artifacts/models/random_forest_tuned.joblib")
if not MODEL_PATH.exists():
    MODEL_PATH = Path("artifacts/models/random_forest.joblib")

# Global loaded assets
MODEL_PIPELINE = None


def get_model():
    global MODEL_PIPELINE
    if MODEL_PIPELINE is None:
        if MODEL_PATH.exists():
            MODEL_PIPELINE = joblib.load(MODEL_PATH)
            logger.info(f"Loaded trained classifier successfully from {MODEL_PATH}")
        else:
            raise HTTPException(status_code=500, detail=f"Model artifact not found at {MODEL_PATH}")
    return MODEL_PIPELINE


@app.on_event("startup")
def startup_event():
    logger.info("Initializing PathSnap Server on Laptop 1...")
    get_model()
    logger.info(f"Using evaluation trips from: {EVAL_DIR.resolve()}")
    logger.info("Ready to process live requests from Laptop 2!")


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "PathSnap Inference Server (Laptop 1)",
        "model_loaded": MODEL_PIPELINE is not None,
        "evaluation_dataset": str(EVAL_DIR)
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/trips")
def list_evaluation_trips():
    """Lists available evaluation trips for Laptop 2 dropdown."""
    if not EVAL_DIR.exists():
        raise HTTPException(status_code=404, detail="Evaluation directory not found")
    trip_folders = sorted([f.name for f in EVAL_DIR.iterdir() if f.is_dir() and not f.name.startswith(".")])
    logger.info(f"Listing {len(trip_folders)} available evaluation trips.")
    return {"trips": trip_folders}


@app.post("/predict/{trip_id}")
@app.get("/predict/{trip_id}")
def evaluate_trip(trip_id: str):
    """
    Core Inference & Evaluation Endpoint:
    1. Loads evaluation trip from server-side files
    2. Runs feature extraction + Random Forest scoring
    3. Solves continuous route using HMM Viterbi decoding
    4. Computes accuracy against internal ground truth (kept server-side)
    5. Returns prediction and visualization
    """
    logger.info(f">>> RECEIVED INFERENCE REQUEST for Trip ID: [{trip_id}] from Laptop 2")
    start_time = time.time()

    trip_folder = EVAL_DIR / trip_id
    if not trip_folder.exists():
        logger.error(f"Trip [{trip_id}] not found in {EVAL_DIR}")
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found on server")

    model = get_model()

    # 1. Load Trip Data (Ground truth stays internal)
    trip = load_single_trip(trip_folder)
    num_gps = len(trip.track)
    logger.info(f"Loaded Trip [{trip_id}]: {num_gps} GPS observations, {len(trip.arcs)} arcs")

    # 2. Extract Candidate Features
    t_feat0 = time.time()
    has_route = trip.route is not None and len(trip.route) > 0
    candidate_df = extract_trip_features(trip, k_candidates=10, is_training=has_route)
    feat_time = time.time() - t_feat0

    # 3. Model Inference (Random Forest Emission Probabilities)
    t_inf0 = time.time()
    X_features = candidate_df[FEATURE_COLS].values
    emission_probs = model.predict_proba(X_features)[:, 1]
    model_time = time.time() - t_inf0

    # 4. HMM & Viterbi Route Reconstruction
    t_hmm0 = time.time()
    decoder = HMMViterbiDecoder(trip.arcs)
    decoded_arc_sequence = decoder.decode(candidate_df, emission_probs)
    compressed_route = decoder.compress_route(decoded_arc_sequence)
    hmm_time = time.time() - t_hmm0

    # 5. Route Continuity
    continuity_score = decoder.evaluate_route_continuity(compressed_route)

    # 6. Server-Side Accuracy Evaluation against Ground Truth (if available)
    obs_accuracy = None
    route_jaccard = None
    if has_route:
        true_arcs = candidate_df.groupby("obs_id")["target_arc_id"].first().values
        correct_count = sum(p == t for p, t in zip(decoded_arc_sequence, true_arcs) if t >= 0)
        obs_accuracy = float(correct_count / len(decoded_arc_sequence)) if len(decoded_arc_sequence) > 0 else 0.0

        true_route_set = set(trip.route["arc_id"].values.tolist())
        pred_route_set = set(compressed_route)
        intersection = len(true_route_set.intersection(pred_route_set))
        union = len(true_route_set.union(pred_route_set))
        route_jaccard = float(intersection / union) if union > 0 else 0.0

    total_latency = time.time() - start_time
    logger.info(f"Trip [{trip_id}] Processed successfully in {total_latency:.3f}s!")
    if obs_accuracy is not None:
        logger.info(f"Obs Accuracy: {obs_accuracy*100:.2f}% | Continuity: {continuity_score*100:.1f}% | Route Length: {len(compressed_route)} arcs")

    # 7. Generate Interactive Folium Map
    mean_lat = float(trip.track["lat"].mean())
    mean_lon = float(trip.track["lon"].mean())
    folium_map = folium.Map(location=[mean_lat, mean_lon], zoom_start=14, tiles="OpenStreetMap")

    # Add Raw GPS Line
    gps_coords = trip.track[["lat", "lon"]].values.tolist()
    folium.PolyLine(gps_coords, color="#3498db", weight=2, dash_array="5, 5", tooltip="Raw GPS Track").add_to(folium_map)

    # Add Predicted Route Lines (Green)
    pred_lines = []
    for aid in compressed_route:
        if aid < len(trip.arcs):
            u = trip.arcs.iloc[aid]["u"]
            v = trip.arcs.iloc[aid]["v"]
            u_coord = [float(trip.nodes.iloc[u]["lat"]), float(trip.nodes.iloc[u]["lon"])]
            v_coord = [float(trip.nodes.iloc[v]["lat"]), float(trip.nodes.iloc[v]["lon"])]
            pred_lines.append([u_coord, v_coord])

    if pred_lines:
        folium.PolyLine(pred_lines, color="#2ecc71", weight=5, opacity=0.85, tooltip="PathSnap Reconstructed Route").add_to(folium_map)

    # Fit bounds
    min_lat, max_lat = float(trip.track["lat"].min()), float(trip.track["lat"].max())
    min_lon, max_lon = float(trip.track["lon"].min()), float(trip.track["lon"].max())
    folium_map.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    map_html = folium_map.get_root().render()

    return {
        "status": "success",
        "trip_id": trip_id,
        "gps_points_count": num_gps,
        "reconstructed_route_length": len(compressed_route),
        "predicted_route_head": compressed_route[:20],
        "full_predicted_route": compressed_route,
        "metrics": {
            "observation_top1_accuracy": round(obs_accuracy, 4) if obs_accuracy is not None else "N/A (Hidden Ground Truth)",
            "route_continuity_score": round(continuity_score, 4),
            "route_jaccard_similarity": round(route_jaccard, 4) if route_jaccard is not None else "N/A"
        },
        "latencies": {
            "feature_extraction_sec": round(feat_time, 3),
            "model_inference_sec": round(model_time, 3),
            "hmm_decoding_sec": round(hmm_time, 3),
            "total_latency_sec": round(total_latency, 3)
        },
        "map_html": map_html
    }


if __name__ == "__main__":
    import uvicorn
    # Listen on 0.0.0.0 so Laptop 2 can connect over local network
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
