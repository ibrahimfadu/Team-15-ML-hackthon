"""
scripts/run_eda.py
Conducts Exploratory Data Analysis (EDA) on the PathSnap dataset and generates:
1. Dataset summary statistics (missing values, sampling rates, counts)
2. Interactive Folium map of a representative trip (GPS track vs. Road network vs. Ground Truth)
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import folium

from src.data_loader import load_single_trip


def run_eda(train_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = output_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    print("=== PathSnap Dataset Exploratory Analysis ===")

    train_folders = [f for f in train_dir.iterdir() if f.is_dir() and not f.name.startswith(".")]
    print(f"Total training trips found: {len(train_folders)}")

    total_gps_points = 0
    total_arcs = 0
    total_nodes = 0
    sampling_intervals = []
    missing_values = 0

    trip_summaries = []

    # Analyze first 10 trips in detail for speed
    for i, folder in enumerate(train_folders[:10]):
        trip = load_single_trip(folder)
        num_gps = len(trip.track)
        total_gps_points += num_gps
        total_arcs += len(trip.arcs)
        total_nodes += len(trip.nodes)

        # Check missing values
        missing_values += trip.track.isna().sum().sum()

        # Check sampling rate
        dt = np.diff(trip.track["t"].values)
        sampling_intervals.extend(dt)

        trip_summaries.append({
            "trip_id": trip.trip_id,
            "gps_points": num_gps,
            "route_arcs": len(trip.route) if trip.route is not None else 0,
            "total_nodes": len(trip.nodes),
            "total_arcs": len(trip.arcs),
            "duration_sec": float(trip.track["t"].iloc[-1] - trip.track["t"].iloc[0]),
            "min_lon": float(trip.track["lon"].min()),
            "max_lon": float(trip.track["lon"].max()),
            "min_lat": float(trip.track["lat"].min()),
            "max_lat": float(trip.track["lat"].max()),
        })

    sampling_arr = np.array(sampling_intervals)
    exact_1s = (np.abs(sampling_arr - 1.0) < 1e-3).sum()
    pct_1s = (exact_1s / len(sampling_arr)) * 100.0

    eda_stats = {
        "num_training_trips": len(train_folders),
        "analyzed_trips": len(trip_summaries),
        "missing_gps_values": int(missing_values),
        "sampling_interval_mean": float(np.mean(sampling_arr)),
        "sampling_interval_median": float(np.median(sampling_arr)),
        "pct_exact_1_second_intervals": float(pct_1s),
        "trips": trip_summaries
    }

    summary_file = output_dir / "results" / "eda_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(eda_stats, f, indent=2)

    print(f"Missing GPS values: {missing_values}")
    print(f"Mean sampling interval: {np.mean(sampling_arr):.2f}s (~{pct_1s:.1f}% exactly 1s apart)")
    print(f"Summary saved to: {summary_file}")

    # Generate Folium Map for Trip 00000001
    sample_trip = load_single_trip(train_dir / "00000001")
    mean_lat = sample_trip.track["lat"].mean()
    mean_lon = sample_trip.track["lon"].mean()

    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=15, tiles="OpenStreetMap")

    # 1. Add Ground Truth Route Arcs (Green Polyline)
    if sample_trip.route is not None:
        route_arc_ids = sample_trip.route["arc_id"].values
        route_lines = []
        for aid in route_arc_ids:
            u = sample_trip.arcs.iloc[aid]["u"]
            v = sample_trip.arcs.iloc[aid]["v"]
            u_lat = sample_trip.nodes.iloc[u]["lat"]
            u_lon = sample_trip.nodes.iloc[u]["lon"]
            v_lat = sample_trip.nodes.iloc[v]["lat"]
            v_lon = sample_trip.nodes.iloc[v]["lon"]
            route_lines.append([[u_lat, u_lon], [v_lat, v_lon]])

        folium.PolyLine(
            locations=route_lines,
            color="#2ecc71",
            weight=4,
            opacity=0.8,
            popup="Ground Truth Route",
            tooltip="Ground Truth Route (Road Network)"
        ).add_to(m)

    # 2. Add Raw GPS Observations (Blue CircleMarkers)
    # Subsample GPS points slightly if too dense, but keep full track path
    gps_coords = sample_trip.track[["lat", "lon"]].values.tolist()
    folium.PolyLine(
        locations=gps_coords,
        color="#3498db",
        weight=2,
        dash_array="5, 5",
        opacity=0.7,
        tooltip="Raw GPS Path"
    ).add_to(m)

    for i, (lat, lon) in enumerate(gps_coords[::5]):  # every 5th point for clean visualization
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="#2980b9",
            fill=True,
            fill_color="#3498db",
            fill_opacity=0.9,
            popup=f"GPS Obs {i*5}: ({lat:.6f}, {lon:.6f})"
        ).add_to(m)

    map_path = maps_dir / "sample_trip_eda.html"
    m.save(str(map_path))
    print(f"Interactive Folium map saved to: {map_path}")


if __name__ == "__main__":
    train_path = Path("Student folder/train")
    out_path = Path("artifacts")
    run_eda(train_path, out_path)
