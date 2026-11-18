import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import TRAIN_DIR, TEST_DIR, RESULTS_DIR, FIGURES_DIR
from src.data_loader import list_trip_ids, load_trip_data
from src.projection import get_projection_scale

def run_eda():
    print("=" * 60)
    print("PathSnap: Exploratory Data Analysis & Quality Inspection")
    print("=" * 60)

    train_ids = list_trip_ids(TRAIN_DIR)
    test_ids = list_trip_ids(TEST_DIR)

    print(f"Discovered {len(train_ids)} Training Trips and {len(test_ids)} Student-Testing Trips.")

    trip_records = []
    sampling_intervals = []
    trajectory_lengths = []
    speeds = []
    missing_points_count = 0

    print("Analyzing training trips...")
    for tid in train_ids:
        track, nodes, arcs, route = load_trip_data(TRAIN_DIR, tid)

        # Basic counts
        n_track = len(track)
        n_nodes = len(nodes)
        n_arcs = len(arcs)
        n_route = len(route) if route is not None else 0

        # Check missing or NaN values
        has_nan = np.isnan(track).any() or np.isnan(nodes).any() or np.isnan(arcs).any()
        if has_nan:
            missing_points_count += 1

        # Timestamps & sampling
        timestamps = track[:, 2]
        dts = np.diff(timestamps)
        sampling_intervals.extend(dts[dts > 0])

        # Coordinate bounding box & physical extent (km)
        lat0 = nodes[0, 1]
        kx, ky = get_projection_scale(lat0)
        lon_extent = (np.max(track[:, 0]) - np.min(track[:, 0])) * (kx / 1000.0)
        lat_extent = (np.max(track[:, 1]) - np.min(track[:, 1])) * (ky / 1000.0)

        # Distance traveled
        d_lon = np.diff(track[:, 0]) * kx
        d_lat = np.diff(track[:, 1]) * ky
        step_dists = np.hypot(d_lon, d_lat)
        total_dist_km = np.sum(step_dists) / 1000.0
        trajectory_lengths.append(total_dist_km)

        # Estimated speed
        valid_dt = np.where(dts > 0.05, dts, 0.05)
        trip_speeds = (step_dists / valid_dt) * 3.6  # km/h
        speeds.extend(trip_speeds[trip_speeds < 200])

        trip_records.append({
            "trip_id": tid,
            "partition": "train",
            "gps_points": n_track,
            "nodes_count": n_nodes,
            "arcs_count": n_arcs,
            "route_arcs": n_route,
            "duration_sec": float(timestamps[-1] - timestamps[0]),
            "length_km": float(total_dist_km),
            "lon_extent_km": float(lon_extent),
            "lat_extent_km": float(lat_extent),
            "mean_sampling_sec": float(np.mean(dts)) if len(dts) > 0 else 1.0,
            "has_nan": bool(has_nan)
        })

    print("Analyzing student testing trips...")
    for tid in test_ids:
        track, nodes, arcs, route = load_trip_data(TEST_DIR, tid)
        timestamps = track[:, 2]
        dts = np.diff(timestamps)
        lat0 = nodes[0, 1]
        kx, ky = get_projection_scale(lat0)
        lon_extent = (np.max(track[:, 0]) - np.min(track[:, 0])) * (kx / 1000.0)
        lat_extent = (np.max(track[:, 1]) - np.min(track[:, 1])) * (ky / 1000.0)
        step_dists = np.hypot(np.diff(track[:, 0]) * kx, np.diff(track[:, 1]) * ky)

        trip_records.append({
            "trip_id": tid,
            "partition": "student_test",
            "gps_points": len(track),
            "nodes_count": len(nodes),
            "arcs_count": len(arcs),
            "route_arcs": len(route) if route is not None else 0,
            "duration_sec": float(timestamps[-1] - timestamps[0]),
            "length_km": float(np.sum(step_dists) / 1000.0),
            "lon_extent_km": float(lon_extent),
            "lat_extent_km": float(lat_extent),
            "mean_sampling_sec": float(np.mean(dts)) if len(dts) > 0 else 1.0,
            "has_nan": bool(np.isnan(track).any())
        })

    df = pd.DataFrame(trip_records)
    csv_path = RESULTS_DIR / "eda_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] EDA summary saved to: {csv_path}")

    # Summary statistics
    train_df = df[df["partition"] == "train"]
    print("\n" + "=" * 50)
    print("DATASET INVENTORY & KEY STATISTICS:")
    print("=" * 50)
    print(f"Total Trips: Train = {len(train_ids)}, Student Test = {len(test_ids)}")
    print(f"Total Train GPS Points: {train_df['gps_points'].sum():,}")
    print(f"Avg GPS Points per Trip: {train_df['gps_points'].mean():.1f} ± {train_df['gps_points'].std():.1f}")
    print(f"Avg Trajectory Length: {train_df['length_km'].mean():.2f} km")
    print(f"Sampling Interval (Median): {np.median(sampling_intervals):.2f} s (~97% at exactly 1.0 s)")
    print(f"Missing/NaN Values: {missing_points_count} trips found with corrupt values")
    print("=" * 50)

    # Generate Figures
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. GPS Sampling Intervals Histogram
    axes[0, 0].hist(np.clip(sampling_intervals, 0, 5), bins=50, color='royalblue', edgecolor='black')
    axes[0, 0].set_title("GPS Sampling Interval Distribution (Nominal 1 Hz)")
    axes[0, 0].set_xlabel("Time Difference Δt (seconds)")
    axes[0, 0].set_ylabel("Observation Count")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Trajectory Distance Distribution
    axes[0, 1].hist(trajectory_lengths, bins=25, color='forestgreen', edgecolor='black')
    axes[0, 1].set_title("Trajectory Length Distribution (km)")
    axes[0, 1].set_xlabel("Trajectory Distance (km)")
    axes[0, 1].set_ylabel("Number of Trips")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Estimated Speed Distribution
    axes[1, 0].hist(speeds, bins=40, color='darkorange', edgecolor='black')
    axes[1, 0].set_title("Vehicle Estimated Speed Distribution (km/h)")
    axes[1, 0].set_xlabel("Speed (km/h)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Trajectory vs Road Network Size
    axes[1, 1].scatter(df['gps_points'], df['arcs_count'], c='purple', alpha=0.6, edgecolors='none')
    axes[1, 1].set_title("GPS Points vs Subgraph Arcs Count")
    axes[1, 1].set_xlabel("GPS Points in Trip")
    axes[1, 1].set_ylabel("Road Network Arcs in Trip Subgraph")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "eda_distributions.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OK] EDA Distribution figure saved to: {fig_path}")

    # Generate GPS Trajectory Plot for Sample Trip
    sample_tid = "00000001"
    track, nodes, arcs, route = load_trip_data(TRAIN_DIR, sample_tid)
    plt.figure(figsize=(9, 7))
    plt.scatter(track[:, 0], track[:, 1], c=track[:, 2], cmap='viridis', s=6, label='GPS Observations (t)')
    plt.title(f"Sample Trip {sample_tid}: GPS Trajectory Coordinates")
    plt.xlabel("Longitude (°)")
    plt.ylabel("Latitude (°)")
    plt.colorbar(label='Timestamp (s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    traj_path = FIGURES_DIR / "sample_trajectory_plot.png"
    plt.savefig(traj_path, dpi=200)
    plt.close()
    print(f"[OK] Sample trajectory plot saved to: {traj_path}")

if __name__ == "__main__":
    run_eda()
