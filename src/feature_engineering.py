"""
src/feature_engineering.py
Extracts spatial and kinematic features for (GPS observation, Candidate Arc) pairs.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from src.data_loader import TripData
from src.geometry import (
    to_metric_coordinates,
    calculate_bearing,
    angle_difference,
    point_to_segment_projection
)


def align_ground_truth_route(trip: TripData, gps_x: np.ndarray, gps_y: np.ndarray, ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray) -> np.ndarray:
    """
    Aligns each GPS point to its ground truth arc in trip.route using forward projection along the route.
    Returns an array of true_arc_id for each GPS point.
    """
    if trip.route is None or len(trip.route) == 0:
        return np.full(len(gps_x), -1, dtype=int)

    route_arcs = trip.route["arc_id"].values
    num_route = len(route_arcs)

    # Coordinates of each route arc
    r_u = trip.arcs.loc[route_arcs, "u"].values
    r_v = trip.arcs.loc[route_arcs, "v"].values
    r_ax, r_ay = ax[r_u], ay[r_u]
    r_bx, r_by = bx[r_v], by[r_v]

    true_arcs = np.zeros(len(gps_x), dtype=int)
    cur_idx = 0
    window = 20

    for i in range(len(gps_x)):
        px, py = gps_x[i], gps_y[i]
        best_d = 1e9
        best_r = cur_idx
        for r in range(cur_idx, min(num_route, cur_idx + window)):
            d, _, _, _ = point_to_segment_projection(px, py, r_ax[r], r_ay[r], r_bx[r], r_by[r])
            if d < best_d:
                best_d = d
                best_r = r
        cur_idx = best_r
        true_arcs[i] = route_arcs[best_r]

    return true_arcs


def extract_trip_features(trip: TripData, k_candidates: int = 10, is_training: bool = True) -> pd.DataFrame:
    """
    Processes a trip into candidate feature vectors for each GPS observation.
    """
    ref_lon, ref_lat = trip.track["lon"].mean(), trip.track["lat"].mean()

    # Metric coordinates
    node_x, node_y = to_metric_coordinates(trip.nodes["lon"].values, trip.nodes["lat"].values, ref_lon, ref_lat)
    gps_x, gps_y = to_metric_coordinates(trip.track["lon"].values, trip.track["lat"].values, ref_lon, ref_lat)

    u_nodes = trip.arcs["u"].values
    v_nodes = trip.arcs["v"].values

    ax, ay = node_x[u_nodes], node_y[u_nodes]
    bx, by = node_x[v_nodes], node_y[v_nodes]

    # Precompute road segment lengths and bearings
    seg_dx = bx - ax
    seg_dy = by - ay
    seg_lengths = np.hypot(seg_dx, seg_dy)
    seg_mid_x = (ax + bx) / 2.0
    seg_mid_y = (ay + by) / 2.0

    seg_bearings = (np.degrees(np.arctan2(seg_dx, seg_dy)) + 360.0) % 360.0

    # Build KD-Tree on segment midpoints
    tree = KDTree(np.column_stack([seg_mid_x, seg_mid_y]))

    # Compute vehicle velocities and headings between consecutive GPS points
    num_gps = len(trip.track)
    times = trip.track["t"].values
    dt = np.diff(times, prepend=times[0] - 1.0)
    dt[dt <= 0] = 1.0

    gps_dx = np.diff(gps_x, prepend=gps_x[0])
    gps_dy = np.diff(gps_y, prepend=gps_y[0])
    speeds = np.hypot(gps_dx, gps_dy) / dt
    headings = (np.degrees(np.arctan2(gps_dx, gps_dy)) + 360.0) % 360.0

    # Ground truth alignment if route exists
    true_arcs = align_ground_truth_route(trip, gps_x, gps_y, node_x, node_y, node_x, node_y) if is_training else np.full(num_gps, -1)

    # Query nearest candidates for all GPS points
    _, cand_indices = tree.query(np.column_stack([gps_x, gps_y]), k=k_candidates)

    records = []
    for i in range(num_gps):
        px, py = gps_x[i], gps_y[i]
        speed = speeds[i]
        heading = headings[i]
        target_arc = true_arcs[i]

        candidates = cand_indices[i]

        # Calculate exact distances for ranking
        cand_dists = []
        cand_projs = []
        for c_idx in candidates:
            d, t, _, _ = point_to_segment_projection(px, py, ax[c_idx], ay[c_idx], bx[c_idx], by[c_idx])
            cand_dists.append(d)
            cand_projs.append(t)

        sorted_order = np.argsort(cand_dists)

        for rank, s_idx in enumerate(sorted_order):
            c_idx = candidates[s_idx]
            d = cand_dists[s_idx]
            t = cand_projs[s_idx]
            seg_len = seg_lengths[c_idx]
            seg_bearing = seg_bearings[c_idx]

            head_diff = angle_difference(heading, seg_bearing)
            d_start = np.hypot(px - ax[c_idx], py - ay[c_idx])
            d_end = np.hypot(px - bx[c_idx], py - by[c_idx])

            label = 1 if (is_training and c_idx == target_arc) else 0

            records.append({
                "trip_id": trip.trip_id,
                "obs_id": i,
                "candidate_arc_id": int(c_idx),
                "dist_to_segment": float(d),
                "projection_ratio": float(t),
                "heading_diff": float(head_diff),
                "segment_length": float(seg_len),
                "dist_to_start": float(d_start),
                "dist_to_end": float(d_end),
                "vehicle_speed": float(speed),
                "candidate_rank": int(rank),
                "target_arc_id": int(target_arc),
                "is_match": int(label)
            })

    return pd.DataFrame(records)
