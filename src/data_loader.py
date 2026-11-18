import os
from pathlib import Path
import numpy as np

def list_trip_ids(partition_dir):
    """
    Returns sorted list of trip IDs in the given partition directory.
    """
    partition_path = Path(partition_dir)
    trips = [d.name for d in partition_path.iterdir() if d.is_dir()]
    return sorted(trips)

def load_trip_data(trip_dir, trip_id):
    """
    Loads .track, .nodes, .arcs, and .route (if available) for a given trip.
    Returns:
        track: (N, 3) float array [lon, lat, timestamp]
        nodes: (M, 2) float array [lon, lat]
        arcs:  (K, 2) int array [from_node, to_node]
        route: (R,) int array [arc_ids] or None
    """
    trip_path = Path(trip_dir) / trip_id

    track_file = trip_path / f"{trip_id}.track"
    nodes_file = trip_path / f"{trip_id}.nodes"
    arcs_file = trip_path / f"{trip_id}.arcs"
    route_file = trip_path / f"{trip_id}.route"

    if not track_file.exists():
        raise FileNotFoundError(f"Missing track file: {track_file}")
    if not nodes_file.exists():
        raise FileNotFoundError(f"Missing nodes file: {nodes_file}")
    if not arcs_file.exists():
        raise FileNotFoundError(f"Missing arcs file: {arcs_file}")

    # Fast file loading using numpy
    track = np.loadtxt(track_file, dtype=np.float64)
    nodes = np.loadtxt(nodes_file, dtype=np.float64)
    arcs = np.loadtxt(arcs_file, dtype=np.int64)

    route = None
    if route_file.exists():
        route = np.loadtxt(route_file, dtype=np.int64)
        if route.ndim == 0:
            route = np.array([route])

    return track, nodes, arcs, route
