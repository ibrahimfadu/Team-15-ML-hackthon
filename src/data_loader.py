"""
src/data_loader.py
Handles loading and parsing of PathSnap trip files:
- .track (GPS observations: lon, lat, timestamp)
- .nodes (Road network nodes: lon, lat)
- .arcs  (Road network edges: from_node, to_node)
- .route (Ground truth sequential arc IDs)
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class TripData:
    def __init__(self, trip_id: str, track: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame, route: Optional[pd.DataFrame] = None):
        self.trip_id = trip_id
        self.track = track  # lon, lat, t
        self.nodes = nodes  # lon, lat (index is node_id)
        self.arcs = arcs    # u, v (index is arc_id)
        self.route = route  # arc_id (chronological)

    def __repr__(self):
        return f"<TripData {self.trip_id}: {len(self.track)} GPS pts, {len(self.arcs)} arcs, {len(self.route) if self.route is not None else 0} route arcs>"


def load_single_trip(trip_dir: Path) -> TripData:
    trip_id = trip_dir.name
    track_path = trip_dir / f"{trip_id}.track"
    nodes_path = trip_dir / f"{trip_id}.nodes"
    arcs_path = trip_dir / f"{trip_id}.arcs"
    route_path = trip_dir / f"{trip_id}.route"

    track = pd.read_csv(track_path, sep=r"\s+", header=None, names=["lon", "lat", "t"], dtype={"lon": float, "lat": float, "t": float})
    nodes = pd.read_csv(nodes_path, sep=r"\s+", header=None, names=["lon", "lat"], dtype={"lon": float, "lat": float})
    arcs = pd.read_csv(arcs_path, sep=r"\s+", header=None, names=["u", "v"], dtype={"u": int, "v": int})

    route = None
    if route_path.exists():
        route = pd.read_csv(route_path, sep=r"\s+", header=None, names=["arc_id"], dtype={"arc_id": int})

    return TripData(trip_id=trip_id, track=track, nodes=nodes, arcs=arcs, route=route)


def load_dataset_partition(base_dir: Path) -> Dict[str, TripData]:
    """
    Loads all trips in a given partition folder (e.g. Student folder/train or student_test).
    Returns a dict mapping trip_id -> TripData.
    """
    trips = {}
    for trip_folder in sorted(base_dir.iterdir()):
        if trip_folder.is_dir() and not trip_folder.name.startswith("."):
            trips[trip_folder.name] = load_single_trip(trip_folder)
    return trips
