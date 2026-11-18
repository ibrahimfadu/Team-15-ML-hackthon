"""Readers and validators for the PathSnap whitespace-delimited trip format."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class TripData:
    trip_id: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: np.ndarray | None = None


def _read_table(path: Path, names: list[str], dtypes: dict | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtypes, engine="python")
    if df.empty:
        raise ValueError(f"File is empty: {path}")
    return df


def read_trip(trip_dir: str | Path, require_route: bool = False) -> TripData:
    """Read one trip directory. .route is optional for prediction-only trips."""
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name
    track = _read_table(trip_dir / f"{trip_id}.track", ["longitude", "latitude", "timestamp"],
                        {0: "float64", 1: "float64", 2: "float64"})
    nodes = _read_table(trip_dir / f"{trip_id}.nodes", ["longitude", "latitude"],
                        {0: "float64", 1: "float64"})
    arcs = _read_table(trip_dir / f"{trip_id}.arcs", ["from_node_id", "to_node_id"],
                       {0: "int64", 1: "int64"})
    route_path = trip_dir / f"{trip_id}.route"
    route = None
    if route_path.exists():
        route_df = _read_table(route_path, ["arc_id"], {0: "int64"})
        route = route_df["arc_id"].to_numpy(dtype=np.int64)
    elif require_route:
        raise FileNotFoundError(f"Reference route required but missing: {route_path}")
    return TripData(trip_id, track, nodes, arcs, route)


def validate_trip(trip: TripData) -> dict:
    """Return data-quality counts without modifying the source data."""
    n_nodes, n_arcs = len(trip.nodes), len(trip.arcs)
    endpoints = trip.arcs[["from_node_id", "to_node_id"]].to_numpy()
    invalid_endpoints = int(((endpoints < 0) | (endpoints >= n_nodes)).any(axis=1).sum())
    ts = trip.track["timestamp"].to_numpy(dtype=float)
    gaps = np.diff(ts)
    route = trip.route
    missing_route_ids = int(((route < 0) | (route >= n_arcs)).sum()) if route is not None else None
    return {
        "trip_id": trip.trip_id,
        "gps_observations": len(trip.track),
        "nodes": n_nodes,
        "arcs": n_arcs,
        "route_edges": int(len(route)) if route is not None else None,
        "missing_route_ids": missing_route_ids,
        "invalid_arc_endpoints": invalid_endpoints,
        "missing_track_values": int(trip.track.isna().sum().sum()),
        "nonpositive_time_steps": int((gaps <= 0).sum()),
        "time_gaps_gt_1s": int((gaps > 1).sum()),
        "max_time_gap_seconds": float(gaps.max()) if len(gaps) else 0.0,
        "median_time_gap_seconds": float(np.median(gaps)) if len(gaps) else 0.0,
    }
