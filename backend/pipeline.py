"""PathSnap inference: original candidate generation + tuned RF + HMM/Viterbi.

Evaluator prediction reads only .track, .nodes and .arcs. It never opens .route.
"""
from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import json
import math
import time

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model_training_deliverables" / "models" / "tuned_random_forest.joblib"
CONFIG_PATH = PROJECT_ROOT / "models" / "model_training_deliverables" / "models" / "pathsnap_model_config.json"
TRAIN_ROOT = PROJECT_ROOT / "data" / "training_source" / "Student folder" / "train"
FEATURES = [
    "dist_to_segment", "projection_ratio", "heading_diff",
    "segment_length", "dist_to_start", "dist_to_end",
    "gps_speed", "candidate_rank",
]
MAX_OBS_PER_TRIP = 600
MIDPOINT_K = 16
NEAR_NODE_K = 12
MAX_CANDIDATES = 40
EARTH_R = 6371000.0

HMM_TOP_K = 8
HMM_MAX_HOPS = 8
HMM_SIGMA_METRES = 100.0
HMM_TRANSITION_WEIGHT = 0.05
HMM_UNREACHABLE_LOG_PENALTY = 10.0
HMM_EPS = 1e-9

_model = None
_feature_medians = None


def _load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        if getattr(_model, "n_features_in_", None) != len(FEATURES):
            raise ValueError("Loaded model does not expect exactly 8 features.")
    return _model


def _xy(lon, lat, lon0, lat0):
    x = EARTH_R * np.radians(np.asarray(lon, dtype=float) - lon0) * np.cos(np.radians(lat0))
    y = EARTH_R * np.radians(np.asarray(lat, dtype=float) - lat0)
    return np.column_stack([x, y])


def _angle_diff(a, b):
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def _select_obs(track):
    n = len(track)
    if n <= MAX_OBS_PER_TRIP:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, MAX_OBS_PER_TRIP).round().astype(int))


def _read_trip_files(trip_dir: Path):
    """Read only inference inputs. Deliberately never reads the .route file."""
    tid = trip_dir.name
    track = pd.read_csv(
        trip_dir / f"{tid}.track", sep=r"\s+", header=None,
        names=["longitude", "latitude", "timestamp"],
    )
    nodes = pd.read_csv(
        trip_dir / f"{tid}.nodes", sep=r"\s+", header=None,
        names=["longitude", "latitude"],
    )
    arcs = pd.read_csv(
        trip_dir / f"{tid}.arcs", sep=r"\s+", header=None,
        names=["from_node_id", "to_node_id"],
    )
    if track.empty or nodes.empty or arcs.empty:
        raise ValueError(f"Trip {tid} has an empty track, nodes or arcs file.")
    return track, nodes, arcs


def _candidate_table(trip_dir: Path):
    """Faithful inference-only version of notebook cell 25 candidate generation."""
    tid = trip_dir.name
    track, nodes, arcs = _read_trip_files(trip_dir)
    endpoints = arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=int)
    valid_mask = (
        (endpoints[:, 0] >= 0) & (endpoints[:, 1] >= 0)
        & (endpoints[:, 0] < len(nodes)) & (endpoints[:, 1] < len(nodes))
    )
    valid_arc_ids = np.flatnonzero(valid_mask)
    if not len(valid_arc_ids):
        raise ValueError(f"Trip {tid} has no arcs with valid node endpoints.")

    lon0 = float(track.longitude.median())
    lat0 = float(track.latitude.median())
    node_xy = _xy(nodes.longitude, nodes.latitude, lon0, lat0)
    gps_all = _xy(track.longitude, track.latitude, lon0, lat0)

    u = endpoints[valid_arc_ids, 0]
    v = endpoints[valid_arc_ids, 1]
    a_all, b_all = node_xy[u], node_xy[v]
    vec_all = b_all - a_all
    seg_len_all = np.linalg.norm(vec_all, axis=1)
    mid_all = (a_all + b_all) / 2.0

    mid_tree = cKDTree(mid_all)
    node_tree = cKDTree(node_xy)
    node_to_arcs = defaultdict(list)
    for local_idx, (uu, vv) in enumerate(zip(u, v)):
        node_to_arcs[int(uu)].append(local_idx)
        node_to_arcs[int(vv)].append(local_idx)

    obs_idx = _select_obs(track)
    gps_xy = gps_all[obs_idx]
    timestamps = track.timestamp.to_numpy(dtype=float)[obs_idx]
    rows = []

    for oi, (point, stamp) in enumerate(zip(gps_xy, timestamps)):
        if oi < len(gps_xy) - 1:
            delta = gps_xy[oi + 1] - point
            dt = timestamps[oi + 1] - stamp
        elif oi > 0:
            delta = point - gps_xy[oi - 1]
            dt = stamp - timestamps[oi - 1]
        else:
            delta, dt = np.array([0.0, 0.0]), np.nan

        gps_heading = (
            float(np.arctan2(delta[0], delta[1]))
            if np.linalg.norm(delta) > 0 else np.nan
        )
        speed = float(np.linalg.norm(delta) / dt) if dt > 0 else np.nan

        _, mid_ids = mid_tree.query(point, k=min(MIDPOINT_K, len(mid_all)))
        mid_ids = np.atleast_1d(mid_ids).astype(int).tolist()
        _, node_ids = node_tree.query(point, k=min(NEAR_NODE_K, len(node_xy)))
        node_ids = np.atleast_1d(node_ids).astype(int).tolist()

        proposals = set(mid_ids)
        for node_id in node_ids:
            proposals.update(node_to_arcs.get(int(node_id), []))
        proposals = sorted(proposals, key=lambda j: np.linalg.norm(mid_all[j] - point))[:MAX_CANDIDATES]

        candidates = []
        for local_j in proposals:
            vec = vec_all[local_j]
            denom = float(vec @ vec)
            projection = 0.0 if denom == 0 else float(
                np.clip(((point - a_all[local_j]) @ vec) / denom, 0, 1)
            )
            nearest = a_all[local_j] + projection * vec
            distance = float(np.linalg.norm(point - nearest))
            segment_bearing = (
                float(np.arctan2(vec[0], vec[1]))
                if seg_len_all[local_j] > 0 else np.nan
            )
            heading_diff = (
                float(_angle_diff(gps_heading, segment_bearing))
                if np.isfinite(gps_heading) and np.isfinite(segment_bearing)
                else np.nan
            )
            arc_id = int(valid_arc_ids[local_j])
            candidates.append({
                "trip_id": tid,
                "obs_index": int(obs_idx[oi]),
                "timestamp": float(stamp),
                "arc_id": arc_id,
                "dist_to_segment": distance,
                "projection_ratio": projection,
                "heading_diff": heading_diff,
                "segment_length": float(seg_len_all[local_j]),
                "dist_to_start": float(np.linalg.norm(point - a_all[local_j])),
                "dist_to_end": float(np.linalg.norm(point - b_all[local_j])),
                "gps_speed": speed,
                "candidate_rank": 0,
            })
        candidates.sort(key=lambda row: row["dist_to_segment"])
        for rank, row in enumerate(candidates, 1):
            row["candidate_rank"] = rank
            rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError(f"No road candidates generated for trip {tid}.")
    return result, track, nodes, arcs


def _get_feature_medians():
    """Cached training-only median approximation for fast demo inference."""
    global _feature_medians

    if _feature_medians is not None:
        return _feature_medians

    cache_path = MODEL_PATH.parent / "feature_medians_demo.json"

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            values = json.load(f)
        _feature_medians = pd.Series(values).reindex(FEATURES).fillna(0.0)
        return _feature_medians

    if not TRAIN_ROOT.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_ROOT}")

    train_dirs = sorted(p for p in TRAIN_ROOT.iterdir() if p.is_dir())[:5]
    if not train_dirs:
        raise FileNotFoundError("No training trips found.")

    frames = []
    for trip_dir in train_dirs:
        frame, _, _, _ = _candidate_table(trip_dir)
        frames.append(frame[FEATURES])

    train_features = pd.concat(frames, ignore_index=True)
    train_features = train_features.replace([np.inf, -np.inf], np.nan)

    _feature_medians = (
        train_features.median(numeric_only=True)
        .reindex(FEATURES)
        .fillna(0.0)
    )

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(_feature_medians.to_dict(), f)

    return _feature_medians


def _haversine_m(lon1, lat1, lon2, lat2):
    radius = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _load_graph(nodes, arcs):
    arc_frame = arcs.copy()
    arc_frame["arc_id"] = np.arange(len(arc_frame), dtype=int)
    node_xy = nodes[["longitude", "latitude"]].to_numpy(float)
    u = arc_frame.from_node_id.to_numpy(int)
    v = arc_frame.to_node_id.to_numpy(int)
    lengths = _haversine_m(node_xy[u, 0], node_xy[u, 1], node_xy[v, 0], node_xy[v, 1])
    arc_frame["length_m"] = lengths
    arc_from = dict(zip(arc_frame.arc_id, arc_frame.from_node_id))
    arc_to = dict(zip(arc_frame.arc_id, arc_frame.to_node_id))
    outgoing = defaultdict(list)
    for row in arc_frame.itertuples(index=False):
        outgoing[int(row.from_node_id)].append((int(row.arc_id), int(row.to_node_id), float(row.length_m)))
    return arc_frame, arc_from, arc_to, outgoing


def _bounded_paths_to_targets(start_node, target_nodes, outgoing):
    targets = set(int(x) for x in target_nodes)
    found = {}
    if int(start_node) in targets:
        found[int(start_node)] = (0.0, [])
        targets.remove(int(start_node))
    queue = deque([(int(start_node), 0.0, [])])
    visited_depth = {int(start_node): 0}
    while queue and targets:
        node, distance, path = queue.popleft()
        depth = len(path)
        if depth >= HMM_MAX_HOPS:
            continue
        for arc_id, nxt, length in outgoing.get(node, []):
            new_path = path + [arc_id]
            new_distance = distance + length
            if nxt in targets:
                found[nxt] = (new_distance, new_path)
                targets.remove(nxt)
                if not targets:
                    break
            new_depth = depth + 1
            if new_depth < visited_depth.get(nxt, 10**9):
                visited_depth[nxt] = new_depth
                queue.append((nxt, new_distance, new_path))
    return found


def _bounded_road_path(start_node, target_node, outgoing):
    if start_node == target_node:
        return 0.0, []
    found = _bounded_paths_to_targets(start_node, [target_node], outgoing)
    return found.get(int(target_node), (None, None))


def _decode_viterbi(candidates, trip_dir, track, nodes, arcs, model):
    arc_frame, arc_from, arc_to, outgoing = _load_graph(nodes, arcs)
    valid = candidates[candidates.arc_id.between(0, len(arc_frame) - 1)].copy()
    if valid.empty:
        raise ValueError("No valid candidates reference arcs in this trip.")

    x = valid[FEATURES].replace([np.inf, -np.inf], np.nan)
    medians = _get_feature_medians()
    if medians is not None:
        x = x.fillna(medians)
    probabilities = model.predict_proba(x)[:, 1]
    valid["emission_prob"] = np.clip(probabilities, HMM_EPS, 1 - HMM_EPS)

    groups = [
        group.nlargest(HMM_TOP_K, "emission_prob").reset_index(drop=True)
        for _, group in valid.groupby("obs_index", sort=True)
    ]
    if not groups:
        raise ValueError("No observation groups available for Viterbi.")

    track_xy = track[["longitude", "latitude"]].to_numpy(float)
    obs_ids = [int(group.obs_index.iloc[0]) for group in groups]
    gps_steps = [0.0]
    for prev_obs, curr_obs in zip(obs_ids[:-1], obs_ids[1:]):
        gps_steps.append(float(_haversine_m(
            track_xy[prev_obs, 0], track_xy[prev_obs, 1],
            track_xy[curr_obs, 0], track_xy[curr_obs, 1],
        )))

    score_history = [np.log(groups[0].emission_prob.to_numpy(float))]
    backpointers = [None]
    for t in range(1, len(groups)):
        prev_group, curr_group = groups[t - 1], groups[t]
        previous_scores = score_history[-1]
        current_emissions = np.log(curr_group.emission_prob.to_numpy(float))
        scores = np.full(len(curr_group), -np.inf)
        pointers = np.zeros(len(curr_group), dtype=int)
        targets = [int(arc_from[int(aid)]) for aid in curr_group.arc_id]
        paths_by_previous = {}
        for i, (_, prev_row) in enumerate(prev_group.iterrows()):
            paths_by_previous[i] = _bounded_paths_to_targets(
                int(arc_to[int(prev_row.arc_id)]), targets, outgoing
            )

        for j, (_, curr_row) in enumerate(curr_group.iterrows()):
            best_score, best_i = -np.inf, 0
            for i, (_, prev_row) in enumerate(prev_group.iterrows()):
                prev_arc, curr_arc = int(prev_row.arc_id), int(curr_row.arc_id)
                if prev_arc == curr_arc:
                    network_mid = 0.0
                else:
                    path_result = paths_by_previous[i].get(int(arc_from[curr_arc]))
                    if path_result is None:
                        transition_logp = -HMM_UNREACHABLE_LOG_PENALTY
                        candidate_score = (
                            previous_scores[i] + current_emissions[j]
                            + HMM_TRANSITION_WEIGHT * transition_logp
                        )
                        if candidate_score > best_score:
                            best_score, best_i = candidate_score, i
                        continue
                    network_mid = path_result[0]

                prev_remaining = max(
                    0.0, (1.0 - float(prev_row.projection_ratio)) * float(prev_row.segment_length)
                )
                curr_prefix = max(
                    0.0, float(curr_row.projection_ratio) * float(curr_row.segment_length)
                )
                road_distance = prev_remaining + float(network_mid) + curr_prefix
                scale = max(HMM_SIGMA_METRES, 0.35 * float(gps_steps[t]) + 25.0)
                transition_logp = -0.5 * ((road_distance - gps_steps[t]) / scale) ** 2
                candidate_score = (
                    previous_scores[i] + current_emissions[j]
                    + HMM_TRANSITION_WEIGHT * transition_logp
                )
                if candidate_score > best_score:
                    best_score, best_i = candidate_score, i
            scores[j], pointers[j] = best_score, best_i
        score_history.append(scores)
        backpointers.append(pointers)

    state = int(np.argmax(score_history[-1]))
    chosen = [state]
    for t in range(len(groups) - 1, 0, -1):
        state = int(backpointers[t][state])
        chosen.append(state)
    chosen.reverse()

    records = []
    for group, state_idx in zip(groups, chosen):
        row = group.iloc[state_idx]
        records.append({
            "obs_index": int(row.obs_index),
            "predicted_arc": int(row.arc_id),
            "emission_prob": float(row.emission_prob),
            "projection_ratio": float(row.projection_ratio),
            "segment_length": float(row.segment_length),
        })
    decoded = pd.DataFrame(records)

    expanded_route = []
    connected = []
    for t in range(1, len(groups)):
        prev_arc = int(groups[t - 1].iloc[chosen[t - 1]].arc_id)
        curr_arc = int(groups[t].iloc[chosen[t]].arc_id)
        if t == 1:
            expanded_route.append(prev_arc)
        if prev_arc == curr_arc:
            connected.append(True)
            continue
        middle_distance, middle_path = _bounded_road_path(
            int(arc_to[prev_arc]), int(arc_from[curr_arc]), outgoing
        )
        reachable = middle_path is not None
        connected.append(reachable)
        if reachable:
            expanded_route.extend(middle_path)
        expanded_route.append(curr_arc)

    if len(groups) == 1:
        expanded_route = [int(groups[0].iloc[chosen[0]].arc_id)]

    decoded.attrs["expanded_route_arc_ids"] = expanded_route
    decoded.attrs["route_continuity"] = float(np.mean(connected)) if connected else 1.0
    decoded.attrs["transition_reachability"] = decoded.attrs["route_continuity"]
    return decoded, arc_frame


def predict_one_trip(trip_dir: Path) -> dict:
    """FastAPI contract: predict one evaluator trip without reading ground truth."""
    started = time.perf_counter()
    trip_dir = Path(trip_dir)
    model = _load_model()
    candidates, track, nodes, arcs = _candidate_table(trip_dir)
    decoded, arc_frame = _decode_viterbi(candidates, trip_dir, track, nodes, arcs, model)

    # Convert decoded arc IDs into a displayable coordinate sequence.
    route_coordinates = []
    for arc_id in decoded.attrs["expanded_route_arc_ids"]:
        if not 0 <= int(arc_id) < len(arc_frame):
            continue
        u = int(arc_frame.iloc[int(arc_id)].from_node_id)
        v = int(arc_frame.iloc[int(arc_id)].to_node_id)
        if not route_coordinates:
            route_coordinates.append([
                float(nodes.iloc[u].latitude), float(nodes.iloc[u].longitude)
            ])
        route_coordinates.append([
            float(nodes.iloc[v].latitude), float(nodes.iloc[v].longitude)
        ])

    elapsed = time.perf_counter() - started
    return {
        "trip_id": trip_dir.name,
        "observations_processed": int(len(decoded)),
        "candidate_rows": int(len(candidates)),
        "predictions": decoded.to_dict(orient="records"),
        "predicted_arc_ids": decoded.predicted_arc.astype(int).tolist(),
        "reconstructed_route_arc_ids": [int(x) for x in decoded.attrs["expanded_route_arc_ids"]],
        "route_coordinates": route_coordinates,
        "mean_emission_confidence": float(decoded.emission_prob.mean()),
        "route_continuity": float(decoded.attrs["route_continuity"]),
        "inference_time_seconds": round(elapsed, 3),
        "decoder": "Tuned Random Forest + HMM/Viterbi",
        "ground_truth_used": False,
    }
