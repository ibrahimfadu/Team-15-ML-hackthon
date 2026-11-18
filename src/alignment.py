import numpy as np
from .projection import get_projection_scale, project_coords, point_to_segment_project

def align_track_to_route(track, nodes, arcs, route):
    """
    Assigns each GPS observation in track to its corresponding ground-truth arc in route.
    Uses monotonic dynamic window search to guarantee continuous vehicle progression.

    Returns:
        matched_arcs: list of ground truth arc_id for each GPS observation (length N)
        arc_distances: list of distances to the matched ground truth arc (meters)
    """
    if route is None or len(route) == 0:
        return None, None

    ref_lon, ref_lat = nodes[0, 0], nodes[0, 1]
    kx, ky = get_projection_scale(ref_lat)

    nodes_m = project_coords(nodes, ref_lon, ref_lat, kx, ky)
    gps_m = project_coords(track[:, :2], ref_lon, ref_lat, kx, ky)

    # Extract start and end coordinates of all route segments
    from_nodes = arcs[route, 0]
    to_nodes = arcs[route, 1]
    u_coords = nodes_m[from_nodes]
    v_coords = nodes_m[to_nodes]

    n_gps = len(gps_m)
    n_route = len(route)
    matched_arcs = np.zeros(n_gps, dtype=np.int64)
    arc_distances = np.zeros(n_gps, dtype=np.float64)

    curr_route_idx = 0

    for i in range(n_gps):
        px, py = gps_m[i]

        # Search window around curr_route_idx: allow slight backtrack (-1) and forward lookahead (+12)
        win_start = max(0, curr_route_idx - 1)
        win_end = min(n_route, curr_route_idx + 12)

        best_dist = 1e9
        best_idx = curr_route_idx

        for r_idx in range(win_start, win_end):
            x1, y1 = u_coords[r_idx]
            x2, y2 = v_coords[r_idx]
            dist, _, _, _, _ = point_to_segment_project(px, py, x1, y1, x2, y2)
            if dist < best_dist:
                best_dist = dist
                best_idx = r_idx

        curr_route_idx = best_idx
        matched_arcs[i] = route[best_idx]
        arc_distances[i] = best_dist

    return matched_arcs, arc_distances
