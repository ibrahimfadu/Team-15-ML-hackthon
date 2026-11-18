import numpy as np

def extract_features_for_trip(track, nodes_m, arcs, candidates_per_point, gps_m, matched_arcs=None):
    """
    Extracts tabular features for each (GPS observation, candidate arc) pair.
    Returns:
        X: (num_candidate_pairs, num_features) float array
        y: (num_candidate_pairs,) int array (if matched_arcs is provided, else None)
        meta: list of (gps_point_idx, candidate_arc_id)
    """
    n_pts = len(track)
    X_list = []
    y_list = []
    meta_list = []

    u_coords = nodes_m[arcs[:, 0]]
    v_coords = nodes_m[arcs[:, 1]]
    arc_vecs = v_coords - u_coords
    arc_lens = np.hypot(arc_vecs[:, 0], arc_vecs[:, 1])

    # Precalculate GPS headings and speeds
    gps_headings = np.zeros((n_pts, 2))
    gps_speeds = np.zeros(n_pts)

    for i in range(n_pts):
        if i > 0:
            dt = max(0.1, track[i, 2] - track[i - 1, 2])
            d_vec = gps_m[i] - gps_m[i - 1]
            dist = np.hypot(d_vec[0], d_vec[1])
            gps_speeds[i] = dist / dt
            if dist > 0.1:
                gps_headings[i] = d_vec / dist
            else:
                gps_headings[i] = gps_headings[i - 1]
        elif n_pts > 1:
            d_vec = gps_m[1] - gps_m[0]
            dist = np.hypot(d_vec[0], d_vec[1])
            if dist > 0.1:
                gps_headings[i] = d_vec / dist

    for i in range(n_pts):
        px, py = gps_m[i]
        cands = candidates_per_point[i]
        n_cands = len(cands)
        if n_cands == 0:
            continue

        true_arc = matched_arcs[i] if matched_arcs is not None else -1

        # Calculate distances for all candidates at this point
        c_u = u_coords[cands]
        c_v = v_coords[cands]
        dx = c_v[:, 0] - c_u[:, 0]
        dy = c_v[:, 1] - c_u[:, 1]
        seg_len_sq = dx * dx + dy * dy
        seg_lens = np.sqrt(np.maximum(seg_len_sq, 1e-6))

        t_raw = ((px - c_u[:, 0]) * dx + (py - c_u[:, 1]) * dy) / np.maximum(seg_len_sq, 1e-6)
        t_clamped = np.clip(t_raw, 0.0, 1.0)

        proj_x = c_u[:, 0] + t_clamped * dx
        proj_y = c_u[:, 1] + t_clamped * dy
        perp_dists = np.hypot(px - proj_x, py - proj_y)

        # Distances to nodes
        dist_u = np.hypot(px - c_u[:, 0], py - c_u[:, 1])
        dist_v = np.hypot(px - c_v[:, 0], py - c_v[:, 1])

        # Heading alignment
        arc_dir_x = dx / seg_lens
        arc_dir_y = dy / seg_lens
        h_x, h_y = gps_headings[i]
        has_heading = (np.hypot(h_x, h_y) > 0.5)

        cos_diff = np.where(has_heading, h_x * arc_dir_x + h_y * arc_dir_y, 1.0)
        cos_diff = np.clip(cos_diff, -1.0, 1.0)
        abs_angle_diff = np.arccos(cos_diff)

        # Relative rank among candidates
        order = np.argsort(perp_dists)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))

        for j in range(n_cands):
            arc_id = cands[j]
            feat = [
                perp_dists[j],                       # 0: perpendicular distance (m)
                t_clamped[j],                        # 1: projection factor [0, 1]
                abs(t_raw[j] - t_clamped[j]),        # 2: clamp overshoot penalty
                dist_u[j],                           # 3: distance to from_node (m)
                dist_v[j],                           # 4: distance to to_node (m)
                seg_lens[j],                         # 5: road segment length (m)
                cos_diff[j],                         # 6: heading alignment cos(Δθ)
                abs_angle_diff[j],                   # 7: heading angular difference (rad)
                gps_speeds[i],                       # 8: vehicle speed (m/s)
                ranks[j],                            # 9: distance rank among point candidates
                float(n_cands)                       # 10: number of candidates
            ]
            X_list.append(feat)
            meta_list.append((i, arc_id))
            if matched_arcs is not None:
                y_list.append(1 if arc_id == true_arc else 0)

    X_arr = np.array(X_list, dtype=np.float32) if len(X_list) > 0 else np.empty((0, 11), dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.int32) if len(y_list) > 0 else None
    return X_arr, y_arr, meta_list

FEATURE_NAMES = [
    "perp_dist",
    "proj_t",
    "clamp_penalty",
    "dist_u",
    "dist_v",
    "arc_len",
    "heading_cos",
    "angle_diff_rad",
    "gps_speed",
    "dist_rank",
    "candidate_count"
]
