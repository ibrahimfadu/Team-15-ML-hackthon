import numpy as np
from scipy.spatial import cKDTree
from .projection import get_projection_scale, project_coords, point_to_segment_project

class CandidateGenerator:
    """
    Fast spatial candidate generator using KD-Tree indexing on arc midpoints.
    """
    def __init__(self, nodes, arcs, search_radius=45.0):
        self.nodes = nodes
        self.arcs = arcs
        self.search_radius = search_radius

        ref_lon, ref_lat = nodes[0, 0], nodes[0, 1]
        self.kx, self.ky = get_projection_scale(ref_lat)
        self.ref_lon = ref_lon
        self.ref_lat = ref_lat

        self.nodes_m = project_coords(nodes, ref_lon, ref_lat, self.kx, self.ky)
        self.u_coords = self.nodes_m[arcs[:, 0]]
        self.v_coords = self.nodes_m[arcs[:, 1]]

        # Arc vectors, lengths, and midpoints
        self.arc_vecs = self.v_coords - self.u_coords
        self.arc_lens = np.hypot(self.arc_vecs[:, 0], self.arc_vecs[:, 1])
        self.arc_midpoints = 0.5 * (self.u_coords + self.v_coords)

        # Build KD-Tree on midpoints
        self.kdtree = cKDTree(self.arc_midpoints)
        self.max_half_len = np.percentile(self.arc_lens * 0.5, 99.0) if len(self.arc_lens) > 0 else 50.0

    def get_candidates_for_track(self, track):
        """
        Generates candidate arcs for each point in track.
        Returns:
            candidates_per_point: list of arrays of arc_ids for each GPS observation
        """
        gps_m = project_coords(track[:, :2], self.ref_lon, self.ref_lat, self.kx, self.ky)
        n_pts = len(gps_m)

        # Broad search with midpoint tree
        broad_radius = self.search_radius + self.max_half_len
        broad_matches = self.kdtree.query_ball_point(gps_m, r=broad_radius)

        candidates_per_point = []
        for i in range(n_pts):
            px, py = gps_m[i]
            candidate_indices = broad_matches[i]

            if len(candidate_indices) == 0:
                # Fallback to nearest midpoint
                _, nearest_arc = self.kdtree.query([px, py], k=1)
                candidates_per_point.append(np.array([nearest_arc], dtype=np.int64))
                continue

            cand_arr = np.array(candidate_indices, dtype=np.int64)
            u_c = self.u_coords[cand_arr]
            v_c = self.v_coords[cand_arr]

            dx = v_c[:, 0] - u_c[:, 0]
            dy = v_c[:, 1] - u_c[:, 1]
            seg_len_sq = dx * dx + dy * dy

            # Vectorized projection
            seg_len_sq_safe = np.where(seg_len_sq < 1e-6, 1.0, seg_len_sq)
            t_raw = ((px - u_c[:, 0]) * dx + (py - u_c[:, 1]) * dy) / seg_len_sq_safe
            t_clamped = np.clip(t_raw, 0.0, 1.0)

            proj_x = u_c[:, 0] + t_clamped * dx
            proj_y = u_c[:, 1] + t_clamped * dy
            dists = np.hypot(px - proj_x, py - proj_y)

            # Filter candidates within search_radius
            valid_mask = dists <= self.search_radius
            if np.any(valid_mask):
                valid_cands = cand_arr[valid_mask]
            else:
                # Keep top 2 closest
                closest_idx = np.argsort(dists)[:2]
                valid_cands = cand_arr[closest_idx]

            candidates_per_point.append(valid_cands)

        return candidates_per_point, gps_m
