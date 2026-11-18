import numpy as np

R_EARTH = 6371000.0

def get_projection_scale(lat0_deg):
    """
    Returns (kx, ky) conversion factors from degrees to meters
    using equirectangular projection centered at lat0_deg.
    """
    deg2rad = np.pi / 180.0
    kx = R_EARTH * np.cos(np.radians(lat0_deg)) * deg2rad
    ky = R_EARTH * deg2rad
    return kx, ky

def project_coords(coords_deg, ref_lon, ref_lat, kx, ky):
    """
    Project (lon, lat) array in degrees to (x, y) in meters relative to reference point.
    """
    x = (coords_deg[:, 0] - ref_lon) * kx
    y = (coords_deg[:, 1] - ref_lat) * ky
    return np.column_stack([x, y])

def point_to_segment_project(px, py, x1, y1, x2, y2):
    """
    Calculates orthogonal projection of a point (px, py) onto segment (x1,y1)->(x2,y2).
    Returns:
        perp_dist: Shortest distance to segment in meters
        t_param: Unclamped projection parameter (0 at u, 1 at v)
        dist_u: Distance to start node u
        dist_v: Distance to end node v
        seg_len: Length of segment in meters
    """
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    seg_len = np.sqrt(seg_len_sq)

    dist_u = np.hypot(px - x1, py - y1)
    dist_v = np.hypot(px - x2, py - y2)

    if seg_len < 1e-6:
        return dist_u, 0.0, dist_u, dist_v, 0.0

    t_param = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t_clamped = max(0.0, min(1.0, t_param))

    proj_x = x1 + t_clamped * dx
    proj_y = y1 + t_clamped * dy
    perp_dist = np.hypot(px - proj_x, py - proj_y)

    return perp_dist, t_param, dist_u, dist_v, seg_len
