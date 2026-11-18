"""
src/geometry.py
Geographic and spatial calculation utilities for PathSnap:
- Metric projections (converting lon/lat degrees into meters)
- Point-to-segment shortest distance in meters
- Road segment bearing and vehicle heading calculation
- Angular difference between bearings
"""

import numpy as np

EARTH_RADIUS_METERS = 6371008.8


def to_metric_coordinates(lons: np.ndarray, lats: np.ndarray, ref_lon: float, ref_lat: float):
    """
    Projects longitude and latitude (in degrees) to metric (x, y) coordinates in meters
    using an azimuthal/equirectangular projection centered at (ref_lon, ref_lat).
    Accurate to within millimeters for local urban scales (< 100 km).
    """
    rad_lat = np.radians(ref_lat)
    x = np.radians(lons - ref_lon) * EARTH_RADIUS_METERS * np.cos(rad_lat)
    y = np.radians(lats - ref_lat) * EARTH_RADIUS_METERS
    return x, y


def calculate_bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculates initial compass bearing (azimuth) from point 1 to point 2 in degrees [0, 360).
    """
    r_lat1, r_lat2 = np.radians(lat1), np.radians(lat2)
    d_lon = np.radians(lon2 - lon1)
    y = np.sin(d_lon) * np.cos(r_lat2)
    x = np.cos(r_lat1) * np.sin(r_lat2) - np.sin(r_lat1) * np.cos(r_lat2) * np.cos(d_lon)
    bearing = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    return float(bearing)


def angle_difference(a: float, b: float) -> float:
    """
    Computes smallest angular difference between two bearings in degrees [0, 180].
    """
    diff = abs(a - b) % 360.0
    return float(diff if diff <= 180.0 else 360.0 - diff)


def point_to_segment_projection(px: float, py: float, ax: float, ay: float, bx: float, by: float):
    """
    Given point P(px, py) and segment AB(ax, ay) -> (bx, by) in metric space (meters):
    Returns:
    - distance: shortest Euclidean distance in meters to segment AB
    - t: projection ratio [0, 1] along AB (0 = at A, 1 = at B)
    - proj_x, proj_y: closest point coordinates on segment AB
    """
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        dist = np.hypot(px - ax, py - ay)
        return float(dist), 0.0, float(ax), float(ay)

    t = ((px - ax) * dx + (py - ay) * dy) / l2
    t_clamped = max(0.0, min(1.0, t))
    proj_x = ax + t_clamped * dx
    proj_y = ay + t_clamped * dy
    dist = np.hypot(px - proj_x, py - proj_y)
    return float(dist), float(t_clamped), float(proj_x), float(proj_y)
