"""Geometry helpers.

Every trip is in a different city (Moscow, Kyoto, Porto Alegre, ...), so we
cannot use one global projected CRS. For each trip we use a local
equirectangular projection centred on the trip's own GPS points:

    x = R * cos(lat0) * (lon - lon0)      y = R * (lat - lat0)     [radians]

Over the size of one trip (a few km to a few tens of km) the scale error of
this projection is well below 1 %, which is far smaller than GPS noise
(~5-20 m), and it lets us use fast planar point-to-segment maths.
"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self._kx = EARTH_RADIUS_M * np.cos(np.radians(self.lat0)) * np.pi / 180.0
        self._ky = EARTH_RADIUS_M * np.pi / 180.0

    @classmethod
    def for_points(cls, lon: np.ndarray, lat: np.ndarray) -> "LocalProjection":
        return cls(np.mean(lon), np.mean(lat))

    def to_xy(self, lon, lat):
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        return (lon - self.lon0) * self._kx, (lat - self.lat0) * self._ky

    def to_lonlat(self, x, y):
        return np.asarray(x) / self._kx + self.lon0, np.asarray(y) / self._ky + self.lat0


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres (used to sanity-check the projection)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def point_to_segment(px, py, ax, ay, bx, by):
    """Distance from points P to segments AB (all arrays broadcast together).

    Returns (distance_m, t_clamped, t_raw) where t is the position of the
    projection along AB (0 = A, 1 = B). t_raw outside [0, 1] means the GPS
    point lies beyond the end of the segment.
    """
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t_raw = np.where(seg2 > 0, ((px - ax) * dx + (py - ay) * dy) / seg2, 0.0)
    t = np.clip(t_raw, 0.0, 1.0)
    qx, qy = ax + t * dx, ay + t * dy
    return np.hypot(px - qx, py - qy), t, t_raw


def bearing_deg(dx, dy):
    """Compass bearing (0 = north, 90 = east) of a displacement in x/y metres."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def angle_diff_deg(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360.0)
    return np.minimum(d, 360.0 - d)
