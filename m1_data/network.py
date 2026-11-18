"""Road network for one trip: projected geometry, adjacency and a spatial index.

Arc ids are kept exactly as in the raw .arcs file (row index), so anything we
predict can be compared directly with the .route file.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .geo import LocalProjection, bearing_deg
from .loader import Trip

SAMPLE_SPACING_M = 10.0   # densification step for the spatial index
BBOX_MARGIN_M = 400.0     # only index arcs near the trajectory


class RoadNetwork:
    def __init__(self, trip: Trip, proj: LocalProjection | None = None):
        self.proj = proj or LocalProjection.for_points(trip.lon, trip.lat)
        nx, ny = self.proj.to_xy(trip.nodes[:, 0], trip.nodes[:, 1])
        self.node_x, self.node_y = nx, ny
        self.node_lon, self.node_lat = trip.nodes[:, 0], trip.nodes[:, 1]

        self.arc_from = trip.arcs[:, 0]
        self.arc_to = trip.arcs[:, 1]
        self.ax, self.ay = nx[self.arc_from], ny[self.arc_from]
        self.bx, self.by = nx[self.arc_to], ny[self.arc_to]
        dx, dy = self.bx - self.ax, self.by - self.ay
        self.arc_len = np.hypot(dx, dy)
        self.arc_bearing = bearing_deg(dx, dy)
        self.n_nodes = len(nx)
        self.n_arcs = len(self.arc_from)

        # out-degree of the node an arc ends at (= how many ways to continue)
        out_deg = np.bincount(self.arc_from, minlength=self.n_nodes)
        self.arc_next_choices = out_deg[self.arc_to]

        # reverse arc lookup (two-way streets are stored as two arcs)
        key = self.arc_from * self.n_nodes + self.arc_to
        rkey = self.arc_to * self.n_nodes + self.arc_from
        order = np.argsort(key)
        pos = np.searchsorted(key[order], rkey)
        pos = np.clip(pos, 0, self.n_arcs - 1)
        found = key[order][pos] == rkey
        self.arc_reverse = np.where(found, order[pos], -1)

        self._build_index(trip)

    # ---------------------------------------------------------------- index
    def _build_index(self, trip: Trip):
        px, py = self.proj.to_xy(trip.lon, trip.lat)
        lo_x, hi_x = px.min() - BBOX_MARGIN_M, px.max() + BBOX_MARGIN_M
        lo_y, hi_y = py.min() - BBOX_MARGIN_M, py.max() + BBOX_MARGIN_M
        # arc kept if its bounding box overlaps the trajectory bounding box
        near = (
            (np.maximum(self.ax, self.bx) >= lo_x) & (np.minimum(self.ax, self.bx) <= hi_x)
            & (np.maximum(self.ay, self.by) >= lo_y) & (np.minimum(self.ay, self.by) <= hi_y)
        )
        self.indexed_arcs = np.flatnonzero(near)

        a = self.indexed_arcs
        n_samples = np.maximum(1, np.ceil(self.arc_len[a] / SAMPLE_SPACING_M).astype(int)) + 1
        arc_of_sample = np.repeat(a, n_samples)
        # position of every sample along its arc, 0..1
        start = np.repeat(np.cumsum(n_samples) - n_samples, n_samples)
        k = np.arange(len(arc_of_sample)) - start
        frac = k / (np.repeat(n_samples, n_samples) - 1)
        sx = self.ax[arc_of_sample] + frac * (self.bx[arc_of_sample] - self.ax[arc_of_sample])
        sy = self.ay[arc_of_sample] + frac * (self.by[arc_of_sample] - self.ay[arc_of_sample])
        self._sample_arc = arc_of_sample
        self._tree = cKDTree(np.column_stack([sx, sy]))

    def arcs_near(self, x: float, y: float, radius: float) -> np.ndarray:
        """Arc ids that have a sample point within `radius` (+ half spacing)."""
        idx = self._tree.query_ball_point([x, y], radius + SAMPLE_SPACING_M / 2)
        return np.unique(self._sample_arc[idx]) if idx else np.empty(0, dtype=np.int64)

    def arcs_near_many(self, xs, ys, radius: float) -> list[np.ndarray]:
        hits = self._tree.query_ball_point(np.column_stack([xs, ys]), radius + SAMPLE_SPACING_M / 2)
        return [np.unique(self._sample_arc[h]) if h else np.empty(0, dtype=np.int64) for h in hits]

    # ------------------------------------------------------------ geometry
    def arc_lonlat(self, arc_id: int):
        f, t = self.arc_from[arc_id], self.arc_to[arc_id]
        return [(self.node_lat[f], self.node_lon[f]), (self.node_lat[t], self.node_lon[t])]

    def connected(self, a: int, b: int) -> bool:
        """True if b is the same arc as a or directly follows it."""
        return a == b or self.arc_to[a] == self.arc_from[b]
