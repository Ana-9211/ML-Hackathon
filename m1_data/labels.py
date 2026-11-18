"""Per-point labels from the reference route.

The .route file is an ordered list of arc ids with NO timestamps, so it does
not say which GPS point belongs to which arc. We recover that with a
monotonic alignment (dynamic programming, similar to DTW):

    choose j_0 <= j_1 <= ... <= j_{N-1}  (indices into the route)
    minimising  sum_i  min(dist(point_i, route_arc_{j_i}), CAP)

Because the index can only move forward, a point is never matched to a part
of the route the vehicle has already left (this matters for U-turns and
loops where the same road appears twice). Skipping route arcs is allowed
(short arcs can be passed between two GPS fixes).

CAP limits the influence of single outlier fixes. The aligned distance is
kept so that unreliable labels (point far from the whole route) can be
flagged instead of silently trusted.
"""
from __future__ import annotations

import numpy as np

from .geo import point_to_segment
from .network import RoadNetwork

COST_CAP_M = 100.0
UNRELIABLE_LABEL_M = 50.0


def align_route(net: RoadNetwork, px: np.ndarray, py: np.ndarray, route: np.ndarray):
    """Returns (label_arc[N], label_dist_m[N], route_pos[N])."""
    route = np.asarray(route, dtype=np.int64)
    n, m = len(px), len(route)
    ax, ay = net.ax[route], net.ay[route]
    bx, by = net.bx[route], net.by[route]
    ar = np.arange(m)

    back = np.empty((n, m), dtype=np.int32 if m > 32000 else np.int16)
    dist0, _, _ = point_to_segment(px[0], py[0], ax, ay, bx, by)
    cost = np.minimum(dist0, COST_CAP_M)
    back[0] = ar
    for i in range(1, n):
        d, _, _ = point_to_segment(px[i], py[i], ax, ay, bx, by)
        prefix_min = np.minimum.accumulate(cost)
        prefix_arg = np.maximum.accumulate(np.where(cost == prefix_min, ar, 0))
        back[i] = prefix_arg
        cost = np.minimum(d, COST_CAP_M) + prefix_min

    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(cost))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]

    label_arc = route[pos]
    label_dist, _, _ = point_to_segment(px, py, net.ax[label_arc], net.ay[label_arc],
                                        net.bx[label_arc], net.by[label_arc])
    return label_arc, label_dist, pos


def route_quality(net: RoadNetwork, route: np.ndarray) -> dict:
    """How continuous the supplied reference route itself is."""
    route = np.asarray(route)
    if len(route) < 2:
        return {"route_arcs": int(len(route)), "route_breaks": 0}
    joined = net.arc_to[route[:-1]] == net.arc_from[route[1:]]
    same = route[:-1] == route[1:]
    return {
        "route_arcs": int(len(route)),
        "route_unique_arcs": int(len(np.unique(route))),
        "route_breaks": int((~(joined | same)).sum()),
        "route_length_m": float(net.arc_len[route].sum()),
    }
