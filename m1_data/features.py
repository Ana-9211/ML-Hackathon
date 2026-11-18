"""Feature table: one row per (GPS point, candidate arc).

Why these features:
  dist, dist_rank, dist_gap, dist_ratio
      GPS noise is roughly isotropic, so the true road is usually (not
      always) the closest. Relative measures matter because noise level
      differs between trips (open highway vs dense city).
  heading_diff, heading_cos, heading_valid
      Both directions of a two-way street have identical geometry; the
      direction of travel is the only thing that separates them. Heading is
      unreliable when the car is (almost) stationary, hence heading_valid.
  speed
      Fast vehicles are on major roads; also scales how much the heading can
      be trusted.
  dist_prev, dist_next
      Distance of the SAME arc to the GPS points a few seconds before/after.
      A real road stays close over several fixes, a side street crossed at
      an intersection does not.
  overshoot
      How far the point projects beyond the end of the segment; a point
      "past the end" of an arc is more likely on the next arc.
  arc_len, next_choices, two_way
      Road structure: short arcs inside junctions, intersections, one-way.
  n_cand, fallback
      How crowded / ambiguous the neighbourhood is.

All features are computed from the GPS track and road network only; the
label columns are added afterwards and only when a route is supplied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .candidates import generate_candidates
from .geo import angle_diff_deg, bearing_deg, point_to_segment
from .labels import UNRELIABLE_LABEL_M, align_route
from .loader import Trip
from .network import RoadNetwork

HEADING_WINDOW = 2       # heading from point i-2 to i+2
MIN_MOVE_FOR_HEADING_M = 3.0
CONTEXT_OFFSET = 3       # dist_prev / dist_next use points i-3 and i+3

FEATURE_COLUMNS = [
    "dist", "dist_rank", "dist_gap", "dist_ratio",
    "heading_diff", "heading_cos", "heading_valid", "speed",
    "dist_prev", "dist_next", "overshoot",
    "arc_len", "next_choices", "two_way",
    "n_cand", "fallback",
]


def point_motion(px, py, t):
    """Per-point heading (deg), whether it is reliable, and speed (m/s)."""
    n = len(px)
    i = np.arange(n)
    lo = np.clip(i - HEADING_WINDOW, 0, n - 1)
    hi = np.clip(i + HEADING_WINDOW, 0, n - 1)
    dx, dy = px[hi] - px[lo], py[hi] - py[lo]
    dt = np.maximum(t[hi] - t[lo], 1e-6)
    moved = np.hypot(dx, dy)
    heading = bearing_deg(dx, dy)
    valid = moved >= MIN_MOVE_FOR_HEADING_M
    speed = np.where(hi > lo, moved / dt, 0.0)
    return heading, valid, speed


def build_trip_table(trip: Trip, route: np.ndarray | None = None) -> tuple[pd.DataFrame, dict]:
    net = RoadNetwork(trip)
    px, py = net.proj.to_xy(trip.lon, trip.lat)
    t = trip.t
    heading, h_valid, speed = point_motion(px, py, t)

    cand = generate_candidates(net, px, py)
    p = cand["point_idx"].to_numpy()
    a = cand["arc_id"].to_numpy()
    dist = cand["dist"].to_numpy()

    df = pd.DataFrame({"trip_id": trip.trip_id, "point_idx": p, "time": t[p], "arc_id": a,
                       "arc_from": net.arc_from[a], "arc_to": net.arc_to[a],
                       "proj_t": cand["t"].to_numpy()})
    df["dist"] = dist
    grp = df.groupby("point_idx")["dist"]
    df["dist_rank"] = grp.rank(method="dense").to_numpy() - 1
    min_d = grp.transform("min").to_numpy()
    df["dist_gap"] = dist - min_d
    df["dist_ratio"] = (dist + 1.0) / (min_d + 1.0)

    hd = angle_diff_deg(heading[p], net.arc_bearing[a])
    valid = h_valid[p]
    df["heading_diff"] = np.where(valid, hd, 90.0)
    df["heading_cos"] = np.where(valid, np.cos(np.radians(hd)), 0.0)
    df["heading_valid"] = valid.astype(np.int8)
    df["speed"] = speed[p]

    n = len(px)
    for name, off in (("dist_prev", -CONTEXT_OFFSET), ("dist_next", CONTEXT_OFFSET)):
        q = np.clip(p + off, 0, n - 1)
        d, _, _ = point_to_segment(px[q], py[q], net.ax[a], net.ay[a], net.bx[a], net.by[a])
        df[name] = d

    t_raw = cand["t_raw"].to_numpy()
    df["overshoot"] = np.maximum(0.0, np.maximum(-t_raw, t_raw - 1.0)) * net.arc_len[a]
    df["arc_len"] = net.arc_len[a]
    df["next_choices"] = net.arc_next_choices[a]
    df["two_way"] = (net.arc_reverse[a] >= 0).astype(np.int8)
    df["n_cand"] = df.groupby("point_idx")["arc_id"].transform("size").to_numpy()
    df["fallback"] = cand["fallback"].to_numpy()

    stats = {"trip_id": trip.trip_id, "partition": trip.partition, "points": int(n),
             "points_with_candidates": int(len(np.unique(p))), "candidate_rows": int(len(df)),
             "mean_candidates": float(len(df) / max(1, len(np.unique(p)))),
             "fallback_points": int(cand.groupby("point_idx")["fallback"].first().sum()),
             **trip.cleaning}

    if route is not None:
        label_arc, label_dist, _ = align_route(net, px, py, route)
        df["label_arc"] = label_arc[p]
        df["label_dist"] = label_dist[p]
        df["label"] = (a == label_arc[p]).astype(np.int8)
        df["label_reliable"] = (label_dist[p] <= UNRELIABLE_LABEL_M).astype(np.int8)
        hit = np.zeros(n, dtype=bool)
        hit[p[df["label"].to_numpy() == 1]] = True
        stats.update({
            "label_dist_median_m": float(np.median(label_dist)),
            "label_dist_p95_m": float(np.percentile(label_dist, 95)),
            "unreliable_label_points": int((label_dist > UNRELIABLE_LABEL_M).sum()),
            "candidate_miss_rate": float(1 - hit.mean()),
        })
    return df, stats
