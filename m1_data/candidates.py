"""Candidate road segments for every GPS point.

For each point we take all arcs within RADIUS_M, keep the MAX_CANDIDATES
closest, and if nothing is that close we widen the search to
FALLBACK_RADIUS_M so that every usable point still gets some candidates
(these points are flagged with `fallback = 1`).

Both directions of a two-way street are separate arcs with identical
geometry, so they always come in pairs -- distance alone cannot separate
them; heading features do that (see features.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import point_to_segment
from .network import RoadNetwork

RADIUS_M = 50.0
FALLBACK_RADIUS_M = 200.0
MAX_CANDIDATES = 12


def generate_candidates(net: RoadNetwork, px, py, radius=RADIUS_M,
                        fallback_radius=FALLBACK_RADIUS_M, max_candidates=MAX_CANDIDATES) -> pd.DataFrame:
    hits = net.arcs_near_many(px, py, radius)
    lengths = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))

    # widen the search only for points that found nothing
    empty = np.flatnonzero(lengths == 0)
    fallback = np.zeros(len(px), dtype=np.int8)
    if len(empty):
        wide = net.arcs_near_many(px[empty], py[empty], fallback_radius)
        for k, i in enumerate(empty):
            hits[i] = wide[k]
            fallback[i] = len(wide[k]) > 0
        lengths = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))

    point_idx = np.repeat(np.arange(len(px)), lengths)
    arc_id = np.concatenate(hits) if lengths.sum() else np.empty(0, dtype=np.int64)
    dist, t, t_raw = point_to_segment(px[point_idx], py[point_idx], net.ax[arc_id], net.ay[arc_id],
                                      net.bx[arc_id], net.by[arc_id])
    limit = np.where(fallback[point_idx] == 1, fallback_radius, radius)

    df = pd.DataFrame({"point_idx": point_idx, "arc_id": arc_id, "dist": dist, "t": t, "t_raw": t_raw,
                       "fallback": fallback[point_idx]})
    df = df[dist <= limit]
    df = df.sort_values(["point_idx", "dist", "arc_id"], kind="stable")
    df = df.groupby("point_idx", sort=False).head(max_candidates).reset_index(drop=True)
    return df
