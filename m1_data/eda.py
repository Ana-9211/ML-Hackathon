"""Exploratory data analysis for PathSnap (train + student_test only).

Usage (from the repository root):
    python -m m1_data.eda              # needs build_dataset to have been run first

Writes everything to m1_data/outputs/eda/:
    inventory.csv              files, sizes, row counts per trip
    trip_summary.csv           sampling, gaps, extent, network connectivity, route quality
    eda_summary.json           headline numbers used in the report
    *.png                      plots
    map_trip_locations.html    where the trips are in the world
    map_trip_<id>.html         GPS + nearby network + reference route for selected trips
"""
from __future__ import annotations

import json
from pathlib import Path

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import maps
from .geo import LocalProjection, haversine_m
from .labels import route_quality
from .loader import DATA_DIR, list_trips, load_route, load_trip
from .network import RoadNetwork

OUT = Path(__file__).resolve().parent / "outputs" / "eda"
FEATURES = Path(__file__).resolve().parent / "outputs" / "features"
PARTS = ("train", "student_test")
GAP_S = 10.0          # a "gap" = more than 10 s between consecutive fixes
MAP_TRIPS = [("student_test", "00000052"), ("train", "00000010"), ("train", "00000054")]


def file_inventory() -> pd.DataFrame:
    rows = []
    for part in PARTS:
        for tid in list_trips(part):
            row = {"partition": part, "trip_id": tid}
            for ext in ("track", "nodes", "arcs", "route"):
                p = DATA_DIR / part / tid / f"{tid}.{ext}"
                row[f"{ext}_exists"] = p.exists()
                row[f"{ext}_kb"] = round(p.stat().st_size / 1024, 1) if p.exists() else None
                row[f"{ext}_rows"] = sum(1 for _ in open(p)) if p.exists() else None
            rows.append(row)
    return pd.DataFrame(rows)


def trip_summary(part: str, tid: str) -> dict:
    trip = load_trip(part, tid)
    net = RoadNetwork(trip)
    dt = np.diff(trip.t)
    step = haversine_m(trip.lon[:-1], trip.lat[:-1], trip.lon[1:], trip.lat[1:])
    span_x = haversine_m(trip.lon.min(), trip.lat.mean(), trip.lon.max(), trip.lat.mean())
    span_y = haversine_m(trip.lon.mean(), trip.lat.min(), trip.lon.mean(), trip.lat.max())

    n = net.n_nodes
    g = coo_matrix((np.ones(net.n_arcs), (net.arc_from, net.arc_to)), shape=(n, n)).tocsr()
    n_weak, weak_lab = connected_components(g, directed=True, connection="weak")
    n_strong, strong_lab = connected_components(g, directed=True, connection="strong")
    deg = np.bincount(net.arc_from, minlength=n) + np.bincount(net.arc_to, minlength=n)

    s = {
        "partition": part, "trip_id": tid,
        "points": len(trip.t), "duration_min": (trip.t[-1] - trip.t[0]) / 60,
        "dt_median_s": float(np.median(dt)), "share_dt_1s": float(np.mean(dt == 1)),
        "dt_max_s": float(dt.max()), "gaps_over_10s": int((dt > GAP_S).sum()),
        "time_in_gaps_s": float(dt[dt > GAP_S].sum()),
        "max_step_m": float(step.max()), "max_speed_kmh": float((step / np.maximum(dt, 1e-6)).max() * 3.6),
        "lon_center": float(trip.lon.mean()), "lat_center": float(trip.lat.mean()),
        "extent_ew_km": span_x / 1000, "extent_ns_km": span_y / 1000,
        "path_length_km": float(step.sum() / 1000),
        "nodes": n, "arcs": net.n_arcs,
        "arcs_near_track": int(len(net.indexed_arcs)),
        "two_way_share": float(np.mean(net.arc_reverse >= 0)),
        "arc_len_median_m": float(np.median(net.arc_len)),
        "zero_length_arcs": int((net.arc_len < 0.01).sum()),
        "isolated_nodes": int((deg == 0).sum()),
        "weak_components": int(n_weak),
        "largest_weak_share": float(np.bincount(weak_lab).max() / n),
        "strong_components": int(n_strong),
        "largest_strong_share": float(np.bincount(strong_lab).max() / n),
    }
    route = load_route(part, tid)
    s["has_route"] = route is not None
    if route is not None:
        s.update(route_quality(net, route))
        s["route_in_largest_strong"] = float(np.mean(
            strong_lab[net.arc_from[route]] == np.bincount(strong_lab).argmax()))
    return s


def make_plots(summary: pd.DataFrame, feats: pd.DataFrame):
    colors = {"train": "#4c72b0", "student_test": "#dd8452"}

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for part, g in summary.groupby("partition"):
        ax[0].hist(g["points"], bins=30, alpha=0.7, label=part, color=colors[part])
        ax[1].hist(g["duration_min"], bins=30, alpha=0.7, label=part, color=colors[part])
        ax[2].hist(g["path_length_km"], bins=30, alpha=0.7, label=part, color=colors[part])
    for a, t in zip(ax, ["GPS points per trip", "Trip duration (min)", "Driven path length (km)"]):
        a.set_title(t); a.legend()
    fig.tight_layout(); fig.savefig(OUT / "trip_size_distributions.png", dpi=130); plt.close(fig)

    # sampling intervals, pooled over all trips
    dts = []
    for part in PARTS:
        for tid in list_trips(part):
            dts.append(np.diff(load_trip(part, tid).t))
    dts = np.concatenate(dts)
    fig, ax = plt.subplots(figsize=(7, 4))
    vals, counts = np.unique(np.minimum(dts, 60), return_counts=True)
    ax.bar(vals, counts, width=0.8, color="#4c72b0")
    ax.set_yscale("log"); ax.set_xlabel("seconds between consecutive fixes (capped at 60)")
    ax.set_ylabel("count (log)"); ax.set_title(f"Sampling interval: {np.mean(dts == 1):.1%} exactly 1 s")
    fig.tight_layout(); fig.savefig(OUT / "sampling_intervals.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(summary["arcs"], summary["points"], c=summary["partition"].map(colors), s=18)
    ax[0].set_xscale("log"); ax[0].set_xlabel("road arcs in trip network (log)"); ax[0].set_ylabel("GPS points")
    ax[0].set_title("Network size vs trajectory size")
    ax[1].hist(summary["two_way_share"], bins=20, color="#55a868")
    ax[1].set_title("Share of arcs that have a reverse arc (two-way)")
    fig.tight_layout(); fig.savefig(OUT / "network_size.png", dpi=130); plt.close(fig)

    pts = feats.groupby(["trip_id", "point_idx"]).agg(
        n_cand=("n_cand", "first"), label_dist=("label_dist", "first"), has_true=("label", "max"),
        reliable=("label_reliable", "first"), speed=("speed", "first"))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(pts["n_cand"], bins=np.arange(0.5, 13.5), color="#8172b2")
    ax[0].set_title("Candidates per GPS point (train)")
    ax[1].hist(np.minimum(pts["label_dist"], 100), bins=100, color="#c44e52")
    ax[1].set_yscale("log"); ax[1].set_title("GPS point to aligned route arc (m, capped 100)")
    ax[2].hist(np.minimum(pts["speed"] * 3.6, 200), bins=80, color="#4c72b0")
    ax[2].set_title("Speed (km/h, capped 200)")
    fig.tight_layout(); fig.savefig(OUT / "point_level_distributions.png", dpi=130); plt.close(fig)

    # feature distributions: true arc vs other candidates
    rel = feats[feats["label_reliable"] == 1]
    show = ["dist", "dist_gap", "heading_diff", "dist_prev", "dist_next", "overshoot"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for a, col in zip(axes.ravel(), show):
        hi = np.percentile(rel[col], 99)
        for lab, c, name in ((0, "#999999", "other candidate"), (1, "#2ca02c", "true arc")):
            v = rel.loc[rel["label"] == lab, col].clip(upper=hi)
            a.hist(v, bins=60, density=True, alpha=0.6, color=c, label=name)
        a.set_title(col); a.legend()
    fig.suptitle("Feature distributions, true arc vs other candidates (train, reliable labels)")
    fig.tight_layout(); fig.savefig(OUT / "feature_distributions.png", dpi=130); plt.close(fig)

    corr = rel[["dist", "dist_rank", "dist_gap", "dist_ratio", "heading_diff", "heading_cos", "speed",
                "dist_prev", "dist_next", "overshoot", "arc_len", "next_choices", "n_cand", "label"]].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right"); ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im); ax.set_title("Feature correlation (train)")
    fig.tight_layout(); fig.savefig(OUT / "feature_correlation.png", dpi=130); plt.close(fig)


def plot_trajectory(part: str, tid: str):
    trip = load_trip(part, tid)
    net = RoadNetwork(trip)
    route = load_route(part, tid)
    fig, ax = plt.subplots(figsize=(8, 8))
    for a in net.indexed_arcs:
        ax.plot([net.node_lon[net.arc_from[a]], net.node_lon[net.arc_to[a]]],
                [net.node_lat[net.arc_from[a]], net.node_lat[net.arc_to[a]]], color="#cccccc", lw=0.6)
    for a in route:
        ax.plot([net.node_lon[net.arc_from[a]], net.node_lon[net.arc_to[a]]],
                [net.node_lat[net.arc_from[a]], net.node_lat[net.arc_to[a]]], color="#2ca02c", lw=2.5)
    ax.scatter(trip.lon, trip.lat, s=2, color="#1f77b4", zorder=3)
    ax.set_aspect(1 / np.cos(np.radians(trip.lat.mean())))
    ax.set_title(f"Trip {tid} ({part}): GPS (blue), reference route (green), network (grey)")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    fig.tight_layout(); fig.savefig(OUT / f"trajectory_{tid}.png", dpi=130); plt.close(fig)


def folium_maps(summary: pd.DataFrame):
    world = folium.Map(location=[30, 10], zoom_start=2, tiles="OpenStreetMap")
    for part, color in (("train", "blue"), ("student_test", "orange")):
        layer = folium.FeatureGroup(name=part)
        for _, r in summary[summary["partition"] == part].iterrows():
            folium.CircleMarker((r["lat_center"], r["lon_center"]), radius=6, color=color, fill=True,
                                tooltip=f"{part} {r['trip_id']}: {r['points']} pts, {r['path_length_km']:.1f} km").add_to(layer)
        layer.add_to(world)
    maps.finish(world, OUT / "map_trip_locations.html")

    for part, tid in MAP_TRIPS:
        trip = load_trip(part, tid)
        net = RoadNetwork(trip)
        m = maps.trip_map(trip, net, routes=[(load_route(part, tid), "Reference route", maps.LAYER_STYLE["reference"])])
        maps.add_legend(m, [("GPS observations", maps.LAYER_STYLE["gps"]),
                            ("Road network", maps.LAYER_STYLE["network"]),
                            ("Reference route", maps.LAYER_STYLE["reference"])], f"Trip {tid} ({part})")
        maps.finish(m, OUT / f"map_trip_{tid}.html")


def projection_error() -> dict:
    """Projected distance vs haversine between the two most distant-ish points of each trip."""
    errs = []
    for part in PARTS:
        for tid in list_trips(part):
            trip = load_trip(part, tid)
            proj = LocalProjection.for_points(trip.lon, trip.lat)
            x, y = proj.to_xy(trip.lon, trip.lat)
            i, j = np.argmin(x + y), np.argmax(x + y)
            h = haversine_m(trip.lon[i], trip.lat[i], trip.lon[j], trip.lat[j])
            if h > 0:
                errs.append(abs(np.hypot(x[i] - x[j], y[i] - y[j]) - h) / h)
    return {"projection_rel_error_max_pct": 100 * float(np.max(errs)),
            "projection_rel_error_median_pct": 100 * float(np.median(errs))}


def extra_checks(summary: pd.DataFrame, feats: pd.DataFrame) -> dict:
    rel = feats[feats["label_reliable"] == 1]
    n_pts = rel.groupby(["trip_id", "point_idx"]).ngroups
    nearest = rel.sort_values(["trip_id", "point_idx", "dist", "arc_id"]).groupby(["trip_id", "point_idx"]).head(1)
    pts = feats.groupby(["trip_id", "point_idx"]).agg(speed=("speed", "first"), hv=("heading_valid", "first"))
    in_lscc = summary["route_in_largest_strong"]
    out = {
        "extent_ew_km_median": float(summary["extent_ew_km"].median()),
        "extent_ns_km_median": float(summary["extent_ns_km"].median()),
        "extent_km_max": float(summary[["extent_ew_km", "extent_ns_km"]].max().max()),
        "nearest_arc_top1_reliable": float(nearest["label"].mean()),
        "true_arc_is_tied_nearest_reliable": float((rel["label"].eq(1) & rel["dist_rank"].eq(0)).sum() / n_pts),
        "true_arc_two_way_share": float(rel.loc[rel["label"] == 1, "two_way"].mean()),
        "speed_kmh_p10_p50_p90_p99": [float(v) for v in (pts["speed"] * 3.6).quantile([.1, .5, .9, .99])],
        "near_stationary_share": float((pts["speed"] < 1).mean()),
        "heading_invalid_share": float(1 - pts["hv"].mean()),
        "gaps_over_10s_total": int(summary["gaps_over_10s"].sum()),
        "trips_full_route_in_largest_scc": int((in_lscc >= 0.999).sum()),
        "route_in_largest_scc_min": float(in_lscc.min()),
        "trips_with_step_over_200kmh": int((summary["max_speed_kmh"] > 200).sum()),
    }
    out.update(projection_error())
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inv = file_inventory()
    inv.to_csv(OUT / "inventory.csv", index=False)

    summary = pd.DataFrame([trip_summary(p, t) for p in PARTS for t in list_trips(p)])
    summary.to_csv(OUT / "trip_summary.csv", index=False)
    feats = pd.read_parquet(FEATURES / "features_train.parquet")

    make_plots(summary, feats)
    for part, tid in MAP_TRIPS:
        plot_trajectory(part, tid)
    folium_maps(summary)

    pts = feats.groupby(["trip_id", "point_idx"]).agg(has_true=("label", "max"), reliable=("label_reliable", "first"),
                                                      n_cand=("n_cand", "first"))
    headline = {
        "trips": summary.groupby("partition").size().to_dict(),
        "gps_points": summary.groupby("partition")["points"].sum().to_dict(),
        "all_files_present": bool(inv[[c for c in inv if c.endswith("_exists")]].all().all()),
        "invalid_or_duplicate_gps_rows": int(inv["track_rows"].sum() - summary["points"].sum()),
        "share_dt_exactly_1s": float((summary["share_dt_1s"] * (summary["points"] - 1)).sum()
                                     / (summary["points"] - 1).sum()),
        "trips_with_gaps_over_10s": int((summary["gaps_over_10s"] > 0).sum()),
        "max_gap_s": float(summary["dt_max_s"].max()),
        "network_arcs_min_median_max": [int(summary["arcs"].min()), int(summary["arcs"].median()), int(summary["arcs"].max())],
        "two_way_share_median": float(summary["two_way_share"].median()),
        "isolated_nodes_total": int(summary["isolated_nodes"].sum()),
        "routes_with_breaks": int((summary["route_breaks"] > 0).sum()),
        "train_candidate_miss_rate": float(1 - pts["has_true"].mean()),
        "train_candidate_miss_rate_reliable_only": float(1 - pts.loc[pts["reliable"] == 1, "has_true"].mean()),
        "train_unreliable_label_share": float(1 - pts["reliable"].mean()),
        "train_mean_candidates_per_point": float(pts["n_cand"].mean()),
        "train_positive_rate_rows": float(feats["label"].mean()),
    }
    headline.update(extra_checks(summary, feats))
    (OUT / "eda_summary.json").write_text(json.dumps(headline, indent=2))
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
