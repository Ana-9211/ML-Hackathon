"""
m3_hmm.visualize_folium
~~~~~~~~~~~~~~~~~~~~~~~
Generates the mandatory 5-Layer Diagnostic Folium HTML maps for Section 6 of the
Technical Report using `m1_data.maps` helpers for consistent styling.

Visualizes:
  1. Road network geometry within 80m of the trajectory
  2. Original GPS observations (.track) with timestamp tooltips
  3. Reference ground-truth route (.route) in solid green
  4. Classifier-Only predicted route in dashed orange
  5. HMM + Viterbi Full Ensemble predicted route in solid red/purple
  6. Diagnostic markers showing exact GPS points where HMM fixed Classifier errors!
"""
from __future__ import annotations

import logging
from pathlib import Path
import folium
import numpy as np

from m1_data.loader import load_trip, load_route
from m1_data.network import RoadNetwork
from m1_data.maps import trip_map, add_route, add_legend, finish

from .transitions import TripGraph, HMMParams, EmpiricalPriors
from .viterbi import decode_trip_all_methods
from .model_bridge import ensure_predictions_and_models, REPO_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def create_trip_comparison_map(
    partition: str,
    trip_id: str,
    df_trip,
    params: HMMParams,
    priors: EmpiricalPriors,
    out_html_path: Path,
) -> Path:
    """Builds and saves the 5-layer interactive Folium map for one trip."""
    trip = load_trip(partition, str(trip_id))
    ref_route = load_route(partition, str(trip_id))
    net = RoadNetwork(trip)
    px, py = net.proj.to_xy(trip.lon, trip.lat)
    graph = TripGraph(net.arc_from, net.arc_to, net.arc_len)

    decoded = decode_trip_all_methods(df_trip, graph, params, priors, gps_xy=(px, py))
    clf_res = decoded["Classifier-Only (No HMM)"]
    hmm_res = decoded["HMM Full Ensemble (3-Transition + 3-Classifier)"]

    routes = []
    if ref_route is not None:
        routes.append((ref_route, "Reference Route (Ground Truth)", "#2ca02c"))

    # Base map with road network, reference route, and GPS observations
    m = trip_map(trip, net, network_radius_m=80.0, routes=routes)

    # Add Classifier-Only route (dashed orange)
    add_route(
        m,
        net,
        clf_res["route_arcs"],
        name="Classifier-Only Route (No HMM)",
        color="#ff7f0e",
        weight=4,
        dash="6, 6",
    )

    # Add HMM + Viterbi Ensemble route (solid crimson)
    add_route(
        m,
        net,
        hmm_res["route_arcs"],
        name="HMM + Viterbi Ensemble Route",
        color="#d62728",
        weight=5,
    )

    # Add diagnostic layer: GPS points where HMM corrected a Classifier-Only mistake
    if "label_arc" in df_trip.columns:
        df_pts = df_trip.drop_duplicates("point_idx").sort_values("point_idx")
        true_arcs = df_pts["label_arc"].to_numpy(dtype=np.int64)
        rel_mask = df_pts["label_reliable"].to_numpy(dtype=np.int8) == 1
        p_idx = df_pts["point_idx"].to_numpy(dtype=np.int64)

        clf_pts = clf_res["point_arcs"]
        hmm_pts = hmm_res["point_arcs"]

        fixed_mask = (clf_pts != true_arcs) & (hmm_pts == true_arcs) & rel_mask
        fixed_indices = p_idx[fixed_mask]

        if len(fixed_indices) > 0:
            fixed_layer = folium.FeatureGroup(
                name=f"Fixed by HMM ({len(fixed_indices)} points corrected)"
            )
            step = max(1, len(fixed_indices) // 150)
            for idx in fixed_indices[::step]:
                folium.CircleMarker(
                    location=(float(trip.lat[idx]), float(trip.lon[idx])),
                    radius=5,
                    color="#6a3d9a",
                    fill=True,
                    fill_color="#b15928",
                    fill_opacity=0.95,
                    weight=1.5,
                    tooltip=f"Point #{idx} (t={trip.t[idx]:.0f}s): HMM corrected Classifier error -> arc {true_arcs[p_idx == idx][0]}",
                ).add_to(fixed_layer)
            fixed_layer.add_to(m)

    add_legend(
        m,
        entries=[
            ("GPS Observations (.track)", "#1f77b4"),
            ("Road Network (.arcs within 80m)", "#9e9e9e"),
            ("Reference Route (.route)", "#2ca02c"),
            ("Classifier-Only Route (Pointwise)", "#ff7f0e"),
            ("HMM + Viterbi Ensemble Route", "#d62728"),
            ("Points Corrected by HMM", "#6a3d9a"),
        ],
        title=f"PathSnap Trip {trip_id} ({partition})",
    )

    out_html_path.parent.mkdir(parents=True, exist_ok=True)
    finish(m, out_html_path)
    return out_html_path


def generate_all_student_test_maps() -> list[Path]:
    _, df_test, _ = ensure_predictions_and_models()
    artifacts_dir = REPO_ROOT / "m3_hmm" / "artifacts"
    maps_dir = REPO_ROOT / "m3_hmm" / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    params = HMMParams.load(artifacts_dir / "hmm_config.json")
    priors = EmpiricalPriors.load(artifacts_dir / "empirical_priors.npy")

    generated = []
    for tid in sorted(df_test["trip_id"].unique()):
        df_t = df_test[df_test["trip_id"] == tid]
        out_path = maps_dir / f"trip_{tid}_hmm_comparison.html"
        create_trip_comparison_map("student_test", str(tid), df_t, params, priors, out_path)
        logging.info(f"Saved interactive Folium map: {out_path}")
        generated.append(out_path)
    return generated


if __name__ == "__main__":
    generate_all_student_test_maps()
