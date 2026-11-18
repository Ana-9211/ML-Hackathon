"""
m3_hmm.inference
~~~~~~~~~~~~~~~~
Live inference adapter for M4's FastAPI backend (`M4_TEAM_PREDICTOR="m3_hmm.inference:predict_trip"`).

Uses M1's feature builder + M2's final selected classifier (`tuned_gradient_boosting.pkl`,
raw features, no scaling) + M3's tuned 3-Technique Transition Ensemble HMM
(`use_emission_ensemble=False`).
Never accesses or accepts `.route` ground-truth files.
Returns `list[int]`: exactly one predicted `arc_id` per GPS observation.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np

from m1_data.loader import Trip
from m1_data.features import build_trip_table, FEATURE_COLUMNS
from m1_data.network import RoadNetwork

from .transitions import TripGraph, HMMParams, EmpiricalPriors
from .viterbi import viterbi_decode
from .model_bridge import M3_ARTIFACTS_DIR, load_tuned_gb_model


@lru_cache(maxsize=1)
def _load_inference_assets() -> tuple[object, HMMParams, EmpiricalPriors]:
    params = HMMParams.load(M3_ARTIFACTS_DIR / "hmm_config.json")
    priors = EmpiricalPriors.load(M3_ARTIFACTS_DIR / "empirical_priors.npy")
    tuned_gb = load_tuned_gb_model()
    return tuned_gb, params, priors


def _coerce_to_m1_trip(trip_obj) -> Trip:
    """Converts M4's TripInput (or any compatible object) into M1's Trip dataclass."""
    if isinstance(trip_obj, Trip):
        return trip_obj

    track = np.array(
        [[obs.lon, obs.lat, obs.timestamp] for obs in trip_obj.observations],
        dtype=np.float64,
    )
    nodes = np.asarray(trip_obj.nodes, dtype=np.float64)
    arcs = np.asarray(trip_obj.arcs, dtype=np.int64)
    return Trip(
        trip_id=str(getattr(trip_obj, "trip_id", "eval_trip")),
        partition="evaluator_test",
        track=track,
        nodes=nodes,
        arcs=arcs,
    )


def predict_trip(trip_input) -> list[int]:
    """
    Entry point for M4 (`M4_TEAM_PREDICTOR="m3_hmm.inference:predict_trip"`).
    Runs M1 feature extraction -> M2 tuned_gradient_boosting.pkl -> M3 3-Technique Ensemble Viterbi.
    Returns one 0-based `arc_id` for each GPS observation in `trip_input`.
    """
    m1_trip = _coerce_to_m1_trip(trip_input)
    n_obs = len(m1_trip.track)
    if n_obs == 0:
        return []

    df_cand, _ = build_trip_table(m1_trip, route=None)
    tuned_gb, params, priors = _load_inference_assets()

    X_raw = df_cand[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    df_cand["predicted_probability"] = tuned_gb.predict_proba(X_raw)[:, 1].astype(np.float32)

    net = RoadNetwork(m1_trip)
    px, py = net.proj.to_xy(m1_trip.lon, m1_trip.lat)
    graph = TripGraph(net.arc_from, net.arc_to, net.arc_len)

    pred_arcs = viterbi_decode(
        df_cand,
        graph,
        params=params,
        priors=priors,
        mode="ensemble",
        use_emission_ensemble=False,
        gps_xy=(px, py),
    )

    present_pts = df_cand.drop_duplicates("point_idx").sort_values("point_idx")["point_idx"].to_numpy(dtype=np.int64)
    if len(present_pts) == n_obs:
        return [int(a) for a in pred_arcs]

    full_preds = np.zeros(n_obs, dtype=np.int64)
    full_preds[present_pts] = pred_arcs
    last_valid = int(pred_arcs[0]) if len(pred_arcs) else 0
    present_set = set(present_pts.tolist())
    for i in range(n_obs):
        if i in present_set:
            last_valid = int(full_preds[i])
        else:
            full_preds[i] = last_valid
    return [int(a) for a in full_preds]
