"""
m3_hmm.viterbi
~~~~~~~~~~~~~~
Log-space Viterbi decoding, multi-technique comparison (Topological, Kinematic,
Empirical, Majority-Vote, and Soft Log-Linear Ensemble), and shortest-path
bridge insertion for connected route reconstruction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .transitions import (
    TripGraph,
    HMMParams,
    EmpiricalPriors,
    compute_step_transition_components,
    REL_HOP2,
    REL_HOP3,
)


def prepare_trip_steps(df_trip: pd.DataFrame) -> list[dict]:
    """
    Groups a trip's candidate DataFrame by point_idx into an ordered list of step dicts.
    Preserves exact candidate-to-road-segment correspondence.
    """
    df_sorted = df_trip.sort_values(["point_idx", "dist_rank" if "dist_rank" in df_trip.columns else "arc_id"], kind="stable")
    steps: list[dict] = []

    has_prob_ens = all(c in df_sorted.columns for c in ("prob_rf", "prob_hgb", "prob_mlp"))
    for pid, grp in df_sorted.groupby("point_idx", sort=False):
        arcs = grp["arc_id"].to_numpy(dtype=np.int64)
        proj_t = grp["proj_t"].to_numpy(dtype=np.float64)
        t_sec = float(grp["time"].iloc[0])
        spd = float(grp["speed"].iloc[0]) if "speed" in grp.columns else 6.0

        if "predicted_probability" in grp.columns:
            probs = grp["predicted_probability"].to_numpy(dtype=np.float64)
        else:
            # Fallback geometric score if no classifier probability column
            d = grp["dist"].to_numpy(dtype=np.float64)
            probs = np.exp(-0.5 * (d / 15.0) ** 2)

        probs_ens = (
            (
                0.40 * grp["prob_rf"].to_numpy(dtype=np.float64)
                + 0.40 * grp["prob_hgb"].to_numpy(dtype=np.float64)
                + 0.20 * grp["prob_mlp"].to_numpy(dtype=np.float64)
            )
            if has_prob_ens
            else probs
        )

        label_arc = int(grp["label_arc"].iloc[0]) if "label_arc" in grp.columns else -1
        reliable = int(grp["label_reliable"].iloc[0]) if "label_reliable" in grp.columns else 1

        steps.append({
            "point_idx": int(pid),
            "time": t_sec,
            "speed": spd,
            "arcs": arcs,
            "proj_t": proj_t,
            "probs": probs,
            "probs_ens": probs_ens,
            "label_arc": label_arc,
            "label_reliable": reliable,
        })
    return steps


def precompute_trip_transitions(
    steps: list[dict],
    graph: TripGraph,
    params: HMMParams,
    priors: EmpiricalPriors,
    gps_xy: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """
    Precomputes (log_A_topo, log_A_kin, log_A_emp, dt) for every step t = 1..T-1 of a trip.
    Caching these 3 component matrices makes tuning (w_topo, w_kin, w_emp, emission_weight)
    run in < 2 milliseconds per trip!
    """
    n = len(steps)
    trans_list: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    px, py = gps_xy if gps_xy is not None else (None, None)

    for t in range(1, n):
        s_prev = steps[t - 1]
        s_curr = steps[t]
        dt = max(1e-3, s_curr["time"] - s_prev["time"])
        spd = 0.5 * (s_prev["speed"] + s_curr["speed"])

        if px is not None and py is not None:
            p_prev = s_prev["point_idx"]
            p_curr = s_curr["point_idx"]
            gps_step_m = float(np.hypot(px[p_curr] - px[p_prev], py[p_curr] - py[p_prev]))
        else:
            gps_step_m = spd * dt

        log_topo, log_kin, log_emp = compute_step_transition_components(
            graph=graph,
            prev_arcs=s_prev["arcs"],
            prev_proj=s_prev["proj_t"],
            curr_arcs=s_curr["arcs"],
            curr_proj=s_curr["proj_t"],
            dt=dt,
            gps_step_m=gps_step_m,
            speed_mps=spd,
            params=params,
            priors=priors,
        )
        trans_list.append((log_topo, log_kin, log_emp, dt))
    return trans_list


def viterbi_from_precomputed(
    steps: list[dict],
    trans_list: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    params: HMMParams,
    mode: str = "ensemble",
    use_emission_ensemble: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs log-space Viterbi decoding over a trip's steps using the selected transition mode:
      - 'classifier_only': Pointwise argmax of emission probabilities (no HMM)
      - 'topological':     Technique 1 (Graph Step-Cost HMM)
      - 'kinematic':       Technique 2 (Newson & Krumm Distance-Consistency HMM)
      - 'empirical':       Technique 3 (Data-Driven Empirical Transition Matrix HMM)
      - 'ensemble':        Soft Log-Linear Transition Ensemble (w_topo*T1 + w_kin*T2 + w_emp*T3)

    Returns:
        (predicted_arc_per_step[T], selected_candidate_idx_per_step[T])
    """
    n = len(steps)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    prob_key = "probs_ens" if use_emission_ensemble else "probs"

    if mode == "classifier_only":
        idx_seq = np.array([int(np.argmax(s[prob_key])) for s in steps], dtype=np.int64)
        arc_seq = np.array([int(steps[t]["arcs"][idx_seq[t]]) for t in range(n)], dtype=np.int64)
        return arc_seq, idx_seq

    # Helper for normalized log emission at step t
    def _log_emission(step_dict: dict) -> np.ndarray:
        p = np.clip(step_dict[prob_key], params.emission_eps, 1.0)
        p_norm = p / p.sum()
        return params.emission_weight * np.log(p_norm)

    dp = _log_emission(steps[0])
    backpointers: list[np.ndarray] = []

    for t in range(1, n):
        log_topo, log_kin, log_emp, _ = trans_list[t - 1]
        if mode == "topological":
            trans_mat = log_topo
        elif mode == "kinematic":
            trans_mat = log_kin
        elif mode == "empirical":
            trans_mat = log_emp
        else:
            trans_mat = (
                params.w_topo * log_topo
                + params.w_kin * log_kin
                + params.w_emp * log_emp
            )

        # scores[i, j] = dp[i] + trans_mat[i, j]
        scores = dp[:, None] + trans_mat
        best_prev = np.argmax(scores, axis=0)
        best_val = scores[best_prev, np.arange(scores.shape[1])]

        dp = best_val + _log_emission(steps[t])
        backpointers.append(best_prev)

    # Backtrack from T-1 to 0
    idx_seq = np.empty(n, dtype=np.int64)
    idx_seq[-1] = int(np.argmax(dp))
    for t in range(n - 2, -1, -1):
        idx_seq[t] = int(backpointers[t][idx_seq[t + 1]])

    arc_seq = np.array([int(steps[t]["arcs"][idx_seq[t]]) for t in range(n)], dtype=np.int64)
    return arc_seq, idx_seq


def viterbi_decode(
    df_trip: pd.DataFrame,
    graph: TripGraph,
    params: HMMParams | None = None,
    priors: EmpiricalPriors | None = None,
    mode: str = "ensemble",
    use_emission_ensemble: bool = False,
    gps_xy: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """
    End-to-end Viterbi decoder for a single trip DataFrame.
    Returns one predicted arc_id for every GPS observation in df_trip.
    """
    params = params or HMMParams()
    priors = priors or EmpiricalPriors()
    steps = prepare_trip_steps(df_trip)
    trans_list = precompute_trip_transitions(steps, graph, params, priors, gps_xy=gps_xy)
    arc_seq, _ = viterbi_from_precomputed(
        steps, trans_list, params, mode=mode, use_emission_ensemble=use_emission_ensemble
    )
    return arc_seq


def reconstruct_connected_route(
    pred_arcs_per_point: np.ndarray | list[int],
    graph: TripGraph,
    timestamps: np.ndarray | None = None,
    gap_reset_sec: float = 30.0,
    insert_bridges: bool = True,
) -> list[int]:
    """
    Converts the length-T observation-level predicted arc sequence into a deduplicated
    route sequence (matching the .route file format).

    When insert_bridges=True, if two consecutive distinct predicted arcs (a_prev, a_curr)
    are 2 or 3 hops apart in the road network (and not separated by a >30s dropout),
    inserts the shortest-path intermediate connecting arc(s) so the reconstructed route
    maintains road network continuity.
    """
    if len(pred_arcs_per_point) == 0:
        return []

    route: list[int] = [int(pred_arcs_per_point[0])]

    for i in range(1, len(pred_arcs_per_point)):
        a_curr = int(pred_arcs_per_point[i])
        a_prev = route[-1]
        if a_curr == a_prev:
            continue

        if insert_bridges:
            dt = float(timestamps[i] - timestamps[i - 1]) if timestamps is not None else 1.0
            if dt < gap_reset_sec:
                v_prev = int(graph.arc_to[a_prev])
                u_curr = int(graph.arc_from[a_curr])
                if v_prev != u_curr:
                    hop_code, _, bridge_arcs = graph.find_bridge(v_prev, u_curr)
                    if hop_code in (REL_HOP2, REL_HOP3) and bridge_arcs:
                        for b_arc in bridge_arcs:
                            if route[-1] != b_arc:
                                route.append(int(b_arc))

        if route[-1] != a_curr:
            route.append(a_curr)

    return route


def decode_trip_all_methods(
    df_trip: pd.DataFrame,
    graph: TripGraph,
    params: HMMParams,
    priors: EmpiricalPriors,
    gps_xy: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, dict]:
    """
    Runs all 6 decoding variants on a single trip for comprehensive experimental comparison:
      1. Classifier-Only (Single Best M2 Model, No HMM)
      2. HMM Technique 1: Topological Graph Step-Cost
      3. HMM Technique 2: Kinematic Distance-Consistency (Newson & Krumm)
      4. HMM Technique 3: Data-Driven Empirical Transition Matrix
      5. HMM Output Majority Vote (Voting across Techniques 1, 2, 3)
      6. Proposed Soft Ensemble HMM (3-Transition Ensemble + Emission Ensemble)
    """
    steps = prepare_trip_steps(df_trip)
    trans_list = precompute_trip_transitions(steps, graph, params, priors, gps_xy=gps_xy)
    timestamps = np.array([s["time"] for s in steps], dtype=np.float64)

    out: dict[str, dict] = {}
    modes = [
        ("Classifier-Only (No HMM)", "classifier_only", False),
        ("HMM 1: Topological Step-Cost", "topological", False),
        ("HMM 2: Kinematic Distance-Decay", "kinematic", False),
        ("HMM 3: Empirical Transition Matrix", "empirical", False),
        ("HMM Ensemble (3-Transition Soft Average)", "ensemble", False),
        ("HMM Full Ensemble (3-Transition + 3-Classifier)", "ensemble", True),
    ]

    for label, mode, use_em_ens in modes:
        arcs, _ = viterbi_from_precomputed(
            steps, trans_list, params, mode=mode, use_emission_ensemble=use_em_ens
        )
        insert_br = mode != "classifier_only"
        route = reconstruct_connected_route(
            arcs, graph, timestamps=timestamps, gap_reset_sec=params.gap_reset_sec, insert_bridges=insert_br
        )
        out[label] = {
            "point_arcs": arcs,
            "route_arcs": route,
        }

    # Also compute Output Majority Vote across HMM 1, HMM 2, HMM 3
    a1 = out["HMM 1: Topological Step-Cost"]["point_arcs"]
    a2 = out["HMM 2: Kinematic Distance-Decay"]["point_arcs"]
    a3 = out["HMM 3: Empirical Transition Matrix"]["point_arcs"]
    voted = np.where((a1 == a2) | (a1 == a3), a1, np.where(a2 == a3, a2, a1))
    voted_route = reconstruct_connected_route(
        voted, graph, timestamps=timestamps, gap_reset_sec=params.gap_reset_sec, insert_bridges=True
    )
    out["HMM Majority Vote (Output Voting 1+2+3)"] = {
        "point_arcs": voted,
        "route_arcs": voted_route,
    }

    return out
