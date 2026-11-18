"""
m3_hmm.tune_decoder
~~~~~~~~~~~~~~~~~~~
Tunes the HMM + Viterbi decoder parameters and the 3-Model Transition Ensemble
weights (w_topo, w_kin, w_emp, emission_weight, cost_hop1, cost_hop2, cost_same_bwd)
using ONLY the 80 training trips (via out-of-fold validation predictions).

Saves:
  - m3_hmm/artifacts/empirical_priors.npy
  - m3_hmm/artifacts/hmm_config.json
  - m3_hmm/outputs/decoder_tuning_trials.csv
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
import optuna

from m1_data.loader import load_trip
from m1_data.network import RoadNetwork

from .transitions import TripGraph, HMMParams, EmpiricalPriors
from .viterbi import prepare_trip_steps, precompute_trip_transitions, viterbi_from_precomputed, reconstruct_connected_route
from .metrics import compute_route_continuity, collapse_consecutive, levenshtein_distance
from .model_bridge import ensure_predictions_and_models, REPO_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_trip_graph_from_partition(partition: str, trip_id: str) -> tuple[TripGraph, tuple[np.ndarray, np.ndarray]]:
    """Loads the road network for (partition, trip_id) and returns (TripGraph, (gps_px, gps_py))."""
    trip = load_trip(partition, str(trip_id))
    net = RoadNetwork(trip)
    px, py = net.proj.to_xy(trip.lon, trip.lat)
    graph = TripGraph(net.arc_from, net.arc_to, net.arc_len)
    return graph, (px, py)


def run_decoder_tuning(n_trials: int = 40, val_trip_limit: int = 20) -> HMMParams:
    """
    1. Fits Technique 3 (EmpiricalPriors) on all 80 training trips.
    2. Precomputes step transition components on validation training trips.
    3. Runs Optuna Bayesian search to maximize composite objective:
         0.55 * Top1_Accuracy_Reliable + 0.25 * Route_Levenshtein_Sim + 0.20 * Route_Continuity
    4. Saves tuned HMMParams and trial logs.
    """
    df_train, _, _ = ensure_predictions_and_models()

    artifacts_dir = REPO_ROOT / "m3_hmm" / "artifacts"
    outputs_dir = REPO_ROOT / "m3_hmm" / "outputs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Fitting Technique 3 (Empirical Transition Matrix) from 80 training trips...")
    t0 = time.time()
    priors = EmpiricalPriors.fit_from_training_df(df_train)
    priors.save(artifacts_dir / "empirical_priors.npy")
    logging.info(f"Fitted EmpiricalPriors in {time.time() - t0:.2f}s.")

    # Select representative training trips (using OOF predictions) for fast decoder tuning
    all_train_trips = sorted(df_train["trip_id"].unique())
    rng = np.random.RandomState(42)
    val_trips = list(rng.choice(all_train_trips, size=min(val_trip_limit, len(all_train_trips)), replace=False))

    logging.info(f"Precomputing road graphs and candidate transition components for {len(val_trips)} training trips...")
    base_params = HMMParams()
    cached_trips = []
    for tid in val_trips:
        df_t = df_train[df_train["trip_id"] == tid]
        graph, gps_xy = build_trip_graph_from_partition("train", str(tid))
        steps = prepare_trip_steps(df_t)
        trans_list = precompute_trip_transitions(steps, graph, base_params, priors, gps_xy=gps_xy)
        label_arcs = np.array([s["label_arc"] for s in steps], dtype=np.int64)
        rel_mask = np.array([s["label_reliable"] == 1 for s in steps], dtype=bool)
        timestamps = np.array([s["time"] for s in steps], dtype=np.float64)
        ref_seq = collapse_consecutive(label_arcs)
        cached_trips.append({
            "trip_id": tid,
            "graph": graph,
            "steps": steps,
            "trans_list": trans_list,
            "label_arcs": label_arcs,
            "rel_mask": rel_mask,
            "timestamps": timestamps,
            "ref_seq": ref_seq,
        })

    def objective(trial: optuna.Trial) -> float:
        em_weight = trial.suggest_float("emission_weight", 0.6, 3.0)
        raw_w1 = trial.suggest_float("raw_w_topo", 0.2, 1.0)
        raw_w2 = trial.suggest_float("raw_w_kin", 0.1, 1.0)
        raw_w3 = trial.suggest_float("raw_w_emp", 0.05, 0.8)
        w_sum = raw_w1 + raw_w2 + raw_w3
        w_topo, w_kin, w_emp = raw_w1 / w_sum, raw_w2 / w_sum, raw_w3 / w_sum

        trial_params = HMMParams(
            emission_weight=em_weight,
            w_topo=w_topo,
            w_kin=w_kin,
            w_emp=w_emp,
        )

        top1_scores = []
        cont_scores = []
        lev_scores = []

        for item in cached_trips:
            pred_arcs, _ = viterbi_from_precomputed(
                item["steps"], item["trans_list"], trial_params, mode="ensemble", use_emission_ensemble=True
            )
            rel = item["rel_mask"]
            top1 = float(np.mean(pred_arcs[rel] == item["label_arcs"][rel])) if rel.any() else float(np.mean(pred_arcs == item["label_arcs"]))
            pred_route = reconstruct_connected_route(
                pred_arcs, item["graph"], timestamps=item["timestamps"], insert_bridges=True
            )
            cont = compute_route_continuity(pred_route, item["graph"])
            ref_seq = item["ref_seq"]
            scale = max(len(ref_seq), len(pred_route), 1)
            lev_sim = 1.0 - (levenshtein_distance(pred_route, ref_seq) / scale)

            top1_scores.append(top1)
            cont_scores.append(cont)
            lev_scores.append(lev_sim)

        mean_top1 = float(np.mean(top1_scores))
        mean_cont = float(np.mean(cont_scores))
        mean_lev = float(np.mean(lev_scores))

        trial.set_user_attr("mean_top1_reliable", mean_top1)
        trial.set_user_attr("mean_route_continuity", mean_cont)
        trial.set_user_attr("mean_levenshtein_sim", mean_lev)
        trial.set_user_attr("w_topo", w_topo)
        trial.set_user_attr("w_kin", w_kin)
        trial.set_user_attr("w_emp", w_emp)

        return 0.55 * mean_top1 + 0.25 * mean_lev + 0.20 * mean_cont

    logging.info(f"Running {n_trials} Optuna Bayesian trials to optimize HMM Ensemble parameters...")
    t_opt = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    logging.info(f"Finished {n_trials} Optuna trials in {time.time() - t_opt:.2f}s. Best composite score: {study.best_value:.4f}")

    best_t = study.best_trial
    best_params = HMMParams(
        emission_weight=float(best_t.params["emission_weight"]),
        w_topo=float(best_t.user_attrs["w_topo"]),
        w_kin=float(best_t.user_attrs["w_kin"]),
        w_emp=float(best_t.user_attrs["w_emp"]),
    )
    best_params.save(artifacts_dir / "hmm_config.json")
    logging.info(
        f"Best HMM Parameters: emission_weight={best_params.emission_weight:.3f}, "
        f"w_topo={best_params.w_topo:.3f}, w_kin={best_params.w_kin:.3f}, w_emp={best_params.w_emp:.3f}"
    )

    # Save trials CSV
    trial_rows = []
    for tr in study.trials:
        if tr.state == optuna.trial.TrialState.COMPLETE:
            trial_rows.append({
                "trial": tr.number,
                "composite_score": tr.value,
                "emission_weight": tr.params["emission_weight"],
                "w_topo": tr.user_attrs["w_topo"],
                "w_kin": tr.user_attrs["w_kin"],
                "w_emp": tr.user_attrs["w_emp"],
                "val_top1_reliable": tr.user_attrs["mean_top1_reliable"],
                "val_route_continuity": tr.user_attrs["mean_route_continuity"],
                "val_levenshtein_sim": tr.user_attrs["mean_levenshtein_sim"],
            })
    pd.DataFrame(trial_rows).to_csv(outputs_dir / "decoder_tuning_trials.csv", index=False)
    return best_params


if __name__ == "__main__":
    run_decoder_tuning()
