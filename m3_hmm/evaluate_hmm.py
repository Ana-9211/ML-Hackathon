"""
m3_hmm.evaluate_hmm
~~~~~~~~~~~~~~~~~~~
Comprehensive evaluation of Classifier-Only vs. the 3 Individual HMM Transition Models
vs. Output Majority Voting vs. our Proposed Soft Log-Linear Ensemble HMM on BOTH:
  1. Validation Partition (Out-Of-Fold predictions on Training trips)
  2. Student Test Partition (10 student_test trips using M2's official exported .pkl models)

Generates:
  - m3_hmm/outputs/hmm_validation_comparison.csv
  - m3_hmm/outputs/hmm_ablation_comparison.csv
  - m3_hmm/outputs/per_trip_student_test_results.csv
  - m3_hmm/outputs/error_analysis_summary.csv
  - m3_hmm/outputs/plots/hmm_method_comparison.png
  - m3_hmm/outputs/plots/per_trip_accuracy_and_continuity.png
  - m3_hmm/outputs/plots/decoder_tuning_curve.png
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m1_data.loader import load_trip, load_route
from m1_data.network import RoadNetwork

from .transitions import TripGraph, HMMParams, EmpiricalPriors
from .viterbi import decode_trip_all_methods
from .metrics import evaluate_trip_predictions
from .model_bridge import ensure_predictions_and_models, REPO_ROOT
from .tune_decoder import run_decoder_tuning

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

METHOD_ORDER = [
    "Classifier-Only (No HMM)",
    "HMM 1: Topological Step-Cost",
    "HMM 2: Kinematic Distance-Decay",
    "HMM 3: Empirical Transition Matrix",
    "HMM Majority Vote (Output Voting 1+2+3)",
    "HMM Ensemble (3-Transition Soft Average)",
    "HMM Full Ensemble (3-Transition + 3-Classifier)",
]


def _evaluate_partition_trips(
    partition: str,
    df_part: pd.DataFrame,
    trip_ids: list[str],
    params: HMMParams,
    priors: EmpiricalPriors,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_trip_rows = []
    for tid in trip_ids:
        df_t = df_part[df_part["trip_id"] == tid].copy()
        trip = load_trip(partition, str(tid))
        ref_route = load_route(partition, str(tid))
        net = RoadNetwork(trip)
        px, py = net.proj.to_xy(trip.lon, trip.lat)
        graph = TripGraph(net.arc_from, net.arc_to, net.arc_len)

        t0 = time.time()
        decoded_dict = decode_trip_all_methods(df_t, graph, params, priors, gps_xy=(px, py))
        elapsed_all = time.time() - t0
        infer_per_method_ms = (elapsed_all / len(decoded_dict)) * 1000.0

        for method_name, res in decoded_dict.items():
            m = evaluate_trip_predictions(
                df_trip=df_t,
                pred_point_arcs=res["point_arcs"],
                pred_route_arcs=res["route_arcs"],
                graph=graph,
                ref_route=ref_route,
            )
            m["trip_id"] = str(tid)
            m["partition"] = partition
            m["method"] = method_name
            m["infer_time_ms"] = infer_per_method_ms
            per_trip_rows.append(m)

    df_per_trip = pd.DataFrame(per_trip_rows)
    summary_rows = []
    for m_name in METHOD_ORDER:
        sub = df_per_trip[df_per_trip["method"] == m_name]
        w_pts = sub["n_reliable_points"].to_numpy(dtype=np.float64)
        w_all = sub["n_points"].to_numpy(dtype=np.float64)

        summary_rows.append({
            "partition": partition,
            "method": m_name,
            "top1_accuracy_reliable_weighted": float(np.average(sub["top1_accuracy_reliable"], weights=w_pts)),
            "top1_accuracy_reliable_mean": float(sub["top1_accuracy_reliable"].mean()),
            "top1_accuracy_all_weighted": float(np.average(sub["top1_accuracy_all"], weights=w_all)),
            "candidate_macro_f1_mean": float(sub["candidate_macro_f1"].mean()),
            "pointwise_continuity_mean": float(sub["pointwise_continuity"].mean()),
            "route_continuity_mean": float(sub["route_continuity"].mean()),
            "route_levenshtein_sim_mean": float(sub["route_levenshtein_sim"].mean()),
            "route_coverage_mean": float(sub["route_coverage"].mean()),
            "route_f1_mean": float(sub["route_f1"].mean()),
            "infer_time_ms_per_trip": float(sub["infer_time_ms"].mean()),
        })
    return pd.DataFrame(summary_rows), df_per_trip


def run_evaluation() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_val, df_test, _ = ensure_predictions_and_models()

    artifacts_dir = REPO_ROOT / "m3_hmm" / "artifacts"
    outputs_dir = REPO_ROOT / "m3_hmm" / "outputs"
    plots_dir = outputs_dir / "plots"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    config_path = artifacts_dir / "hmm_config.json"
    priors_path = artifacts_dir / "empirical_priors.npy"
    if not config_path.is_file() or not priors_path.is_file():
        params = run_decoder_tuning()
    else:
        params = HMMParams.load(config_path)
    priors = EmpiricalPriors.load(priors_path)

    # 1. Evaluate on Validation Set (20 representative OOF training trips)
    all_train_trips = sorted(df_val["trip_id"].unique())
    rng = np.random.RandomState(42)
    val_trips = sorted(rng.choice(all_train_trips, size=min(20, len(all_train_trips)), replace=False))
    logging.info(f"Evaluating 7 decoding methods on Validation Split ({len(val_trips)} OOF train trips)...")
    df_val_summary, _ = _evaluate_partition_trips("train", df_val, val_trips, params, priors)
    df_val_summary.to_csv(outputs_dir / "hmm_validation_comparison.csv", index=False)

    # 2. Evaluate on Student Test Set (all 10 student_test trips using M2's official .pkl models)
    test_trips = sorted(df_test["trip_id"].unique())
    logging.info(f"Evaluating 7 decoding methods on Student Test Split ({len(test_trips)} student_test trips)...")
    df_test_summary, df_per_trip = _evaluate_partition_trips("student_test", df_test, test_trips, params, priors)
    df_per_trip.to_csv(outputs_dir / "per_trip_student_test_results.csv", index=False)
    df_test_summary.to_csv(outputs_dir / "hmm_ablation_comparison.csv", index=False)

    # 3. Compute detailed per-trip error analysis on student_test
    error_analysis_rows = []
    for tid in test_trips:
        df_t = df_test[df_test["trip_id"] == tid].copy()
        trip = load_trip("student_test", str(tid))
        net = RoadNetwork(trip)
        px, py = net.proj.to_xy(trip.lon, trip.lat)
        graph = TripGraph(net.arc_from, net.arc_to, net.arc_len)
        decoded_dict = decode_trip_all_methods(df_t, graph, params, priors, gps_xy=(px, py))

        df_pts = df_t.drop_duplicates("point_idx").sort_values("point_idx")
        true_arcs = df_pts["label_arc"].to_numpy(dtype=np.int64)
        rel_mask = df_pts["label_reliable"].to_numpy(dtype=np.int8) == 1
        clf_arcs = decoded_dict["Classifier-Only (No HMM)"]["point_arcs"]
        hmm_arcs = decoded_dict["HMM Ensemble (3-Transition Soft Average)"]["point_arcs"]

        clf_ok = clf_arcs == true_arcs
        hmm_ok = hmm_arcs == true_arcs

        sub_t = df_per_trip[df_per_trip["trip_id"] == str(tid)]
        clf_cont = float(sub_t[sub_t["method"] == "Classifier-Only (No HMM)"]["route_continuity"].iloc[0])
        hmm_cont = float(sub_t[sub_t["method"] == "HMM Ensemble (3-Transition Soft Average)"]["route_continuity"].iloc[0])

        fixed_by_hmm = int((~clf_ok & hmm_ok & rel_mask).sum())
        hurt_by_hmm = int((clf_ok & ~hmm_ok & rel_mask).sum())

        error_analysis_rows.append({
            "trip_id": str(tid),
            "total_points": int(len(true_arcs)),
            "reliable_points": int(rel_mask.sum()),
            "both_correct": int((clf_ok & hmm_ok & rel_mask).sum()),
            "fixed_by_hmm": fixed_by_hmm,
            "hurt_by_hmm": hurt_by_hmm,
            "net_points_gained": fixed_by_hmm - hurt_by_hmm,
            "both_wrong": int((~clf_ok & ~hmm_ok & rel_mask).sum()),
            "clf_top1_all": float(np.mean(clf_ok)),
            "hmm_top1_all": float(np.mean(hmm_ok)),
            "clf_top1_reliable": float(np.mean(clf_ok[rel_mask])),
            "hmm_top1_reliable": float(np.mean(hmm_ok[rel_mask])),
            "clf_route_continuity": clf_cont,
            "hmm_route_continuity": hmm_cont,
        })

    df_err = pd.DataFrame(error_analysis_rows)
    df_err.to_csv(outputs_dir / "error_analysis_summary.csv", index=False)

    _generate_plots(df_test_summary, df_err, outputs_dir / "decoder_tuning_trials.csv", plots_dir)
    return df_val_summary, df_test_summary


def _generate_plots(df_summary: pd.DataFrame, df_err: pd.DataFrame, trials_csv: Path, plots_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=160)
    short_names = [
        "Classifier\nOnly",
        "HMM 1:\nTopological",
        "HMM 2:\nKinematic",
        "HMM 3:\nEmpirical",
        "HMM:\nMajority Vote",
        "HMM:\n3-Trans Ens.",
        "HMM:\nFull Ensemble",
    ]
    colors = ["#8c8c8c", "#4c78a8", "#72b7b2", "#f58518", "#e45756", "#54a24b", "#2ca02c"]

    metrics_to_plot = [
        ("top1_accuracy_all_weighted", "Observation Top-1 Accuracy (All Points %)", (80, 96)),
        ("route_continuity_mean", "Reconstructed Route Continuity (%)", (50, 102)),
        ("route_levenshtein_sim_mean", "Sequence Levenshtein Similarity (%)", (40, 96)),
    ]

    for ax, (col, title, _) in zip(axes, metrics_to_plot):
        vals = df_summary[col].to_numpy() * 100.0
        bars = ax.bar(short_names, vals, color=colors, edgecolor="#222", linewidth=0.7)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(max(0, min(vals) - 6), min(102, max(vals) + 4))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", labelsize=8.5)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.4,
                f"{v:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    fig.suptitle("PathSnap M3: Official M2 Models + HMM Ensemble Comparison (Student Test)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "hmm_method_comparison.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160)
    x = np.arange(len(df_err))
    w = 0.36
    trips = df_err["trip_id"].tolist()

    axes[0].bar(x - w / 2, df_err["clf_top1_all"] * 100, width=w, label="M2 Classifier-Only", color="#e45756")
    axes[0].bar(x + w / 2, df_err["hmm_top1_all"] * 100, width=w, label="M2 + M3 HMM Full Ensemble", color="#2ca02c")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(trips, rotation=45, fontsize=8.5)
    axes[0].set_ylabel("Top-1 Accuracy (%)")
    axes[0].set_title("Per-Trip Observation Top-1 Accuracy (Student Test)", fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    axes[1].bar(x - w / 2, df_err["clf_route_continuity"] * 100, width=w, label="M2 Classifier-Only", color="#e45756")
    axes[1].bar(x + w / 2, df_err["hmm_route_continuity"] * 100, width=w, label="M2 + M3 HMM Full Ensemble", color="#2ca02c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(trips, rotation=45, fontsize=8.5)
    axes[1].set_ylabel("Route Continuity (%)")
    axes[1].set_title("Per-Trip Route Network Continuity (Student Test)", fontweight="bold")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(plots_dir / "per_trip_accuracy_and_continuity.png")
    plt.close(fig)

    if trials_csv.is_file():
        df_tr = pd.read_csv(trials_csv)
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
        ax.plot(df_tr["trial"], df_tr["composite_score"] * 100, "o-", alpha=0.65, label="Trial Score (%)", color="#4c78a8")
        ax.plot(df_tr["trial"], df_tr["composite_score"].cummax() * 100, "-", linewidth=2.2, label="Best So Far (%)", color="#d62728")
        ax.set_xlabel("Optuna Bayesian Trial")
        ax.set_ylabel("Composite Validation Objective (%)")
        ax.set_title("HMM Ensemble Decoder Parameter Tuning Progression (Train OOF)", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "decoder_tuning_curve.png")
        plt.close(fig)


if __name__ == "__main__":
    run_evaluation()
