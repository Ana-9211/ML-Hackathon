"""
m2_models.evaluate_models
~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluates all trained candidate classification models (6 baseline models +
Optuna-tuned Gradient Boosting) on the held-out 10 Student Testing trips
(`features_student_test.parquet`).

Calculates:
- Observation-level Top-1 Accuracy
- Macro F1-score
- Precision & Recall
- Candidate-level Accuracy
- Log Loss
- Classifier-only Route Continuity (mean over trips)
- Candidate Miss Rate (share of points whose true arc is not among their candidates)
- Inference Latency (total seconds and ms per 1,000 candidates)
- Saves a confusion-matrix PNG per model into `m2_models/plots/`
Exports results to `m2_models/results/student_test_results.csv`.
"""

import time
import logging
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from m2_models.utils import (
    FEATURE_COLUMNS,
    load_dataset,
    compute_all_metrics,
    compute_point_continuity,
    compute_candidate_miss_rate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODELS_TO_EVALUATE = [
    ("Logistic Regression", "logistic_regression.pkl", True, "logistic_regression"),
    ("Random Forest", "random_forest.pkl", False, "random_forest"),
    ("Extra Trees", "extra_trees.pkl", False, "extra_trees"),
    ("Support Vector Machine", "svm.pkl", True, "svm"),
    ("Gradient Boosting", "gradient_boosting.pkl", False, "gradient_boosting"),
    ("MLP Classifier", "mlp.pkl", True, "mlp"),
    ("Tuned Gradient Boosting (Optuna)", "tuned_gradient_boosting.pkl", False, "tuned_gradient_boosting"),
]


def _save_confusion_matrix_png(cm: list[list[int]], model_name: str, out_path: Path) -> None:
    arr = np.asarray(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=180)
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0 (Wrong Road)", "Pred 1 (True Road)"])
    ax.set_yticklabels(["Actual 0 (Wrong Road)", "Actual 1 (True Road)"])
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=11, fontweight="bold", pad=10)

    thresh = arr.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = arr[i, j]
            pct = 100.0 * val / max(arr[i].sum(), 1)
            ax.text(
                j,
                i,
                f"{val:,}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if val > thresh else "#111",
                fontsize=9.5,
                fontweight="bold",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def evaluate_student_test(
    data_path: str = "m2_models/data/features_student_test.parquet",
    models_dir: str = "m2_models/models",
    output_dir: str = "m2_models/results",
    plots_dir: str = "m2_models/plots",
):
    models_path = Path(models_dir)
    output_path = Path(output_dir)
    plots_path = Path(plots_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plots_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading student test dataset from {data_path}...")
    df_test = load_dataset(data_path, filter_reliable=False)  # Evaluate on all test observations
    logging.info(f"Loaded {len(df_test):,} candidate rows across {df_test['trip_id'].nunique()} student testing trips.")

    X_test_raw = df_test[FEATURE_COLUMNS]
    y_test = df_test["label"].to_numpy()
    miss_rate = compute_candidate_miss_rate(df_test)

    scaler_file = models_path / "scaler.pkl"
    scaler = joblib.load(scaler_file) if scaler_file.exists() else None
    X_test_scaled = scaler.transform(X_test_raw) if scaler is not None else None

    eval_results = []

    for name, pkl_name, needs_scaling, key in MODELS_TO_EVALUATE:
        pkl_path = models_path / pkl_name
        if not pkl_path.exists():
            logging.warning(f"Model file {pkl_path} not found. Skipping...")
            continue

        logging.info(f"Evaluating {name} on 10 Student Testing trips...")
        model = joblib.load(pkl_path)
        X_eval = X_test_scaled if needs_scaling else X_test_raw.to_numpy()

        t0 = time.time()
        if hasattr(model, "predict_proba"):
            pred_probs = model.predict_proba(X_eval)[:, 1]
        elif hasattr(model, "decision_function"):
            dfn = model.decision_function(X_eval)
            pred_probs = 1 / (1 + np.exp(-dfn))
        else:
            pred_probs = model.predict(X_eval).astype(float)
        infer_time_sec = time.time() - t0

        latency_per_1k_ms = (infer_time_sec / len(X_eval)) * 1000 * 1000

        metrics = compute_all_metrics(df_test, y_test, pred_probs)
        route_cont = compute_point_continuity(df_test, pred_probs)

        cm_file = plots_path / f"confusion_matrix_{key}.png"
        _save_confusion_matrix_png(metrics["confusion_matrix"], name, cm_file)

        row = {
            "model": name,
            "top1_accuracy": metrics["top1_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "candidate_accuracy": metrics["accuracy"],
            "log_loss": metrics["log_loss"],
            "route_continuity": route_cont,
            "candidate_miss_rate": miss_rate,
            "inference_time_sec": infer_time_sec,
            "latency_ms_per_1k_cand": latency_per_1k_ms,
        }
        eval_results.append(row)
        logging.info(
            f"--> {name}: "
            f"Top-1 Acc = {row['top1_accuracy']:.4f} | "
            f"Macro F1 = {row['macro_f1']:.4f} | "
            f"Log Loss = {row['log_loss']:.4f} | "
            f"Route Cont = {row['route_continuity']:.4f} | "
            f"Miss Rate = {row['candidate_miss_rate']:.4f} | "
            f"Latency = {latency_per_1k_ms:.2f} ms/1k cands"
        )

    df_results = pd.DataFrame(eval_results).sort_values(by="top1_accuracy", ascending=False)
    csv_file = output_path / "student_test_results.csv"
    df_results.to_csv(csv_file, index=False)
    logging.info(f"\nStudent testing evaluation complete! Saved to {csv_file}")
    return df_results


if __name__ == "__main__":
    evaluate_student_test()
