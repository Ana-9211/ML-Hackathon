"""
m2_models.evaluate_models
~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluates all trained candidate classification models on the held-out
10 Student Testing trips (`features_student_test.parquet`).

Calculates:
- Observation-level Top-1 Accuracy
- Macro F1-score
- Precision & Recall
- Candidate-level Accuracy
- Log Loss
- Inference Latency (ms per 1,000 candidates)
Exports results to `m2_models/results/student_test_results.csv`.
"""

import time
import logging
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from m2_models.utils import (
    FEATURE_COLUMNS,
    load_dataset,
    compute_all_metrics,
    compute_point_continuity,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODELS_TO_EVALUATE = [
    ("Logistic Regression", "logistic_regression.pkl", True),
    ("Random Forest", "random_forest.pkl", False),
    ("Extra Trees", "extra_trees.pkl", False),
    ("Support Vector Machine", "svm.pkl", True),
    ("Gradient Boosting", "gradient_boosting.pkl", False),
    ("MLP Classifier", "mlp.pkl", True),
    ("Tuned Gradient Boosting (Optuna)", "tuned_gradient_boosting.pkl", False),
]


def evaluate_student_test(
    data_path: str = "m2_models/data/features_student_test.parquet",
    models_dir: str = "m2_models/models",
    output_dir: str = "m2_models/results",
):
    models_path = Path(models_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading student test dataset from {data_path}...")
    df_test = load_dataset(data_path, filter_reliable=False)  # Evaluate on all test observations
    logging.info(f"Loaded {len(df_test):,} candidate rows across {df_test['trip_id'].nunique()} student testing trips.")

    X_test_raw = df_test[FEATURE_COLUMNS]
    y_test = df_test["label"].to_numpy()

    scaler_file = models_path / "scaler.pkl"
    scaler = joblib.load(scaler_file) if scaler_file.exists() else None
    X_test_scaled = scaler.transform(X_test_raw) if scaler is not None else None

    eval_results = []

    for name, pkl_name, needs_scaling in MODELS_TO_EVALUATE:
        pkl_path = models_path / pkl_name
        if not pkl_path.exists():
            logging.warning(f"Model file {pkl_path} not found. Skipping...")
            continue

        logging.info(f"Evaluating {name} on 10 Student Testing trips...")
        model = joblib.load(pkl_path)
        X_eval = X_test_scaled if needs_scaling else X_test_raw.to_numpy()

        # Measure inference latency
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
        row = {
            "model": name,
            "top1_accuracy": metrics["top1_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "candidate_accuracy": metrics["accuracy"],
            "log_loss": metrics["log_loss"],
            "route_continuity": compute_point_continuity(df_test, pred_probs),
            "inference_time_sec": infer_time_sec,
            "latency_ms_per_1k_cand": latency_per_1k_ms,
        }
        eval_results.append(row)
        logging.info(
            f"--> {name}: "
            f"Top-1 Acc = {row['top1_accuracy']:.4f} | "
            f"Macro F1 = {row['macro_f1']:.4f} | "
            f"Log Loss = {row['log_loss']:.4f} | "
            f"Latency = {latency_per_1k_ms:.2f} ms/1k cands"
        )

    df_results = pd.DataFrame(eval_results).sort_values(by="top1_accuracy", ascending=False)
    csv_file = output_path / "student_test_results.csv"
    df_results.to_csv(csv_file, index=False)
    logging.info(f"\nStudent testing evaluation complete! Saved to {csv_file}")
    return df_results


if __name__ == "__main__":
    evaluate_student_test()
