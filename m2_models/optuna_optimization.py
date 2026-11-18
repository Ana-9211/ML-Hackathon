"""
m2_models.optuna_optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Bayesian Hyperparameter Optimisation using Optuna for the selected
top model (Gradient Boosting / HistGradientBoostingClassifier).

Process:
1. Define hyperparameter search space (learning_rate, max_iter, max_depth,
   min_samples_leaf, l2_regularization).
2. Optimisation objective: Maximize Mean 5-Fold GroupKFold Macro F1.
3. Run 30 Bayesian trials.
4. Export:
   - Trial logs to `m2_models/results/optuna_trials.csv`
   - Optimal parameters to `m2_models/results/best_parameters.json`
5. Retrain the tuned model on all 80 trips, evaluate on 10 Student Test trips,
   and export `m2_models/models/tuned_gradient_boosting.pkl`.
"""

import json
import time
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier

from m2_models.utils import (
    FEATURE_COLUMNS,
    load_dataset,
    compute_all_metrics,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_optuna_study(
    train_path: str = "m2_models/data/features_train.parquet",
    test_path: str = "m2_models/data/features_student_test.parquet",
    output_results_dir: str = "m2_models/results",
    models_dir: str = "m2_models/models",
    n_trials: int = 30,
):
    results_path = Path(output_results_dir)
    models_path = Path(models_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading training data from {train_path}...")
    df_train = load_dataset(train_path, filter_reliable=True)
    X = df_train[FEATURE_COLUMNS].to_numpy()
    y = df_train["label"].to_numpy()
    groups = df_train["trip_id"].to_numpy()

    # Pre-compute GroupKFold indices
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X, y, groups=groups))

    logging.info(f"Starting Bayesian Hyperparameter Optimisation ({n_trials} trials)...")

    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 0.03, 0.25, log=True)
        max_iter = trial.suggest_int("max_iter", 50, 180, step=20)
        max_depth = trial.suggest_int("max_depth", 6, 18)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 100, step=10)
        l2_regularization = trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True)

        fold_f1s = []

        for train_idx, val_idx in splits:
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]
            df_val = df_train.iloc[val_idx]

            clf = HistGradientBoostingClassifier(
                learning_rate=learning_rate,
                max_iter=max_iter,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization,
                random_state=42,
            )
            clf.fit(X_tr, y_tr)
            pred_probs = clf.predict_proba(X_va)[:, 1]
            metrics = compute_all_metrics(df_val, y_va, pred_probs)
            fold_f1s.append(metrics["macro_f1"])

        mean_f1 = float(np.mean(fold_f1s))
        return mean_f1

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logging.info(f"\nOptuna study complete!")
    logging.info(f"Best Trial: #{study.best_trial.number}")
    logging.info(f"Best CV Macro F1: {study.best_value:.4f}")
    logging.info(f"Best Parameters: {json.dumps(study.best_params, indent=2)}")

    # 1. Save trials history
    trials_df = study.trials_dataframe()
    trials_file = results_path / "optuna_trials.csv"
    trials_df.to_csv(trials_file, index=False)
    logging.info(f"Saved trials log to {trials_file}")

    # 2. Save best parameters
    best_params = study.best_params
    best_params_record = {
        "best_cv_macro_f1": study.best_value,
        "best_trial_number": study.best_trial.number,
        "parameters": best_params,
    }
    params_file = results_path / "best_parameters.json"
    with open(params_file, "w") as f:
        json.dump(best_params_record, f, indent=4)
    logging.info(f"Saved best parameters to {params_file}")

    # 3. Retrain tuned model on all 80 trips
    logging.info("\nRetraining tuned model on all 80 training trips with optimal parameters...")
    tuned_model = HistGradientBoostingClassifier(
        **best_params,
        random_state=42,
    )
    t0 = time.time()
    tuned_model.fit(X, y)
    train_time = time.time() - t0

    model_out = models_path / "tuned_gradient_boosting.pkl"
    joblib.dump(tuned_model, model_out, compress=3)
    logging.info(f"Exported tuned model to {model_out} (training time: {train_time:.2f}s)")

    # 4. Evaluate on Student Test (student_test_results.csv is generated centrally by evaluate_models.py)
    logging.info(f"Evaluating tuned model on Student Test ({test_path})...")
    df_test = load_dataset(test_path, filter_reliable=False)
    X_test = df_test[FEATURE_COLUMNS].to_numpy()
    y_test = df_test["label"].to_numpy()

    t_inf0 = time.time()
    test_probs = tuned_model.predict_proba(X_test)[:, 1]
    inf_time = time.time() - t_inf0
    test_metrics = compute_all_metrics(df_test, y_test, test_probs)

    logging.info(
        f"\n=== Tuned Gradient Boosting on Student Test ===\n"
        f"Top-1 Accuracy: {test_metrics['top1_accuracy']:.4f}\n"
        f"Macro F1:       {test_metrics['macro_f1']:.4f}\n"
        f"Log Loss:       {test_metrics['log_loss']:.4f}\n"
        f"Candidate Acc:  {test_metrics['accuracy']:.4f}\n"
        f"Inference Latency: {(inf_time / len(X_test)) * 1e6:.2f} ms/1k cands"
    )

    return best_params_record


if __name__ == "__main__":
    run_optuna_study(n_trials=30)
