"""
m2_models.cross_validation
~~~~~~~~~~~~~~~~~~~~~~~~~~
Executes 5-Fold GroupKFold Cross-Validation on the 80 training trips
for all 6 candidate classification models.

Algorithm Families Tested:
1. Logistic Regression (Linear / Probabilistic baseline)
2. Random Forest (Ensemble / Tree-based bagging)
3. Extra Trees Classifier (Extremely Randomized Trees)
4. Support Vector Machine (Kernel-based RBF classifier)
5. Gradient Boosting (HistGradientBoosting / Sequential boosting)
6. MLP Classifier (Neural network multi-layer perceptron)

Grouping:
- Grouped strictly by `trip_id` using GroupKFold(n_splits=5)
- Ensures all GPS observations and candidate rows from a trip remain
  together in either train or validation, preventing data leakage.
"""

import time
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from m2_models.utils import (
    FEATURE_COLUMNS,
    load_dataset,
    compute_all_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def get_model_definitions():
    """
    Returns the 6 models covering diverse algorithmic families.
    Models requiring feature scaling are flagged with needs_scaling=True.
    """
    return {
        "Logistic Regression": {
            "family": "Linear / Probabilistic",
            "model_fn": lambda: LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
            "needs_scaling": True,
        },
        "Random Forest": {
            "family": "Ensemble (Bagging)",
            "model_fn": lambda: RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                n_jobs=-1,
                random_state=42,
            ),
            "needs_scaling": False,
        },
        "Extra Trees": {
            "family": "Ensemble (Randomized Trees)",
            "model_fn": lambda: ExtraTreesClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                n_jobs=-1,
                random_state=42,
            ),
            "needs_scaling": False,
        },
        "Support Vector Machine": {
            "family": "Kernel-based (RBF)",
            "model_fn": lambda: SVC(
                C=1.0,
                kernel="rbf",
                probability=True,
                random_state=42,
            ),
            "needs_scaling": True,
            "subsample_for_fit": 50000,  # SVC scales quadratically O(N^2); subsample training partition to stay fast
        },
        "Gradient Boosting": {
            "family": "Boosting",
            "model_fn": lambda: HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
            ),
            "needs_scaling": False,
        },
        "MLP Classifier": {
            "family": "Neural Network",
            "model_fn": lambda: MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=200,
                early_stopping=True,
                random_state=42,
            ),
            "needs_scaling": True,
        },
    }


def run_cross_validation(
    data_path: str = "m2_models/data/features_train.parquet",
    output_dir: str = "m2_models/results",
    n_splits: int = 5,
    sample_subsample: float = 1.0,
):
    """
    Runs 5-fold GroupKFold cross-validation across all 6 models.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading training data from {data_path}...")
    df = load_dataset(data_path, filter_reliable=True)
    logging.info(f"Loaded {len(df):,} candidate rows across {df['trip_id'].nunique()} trips.")

    if sample_subsample < 1.0:
        # For rapid validation testing if needed
        trips = df["trip_id"].drop_duplicates().sample(frac=sample_subsample, random_state=42)
        df = df[df["trip_id"].isin(trips)].reset_index(drop=True)
        logging.info(f"Subsampled to {len(df):,} candidate rows across {df['trip_id'].nunique()} trips.")

    X = df[FEATURE_COLUMNS]
    y = df["label"].to_numpy()
    groups = df["trip_id"].to_numpy()

    models = get_model_definitions()
    gkf = GroupKFold(n_splits=n_splits)

    all_fold_results = []
    summary_results = []

    for model_name, config in models.items():
        logging.info(f"\n==================================================")
        logging.info(f"Starting 5-Fold GroupKFold CV: {model_name} ({config['family']})")
        logging.info(f"==================================================")

        fold_metrics = []
        train_times = []
        infer_times = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
            X_train_df = X.iloc[train_idx]
            y_train = y[train_idx]
            
            X_val_df = X.iloc[val_idx]
            y_val = y[val_idx]
            df_val = df.iloc[val_idx]

            # Scaling strictly inside the fold to prevent data leakage
            if config["needs_scaling"]:
                scaler = StandardScaler()
                X_train_fit = scaler.fit_transform(X_train_df)
                X_val_eval = scaler.transform(X_val_df)
            else:
                X_train_fit = X_train_df.to_numpy()
                X_val_eval = X_val_df.to_numpy()

            # Handle SVM training subsample to avoid O(N^2) memory lockup on 1.2M rows
            if "subsample_for_fit" in config and len(X_train_fit) > config["subsample_for_fit"]:
                sub_idx = np.random.RandomState(42).choice(
                    len(X_train_fit), config["subsample_for_fit"], replace=False
                )
                X_train_fit = X_train_fit[sub_idx]
                y_train = y_train[sub_idx]

            model = config["model_fn"]()

            # Train
            t0 = time.time()
            model.fit(X_train_fit, y_train)
            t_train = time.time() - t0
            train_times.append(t_train)

            # Predict probabilities
            t1 = time.time()
            if hasattr(model, "predict_proba"):
                pred_probs = model.predict_proba(X_val_eval)[:, 1]
            elif hasattr(model, "decision_function"):
                dfn = model.decision_function(X_val_eval)
                pred_probs = 1 / (1 + np.exp(-dfn))
            else:
                pred_probs = model.predict(X_val_eval).astype(float)
            t_infer = time.time() - t1
            infer_times.append(t_infer)

            metrics = compute_all_metrics(df_val, y_val, pred_probs)
            metrics["fold"] = fold
            metrics["model"] = model_name
            metrics["family"] = config["family"]
            metrics["train_time_sec"] = t_train
            metrics["infer_time_sec"] = t_infer
            metrics["val_rows"] = len(val_idx)

            fold_metrics.append(metrics)
            all_fold_results.append(metrics)

            logging.info(
                f"Fold {fold}/{n_splits} - "
                f"Top-1 Acc: {metrics['top1_accuracy']:.4f} | "
                f"Macro F1: {metrics['macro_f1']:.4f} | "
                f"Log Loss: {metrics['log_loss']:.4f} | "
                f"Train: {t_train:.1f}s | "
                f"Infer: {t_infer:.2f}s"
            )

        # Average across folds
        df_model_folds = pd.DataFrame(fold_metrics)
        summary = {
            "model": model_name,
            "family": config["family"],
            "cv_top1_accuracy_mean": float(df_model_folds["top1_accuracy"].mean()),
            "cv_top1_accuracy_std": float(df_model_folds["top1_accuracy"].std()),
            "cv_macro_f1_mean": float(df_model_folds["macro_f1"].mean()),
            "cv_macro_f1_std": float(df_model_folds["macro_f1"].std()),
            "cv_precision_mean": float(df_model_folds["precision"].mean()),
            "cv_recall_mean": float(df_model_folds["recall"].mean()),
            "cv_log_loss_mean": float(df_model_folds["log_loss"].mean()),
            "cv_train_time_sec_mean": float(df_model_folds["train_time_sec"].mean()),
            "cv_infer_time_sec_mean": float(df_model_folds["infer_time_sec"].mean()),
        }
        summary_results.append(summary)
        logging.info(
            f"--> {model_name} Summary: "
            f"Mean Top-1: {summary['cv_top1_accuracy_mean']:.4f} (+/- {summary['cv_top1_accuracy_std']:.4f}), "
            f"Mean Macro F1: {summary['cv_macro_f1_mean']:.4f}"
        )

    # Save detailed fold-by-fold results
    df_folds = pd.DataFrame(all_fold_results)
    df_folds.to_csv(output_path / "cv_detailed_folds.csv", index=False)

    # Save summary comparison
    df_summary = pd.DataFrame(summary_results).sort_values(by="cv_top1_accuracy_mean", ascending=False)
    df_summary.to_csv(output_path / "cv_results.csv", index=False)

    logging.info(f"\nSaved cross-validation results to {output_path / 'cv_results.csv'}")
    return df_summary


if __name__ == "__main__":
    run_cross_validation()
