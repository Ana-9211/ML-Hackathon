"""
m2_models.train_models
~~~~~~~~~~~~~~~~~~~~~~
Trains candidate classification models on the full 80 training trips
(filtered by label_reliable == 1) and exports trained model artifacts (.pkl)
along with the fitted preprocessing scaler to `m2_models/models/`.

Supports `--models` (e.g. `python -m m2_models.train_models --models svm`)
to retrain specific models without overwriting existing artifacts.
"""

import argparse
import time
import logging
from pathlib import Path
import joblib
import numpy as np

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
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def get_model_specs():
    return {
        "logistic_regression": {
            "name": "Logistic Regression",
            "model": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
            "needs_scaling": True,
        },
        "random_forest": {
            "name": "Random Forest",
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                n_jobs=-1,
                random_state=42,
            ),
            "needs_scaling": False,
        },
        "extra_trees": {
            "name": "Extra Trees",
            "model": ExtraTreesClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                n_jobs=-1,
                random_state=42,
            ),
            "needs_scaling": False,
        },
        "svm": {
            "name": "Support Vector Machine",
            "model": SVC(
                C=1.0,
                kernel="rbf",
                probability=True,
                random_state=42,
            ),
            "needs_scaling": True,
            "subsample_for_fit": 50000,
        },
        "gradient_boosting": {
            "name": "Gradient Boosting",
            "model": HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
            ),
            "needs_scaling": False,
        },
        "mlp": {
            "name": "MLP Classifier",
            "model": MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=200,
                early_stopping=True,
                random_state=42,
            ),
            "needs_scaling": True,
        },
    }


def train_and_save_all(
    data_path: str = "m2_models/data/features_train.parquet",
    output_dir: str = "m2_models/models",
    selected_models: list[str] | None = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading training partition from {data_path}...")
    df = load_dataset(data_path, filter_reliable=True)
    logging.info(f"Loaded {len(df):,} training candidates across {df['trip_id'].nunique()} trips.")

    X = df[FEATURE_COLUMNS]
    y = df["label"].to_numpy()

    scaler_path = output_path / "scaler.pkl"
    if selected_models and scaler_path.is_file():
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
        logging.info(f"Loaded existing feature scaler from {scaler_path}")
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, scaler_path, compress=3)
        logging.info(f"Fitted and saved global feature scaler to {scaler_path}")

    specs = get_model_specs()
    if selected_models:
        unknown = [m for m in selected_models if m not in specs]
        if unknown:
            raise ValueError(f"Unknown model key(s): {unknown}. Available: {list(specs.keys())}")
        specs = {k: specs[k] for k in selected_models}

    training_summary = {}

    for key, spec in specs.items():
        name = spec["name"]
        model = spec["model"]
        logging.info(f"\nTraining model: {name}...")

        X_fit = X_scaled if spec["needs_scaling"] else X.to_numpy()
        y_fit = y

        if "subsample_for_fit" in spec and len(X_fit) > spec["subsample_for_fit"]:
            sub_idx = np.random.RandomState(42).choice(
                len(X_fit), spec["subsample_for_fit"], replace=False
            )
            X_fit = X_fit[sub_idx]
            y_fit = y_fit[sub_idx]

        t0 = time.time()
        model.fit(X_fit, y_fit)
        train_time = time.time() - t0

        model_file = output_path / f"{key}.pkl"
        joblib.dump(model, model_file, compress=3)
        training_summary[name] = {"train_time_sec": train_time, "saved_path": str(model_file)}
        logging.info(f"Trained {name} in {train_time:.2f}s and saved to {model_file}")

    logging.info("\nSelected models successfully trained and exported!")
    return training_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train M2 candidate classifiers.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional subset of model keys to train (e.g. --models svm).",
    )
    args = parser.parse_args()
    train_and_save_all(selected_models=args.models)
