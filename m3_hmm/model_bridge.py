"""
m3_hmm.model_bridge
~~~~~~~~~~~~~~~~~~~
Bridges M1 (feature tables) and M2's official exported `.pkl` models in `m2_models/models/`
into M3.

Primary emission model:
  - `m2_models/models/tuned_gradient_boosting.pkl` (with best parameters from
    `m2_models/results/best_parameters.json` used for 5-fold GroupKFold OOF probabilities).
Ablation 3-classifier blend (kept only as a comparison row):
  - `random_forest.pkl` + untuned `gradient_boosting.pkl` + `mlp.pkl` (with `scaler.pkl`).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

from m1_data.features import FEATURE_COLUMNS

REPO_ROOT = Path(__file__).resolve().parent.parent
M2_DATA_DIR = REPO_ROOT / "m2_models" / "data"
M2_MODELS_DIR = REPO_ROOT / "m2_models" / "models"
M2_RESULTS_DIR = REPO_ROOT / "m2_models" / "results"
M3_ARTIFACTS_DIR = REPO_ROOT / "m3_hmm" / "artifacts"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_tuned_gb_model():
    """Loads the final selected M2 classifier: `tuned_gradient_boosting.pkl`."""
    p = M2_MODELS_DIR / "tuned_gradient_boosting.pkl"
    if not p.is_file():
        raise FileNotFoundError(f"Final tuned model not found at {p}")
    return joblib.load(p)


def load_m2_models() -> dict:
    """
    Loads M2's official trained models from `m2_models/models/`:
      - tuned_hgb: tuned_gradient_boosting.pkl (final selected classifier)
      - hgb:       gradient_boosting.pkl (untuned HGB for 3-classifier ablation blend)
      - rf:        random_forest.pkl
      - mlp:       mlp.pkl + scaler.pkl
    """
    tuned_hgb_path = M2_MODELS_DIR / "tuned_gradient_boosting.pkl"
    hgb_path = M2_MODELS_DIR / "gradient_boosting.pkl"
    rf_path = M2_MODELS_DIR / "random_forest.pkl"
    mlp_path = M2_MODELS_DIR / "mlp.pkl"
    scaler_path = M2_MODELS_DIR / "scaler.pkl"

    if all(p.is_file() for p in (tuned_hgb_path, hgb_path, rf_path, mlp_path, scaler_path)):
        return {
            "tuned_hgb": joblib.load(tuned_hgb_path),
            "hgb": joblib.load(hgb_path),
            "rf": joblib.load(rf_path),
            "mlp": joblib.load(mlp_path),
            "scaler": joblib.load(scaler_path),
            "feature_columns": FEATURE_COLUMNS,
        }
    raise FileNotFoundError(f"M2 models not found in {M2_MODELS_DIR}")


def predict_m2_probabilities_on_df(df: pd.DataFrame, m2_bundle: dict | None = None) -> pd.DataFrame:
    """
    Applies M2's official exported models to `df`:
      - `predicted_probability`: Final classifier (`tuned_gradient_boosting.pkl`, raw features)
      - `prob_rf`, `prob_hgb` (untuned), `prob_mlp`: Used only for the 3-classifier ablation row.
    """
    if m2_bundle is None:
        m2_bundle = load_m2_models()

    df = df.copy()
    X_raw = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    X_scaled = m2_bundle["scaler"].transform(df[FEATURE_COLUMNS])

    p_tuned_hgb = m2_bundle["tuned_hgb"].predict_proba(X_raw)[:, 1].astype(np.float32)
    p_hgb = m2_bundle["hgb"].predict_proba(X_raw)[:, 1].astype(np.float32)
    p_rf = m2_bundle["rf"].predict_proba(X_raw)[:, 1].astype(np.float32)
    p_mlp = m2_bundle["mlp"].predict_proba(X_scaled)[:, 1].astype(np.float32)

    df["predicted_probability"] = p_tuned_hgb
    df["predicted_label"] = (p_tuned_hgb >= 0.5).astype(np.int8)

    df["prob_rf"] = p_rf
    df["prob_hgb"] = p_hgb
    df["prob_mlp"] = p_mlp
    return df


def ensure_predictions_and_models(force_recompute: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Returns (df_train_oof, df_student_test, m2_bundle).
    Uses `tuned_gradient_boosting.pkl` (and `best_parameters.json` for 5-fold GroupKFold OOF)
    for `predicted_probability`, and keeps `RF + untuned HGB + MLP` for the ablation table row.
    """
    M3_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cached_test = M3_ARTIFACTS_DIR / "candidate_predictions_student_test.parquet"
    cached_val = M3_ARTIFACTS_DIR / "candidate_predictions_val_oof.parquet"

    m2_bundle = load_m2_models()

    if not force_recompute and cached_test.is_file() and cached_val.is_file():
        logging.info("Loading cached OOF validation and M2 student_test predictions from m3_hmm/artifacts/...")
        df_val = pd.read_parquet(cached_val)
        df_test = pd.read_parquet(cached_test)
        return df_val, df_test, m2_bundle

    logging.info("Computing student_test candidate probabilities using M2's official .pkl models...")
    df_test_raw = pd.read_parquet(M2_DATA_DIR / "features_student_test.parquet")
    df_test = predict_m2_probabilities_on_df(df_test_raw, m2_bundle)
    df_test.to_parquet(cached_test, index=False)

    best_params_file = M2_RESULTS_DIR / "best_parameters.json"
    with best_params_file.open("r", encoding="utf-8") as f:
        best_params = json.load(f)["parameters"]

    logging.info("Computing 5-fold GroupKFold OOF probabilities on 80 training trips with tuned HistGradientBoosting...")
    df_train = pd.read_parquet(M2_DATA_DIR / "features_train.parquet")
    gkf = GroupKFold(n_splits=5)
    groups = df_train["trip_id"].to_numpy()
    X_all = df_train[FEATURE_COLUMNS]
    y_all = df_train["label"].to_numpy(dtype=np.int8)
    rel_mask = (df_train["label_reliable"].to_numpy(dtype=np.int8) == 1)

    oof_tuned_hgb = np.zeros(len(df_train), dtype=np.float32)
    oof_hgb = np.zeros(len(df_train), dtype=np.float32)
    oof_rf = np.zeros(len(df_train), dtype=np.float32)
    oof_mlp = np.zeros(len(df_train), dtype=np.float32)

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups=groups), 1):
        tr_rel = tr_idx[rel_mask[tr_idx]]
        X_tr_df = X_all.iloc[tr_rel]
        y_tr = y_all[tr_rel]
        X_va_df = X_all.iloc[val_idx]

        X_tr_np = X_tr_df.to_numpy()
        X_va_np = X_va_df.to_numpy()

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_df)
        X_va_sc = scaler.transform(X_va_df)

        tuned_hgb = HistGradientBoostingClassifier(**best_params, random_state=42)
        hgb = HistGradientBoostingClassifier(max_iter=100, max_depth=10, learning_rate=0.1, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=10, n_jobs=-1, random_state=42)
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True, random_state=42)

        tuned_hgb.fit(X_tr_np, y_tr)
        hgb.fit(X_tr_np, y_tr)
        rf.fit(X_tr_np, y_tr)
        mlp.fit(X_tr_sc, y_tr)

        oof_tuned_hgb[val_idx] = tuned_hgb.predict_proba(X_va_np)[:, 1]
        oof_hgb[val_idx] = hgb.predict_proba(X_va_np)[:, 1]
        oof_rf[val_idx] = rf.predict_proba(X_va_np)[:, 1]
        oof_mlp[val_idx] = mlp.predict_proba(X_va_sc)[:, 1]
        logging.info(f"  Finished OOF Fold {fold}/5")

    df_train["predicted_probability"] = oof_tuned_hgb
    df_train["predicted_label"] = (oof_tuned_hgb >= 0.5).astype(np.int8)
    df_train["prob_hgb"] = oof_hgb
    df_train["prob_rf"] = oof_rf
    df_train["prob_mlp"] = oof_mlp

    keep_cols = [
        "trip_id", "point_idx", "time", "arc_id", "arc_from", "arc_to",
        "proj_t", "dist", "dist_rank", "speed", "arc_len", "fallback",
        "label", "label_arc", "label_reliable",
        "prob_hgb", "prob_rf", "prob_mlp",
        "predicted_probability", "predicted_label",
    ]
    df_val = df_train[keep_cols].copy()
    df_val.to_parquet(cached_val, index=False)
    logging.info(f"Saved compact OOF validation predictions ({cached_val.stat().st_size / 1e6:.1f} MB)")

    return df_val, df_test, m2_bundle
