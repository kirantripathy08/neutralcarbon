"""
neutralcarbon/ml/train.py
--------------------------
Full training pipeline: loads data, fits all models, saves artifacts.
Run: python ml/train.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.preprocess import load_raw, clean, get_feature_matrix, train_test_split_temporal
from ml.anomaly_detection import (
    IsolationForestDetector,
    OneClassSVMDetector,
    IQRDetector,
    AutoencoderDetector,
    ensemble_predict,
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def run_training(use_autoencoder: bool = True) -> dict:
    """
    End-to-end training pipeline. Returns a dict of evaluation metrics.
    """
    print("=" * 60)
    print("NeutralCarbon — ML Training Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── 1. Load & preprocess ────────────────────────────────────────
    df_raw   = load_raw()
    df_clean = clean(df_raw)
    df_train, df_test = train_test_split_temporal(df_clean, test_years=[2016, 2017, 2018])

    X_train, meta_train, scaler = get_feature_matrix(df_train, extra_features=True)
    X_test,  meta_test,  _      = get_feature_matrix(df_test,  extra_features=True)

    # ── 2. Train models ─────────────────────────────────────────────
    models = {
        "IsolationForest": IsolationForestDetector(contamination=0.05),
        "OneClassSVM":     OneClassSVMDetector(nu=0.05),
        "IQR":             IQRDetector(k=2.5),
    }
    if use_autoencoder:
        models["Autoencoder"] = AutoencoderDetector(
            encoding_dim=16, epochs=60, batch_size=32
        )

    fitted = {}
    for name, model in models.items():
        print(f"\n── Training: {name} ──")
        model.fit(X_train)
        fitted[name] = model

    # ── 3. Predict on test set ──────────────────────────────────────
    results = {}
    summaries = {}
    for name, model in fitted.items():
        result = model.predict(X_test)
        results[name]   = result
        summaries[name] = result.summary(meta_test)

    # Ensemble
    ensemble_flags = ensemble_predict(list(results.values()), vote_threshold=2)
    print(f"\n[Ensemble] {ensemble_flags.sum()} anomalies (majority vote ≥ 2 models)")

    # ── 4. Save models ──────────────────────────────────────────────
    fitted["IsolationForest"].save(
        os.path.join(ARTIFACTS_DIR, "isolation_forest.joblib")
    )
    fitted["OneClassSVM"].save(
        os.path.join(ARTIFACTS_DIR, "one_class_svm.joblib")
    )

    # ── 5. Save flagged records ─────────────────────────────────────
    # Combine all flagged records across models
    all_flagged = pd.concat(
        [s.assign(model=name) for name, s in summaries.items()]
    ).drop_duplicates(subset=["Year", "Country.Name"]).sort_values(
        "anomaly_score", ascending=False
    )
    out_path = os.path.join(ARTIFACTS_DIR, "flagged_anomalies.csv")
    all_flagged.to_csv(out_path, index=False)
    print(f"\n[save] Flagged anomalies → {out_path}")

    # ── 6. Metrics summary ──────────────────────────────────────────
    metrics = {}
    for name, result in results.items():
        n_total   = len(result.flags)
        n_flagged = result.flags.sum()
        metrics[name] = {
            "n_total":    int(n_total),
            "n_flagged":  int(n_flagged),
            "flag_rate":  round(n_flagged / n_total * 100, 2),
        }

    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[save] Metrics → {metrics_path}")

    print("\n" + "=" * 60)
    print("Training complete.")
    for name, m in metrics.items():
        print(f"  {name:20s} → {m['n_flagged']:3d} anomalies flagged "
              f"({m['flag_rate']:.1f}%)")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    run_training(use_autoencoder=False)   # set True if TF is installed
