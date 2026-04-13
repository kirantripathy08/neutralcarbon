"""
neutralcarbon/ml/anomaly_detection.py
---------------------------------------
Anomaly detection models for carbon emission fraud detection.

Models implemented:
  1. IsolationForest       — ensemble tree-based (primary)
  2. OneClassSVM           — kernel-based boundary
  3. IQRDetector           — statistical baseline
  4. AutoencoderDetector   — deep learning reconstruction error
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AnomalyResult:
    labels: np.ndarray          # -1 = anomaly, 1 = normal (sklearn convention)
    scores: np.ndarray          # raw anomaly scores (lower = more anomalous)
    flags: np.ndarray           # bool mask: True = anomaly
    model_name: str

    def summary(self, meta: pd.DataFrame) -> pd.DataFrame:
        """Attach meta columns and return flagged records sorted by score."""
        df = meta.copy()
        df["anomaly_score"] = -self.scores          # flip: higher = more anomalous
        df["is_anomaly"]    = self.flags
        df["risk_level"]    = pd.cut(
            df["anomaly_score"],
            bins=[-np.inf, 0.3, 0.6, np.inf],
            labels=["LOW", "MEDIUM", "HIGH"]
        )
        flagged = df[df["is_anomaly"]].sort_values("anomaly_score", ascending=False)
        return flagged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. Isolation Forest
# ---------------------------------------------------------------------------

class IsolationForestDetector:
    """
    Isolation Forest (Liu et al., 2008) — isolates anomalies by randomly
    partitioning the feature space. Anomalous points require fewer splits.

    Tuned for the carbon emissions dataset:
      - contamination: estimated 5% fraud rate in real carbon markets
      - n_estimators : 200 trees for stable scores on small dataset
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200,
                 random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self.name = "IsolationForest"

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> AnomalyResult:
        labels = self.model.predict(X)           # -1 or 1
        scores = self.model.score_samples(X)     # negative avg depth
        flags  = labels == -1
        print(f"[{self.name}] {flags.sum()} anomalies detected out of {len(X)} samples")
        return AnomalyResult(labels, scores, flags, self.name)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)
        print(f"[{self.name}] Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "IsolationForestDetector":
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        obj.name  = "IsolationForest"
        return obj


# ---------------------------------------------------------------------------
# 2. One-Class SVM
# ---------------------------------------------------------------------------

class OneClassSVMDetector:
    """
    One-Class SVM (Schölkopf et al., 1999) — learns a decision boundary
    around normal emission patterns in kernel space.

    Uses RBF kernel; nu controls the upper bound on training errors.
    """

    def __init__(self, nu: float = 0.05, kernel: str = "rbf", gamma: str = "scale"):
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self.name  = "OneClassSVM"

    def fit(self, X: np.ndarray) -> "OneClassSVMDetector":
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> AnomalyResult:
        labels = self.model.predict(X)
        scores = self.model.score_samples(X)
        flags  = labels == -1
        print(f"[{self.name}] {flags.sum()} anomalies detected out of {len(X)} samples")
        return AnomalyResult(labels, scores, flags, self.name)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "OneClassSVMDetector":
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        obj.name  = "OneClassSVM"
        return obj


# ---------------------------------------------------------------------------
# 3. IQR Statistical Baseline
# ---------------------------------------------------------------------------

class IQRDetector:
    """
    Interquartile Range (IQR) statistical detector.
    Flags records where any feature exceeds Q1 - k*IQR or Q3 + k*IQR.
    Simple, interpretable baseline.
    """

    def __init__(self, k: float = 2.5):
        self.k    = k
        self.name = "IQRDetector"
        self.q1   = None
        self.q3   = None
        self.iqr  = None

    def fit(self, X: np.ndarray) -> "IQRDetector":
        self.q1  = np.percentile(X, 25, axis=0)
        self.q3  = np.percentile(X, 75, axis=0)
        self.iqr = self.q3 - self.q1
        return self

    def predict(self, X: np.ndarray) -> AnomalyResult:
        lower = self.q1 - self.k * self.iqr
        upper = self.q3 + self.k * self.iqr
        out_of_bounds = (X < lower) | (X > upper)   # (n, features)
        flags  = out_of_bounds.any(axis=1)
        # Score = max z-score-like deviation across features
        iqr_safe = np.where(self.iqr == 0, 1, self.iqr)
        deviations = np.abs(X - (self.q1 + self.q3) / 2) / iqr_safe
        scores = -deviations.max(axis=1)             # negative for consistency
        labels = np.where(flags, -1, 1)
        print(f"[{self.name}] {flags.sum()} anomalies detected out of {len(X)} samples")
        return AnomalyResult(labels, scores, flags, self.name)


# ---------------------------------------------------------------------------
# 4. Autoencoder (TensorFlow)
# ---------------------------------------------------------------------------

class AutoencoderDetector:
    """
    Autoencoder-based anomaly detector (deep learning).
    Trains to reconstruct normal emission patterns; high reconstruction error
    → anomaly.

    Architecture:
      Input(n) → Dense(64, relu) → Dense(32, relu) → Dense(16, relu) [bottleneck]
               → Dense(32, relu) → Dense(64, relu) → Dense(n, sigmoid)
    """

    def __init__(self, encoding_dim: int = 16, epochs: int = 50,
                 batch_size: int = 32, threshold_percentile: float = 95):
        self.encoding_dim          = encoding_dim
        self.epochs                = epochs
        self.batch_size            = batch_size
        self.threshold_percentile  = threshold_percentile
        self.name                  = "Autoencoder"
        self.model                 = None
        self.threshold             = None

    def _build(self, n_features: int):
        try:
            import tensorflow as tf
            from tensorflow import keras

            inp = keras.Input(shape=(n_features,))
            x = keras.layers.Dense(64, activation="relu")(inp)
            x = keras.layers.Dense(32, activation="relu")(x)
            x = keras.layers.Dense(self.encoding_dim, activation="relu")(x)
            x = keras.layers.Dense(32, activation="relu")(x)
            x = keras.layers.Dense(64, activation="relu")(x)
            out = keras.layers.Dense(n_features, activation="sigmoid")(x)

            self.model = keras.Model(inp, out)
            self.model.compile(optimizer="adam", loss="mse")
        except ImportError:
            raise ImportError(
                "TensorFlow not installed. Run: pip install tensorflow"
            )

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        self._build(X.shape[1])
        # Normalize to [0,1] for sigmoid output
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)

        self.model.fit(
            Xs, Xs,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=0,
        )
        # Determine threshold from training reconstruction errors
        recon  = self.model.predict(Xs, verbose=0)
        errors = np.mean((Xs - recon) ** 2, axis=1)
        self.threshold = np.percentile(errors, self.threshold_percentile)
        print(f"[{self.name}] Threshold set at percentile "
              f"{self.threshold_percentile}: {self.threshold:.6f}")
        return self

    def predict(self, X: np.ndarray) -> AnomalyResult:
        Xs    = self._scaler.transform(X)
        recon = self.model.predict(Xs, verbose=0)
        errors = np.mean((Xs - recon) ** 2, axis=1)
        flags  = errors > self.threshold
        scores = -errors                             # negative → higher = worse
        labels = np.where(flags, -1, 1)
        print(f"[{self.name}] {flags.sum()} anomalies detected out of {len(X)} samples")
        return AnomalyResult(labels, scores, flags, self.name)

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str, threshold: float) -> "AutoencoderDetector":
        import tensorflow as tf
        obj = cls.__new__(cls)
        obj.model     = tf.keras.models.load_model(path)
        obj.threshold = threshold
        obj.name      = "Autoencoder"
        return obj


# ---------------------------------------------------------------------------
# Ensemble: majority vote across all detectors
# ---------------------------------------------------------------------------

def ensemble_predict(
    results: list[AnomalyResult],
    vote_threshold: int = 2,
) -> np.ndarray:
    """
    Returns bool mask where at least `vote_threshold` models agree on anomaly.
    """
    votes = np.stack([r.flags.astype(int) for r in results], axis=1)
    return votes.sum(axis=1) >= vote_threshold


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from data.preprocess import load_raw, clean, get_feature_matrix

    df  = clean(load_raw())
    X, meta, _ = get_feature_matrix(df)

    # Quick smoke test
    iso = IsolationForestDetector(contamination=0.05)
    iso.fit(X)
    result = iso.predict(X)
    flagged = result.summary(meta)
    print(f"\nTop anomalies:\n{flagged.head(10).to_string()}")
