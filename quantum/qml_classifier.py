"""
neutralcarbon/quantum/qml_classifier.py
-----------------------------------------
Quantum Machine Learning classifier pipeline for NeutralCarbon.

Workflow:
  1. Reduce 9 emission features → 4 via PCA
  2. Encode each sample into a 4-qubit quantum circuit
  3. Measure expectation values as quantum feature map
  4. Feed quantum features into a classical SVM or threshold classifier
  5. Flag records exceeding anomaly threshold

Run: python quantum/qml_classifier.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC, OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.preprocess import load_raw, clean, get_feature_matrix, train_test_split_temporal
from quantum.quantum_circuit import SimulatedQuantumClassifier


class QMLAnomalyDetector:
    """
    Quantum-enhanced anomaly detector.

    Pipeline:
      StandardScaler → PCA(4) → QuantumFeatureMap → OneClassSVM
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        contamination: float = 0.05,
        seed: int = 42,
    ):
        self.n_qubits      = n_qubits
        self.n_layers      = n_layers
        self.contamination = contamination
        self.seed          = seed

        self.pca    = PCA(n_components=n_qubits, random_state=seed)
        self.scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
        self.qc     = SimulatedQuantumClassifier(n_qubits, n_layers, seed)
        self.clf    = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")

        self._fitted = False

    def _quantum_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """
        Map each row through the quantum circuit.
        Returns quantum feature matrix of shape (n_samples, n_qubits).
        """
        print(f"  [QML] Encoding {len(X_pca)} samples through quantum circuit…")
        q_features = []
        for i, x in enumerate(X_pca):
            ev = self.qc.encode_and_measure(x)
            q_features.append(ev)
            if (i + 1) % 100 == 0:
                print(f"  [QML] {i+1}/{len(X_pca)} encoded")
        return np.array(q_features)

    def fit(self, X: np.ndarray) -> "QMLAnomalyDetector":
        """Fit PCA → quantum encoding → OneClassSVM on normal training data."""
        print("[QML] Fitting PCA…")
        X_pca = self.pca.fit_transform(X)
        print(f"  Explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")

        X_scaled = self.scaler.fit_transform(X_pca)
        X_q      = self._quantum_transform(X_scaled)

        print("[QML] Fitting OneClassSVM on quantum features…")
        self.clf.fit(X_q)
        self._fitted = True
        print("[QML] Fit complete.")
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            flags  — bool array, True = anomaly
            scores — raw anomaly scores (higher = more anomalous)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        X_pca    = self.pca.transform(X)
        X_scaled = self.scaler.transform(X_pca)
        X_q      = self._quantum_transform(X_scaled)

        labels = self.clf.predict(X_q)       # -1 = anomaly, 1 = normal
        scores = -self.clf.score_samples(X_q)  # higher = more anomalous
        flags  = labels == -1

        print(f"[QML] {flags.sum()} anomalies detected out of {len(X)} samples")
        return flags, scores

    def anomaly_report(self, flags: np.ndarray, scores: np.ndarray,
                       meta: pd.DataFrame) -> pd.DataFrame:
        """Build a human-readable anomaly report."""
        df = meta.copy()
        df["qml_score"]     = np.round(scores, 4)
        df["is_anomaly"]    = flags
        df["risk_level"]    = pd.cut(
            df["qml_score"],
            bins=[-np.inf, 0.3, 0.6, np.inf],
            labels=["LOW", "MEDIUM", "HIGH"],
        )
        flagged = df[df["is_anomaly"]].sort_values("qml_score", ascending=False)
        return flagged.reset_index(drop=True)


def benchmark_qml_vs_classical(X_train: np.ndarray, X_test: np.ndarray,
                                meta_test: pd.DataFrame) -> dict:
    """
    Compare QML accuracy vs classical Isolation Forest.
    Uses simulated ground truth labels.
    """
    from ml.anomaly_detection import IsolationForestDetector

    # Simulated ground truth (same as evaluate.py)
    KNOWN = {(2018, "China"), (2017, "India"), (2015, "United States"),
             (2016, "Russia"), (2014, "Iran")}
    y_true = np.array([(r["Year"], r["Country.Name"]) in KNOWN
                       for _, r in meta_test.iterrows()])

    results = {}

    # Classical
    clf_classical = IsolationForestDetector(contamination=0.05)
    clf_classical.fit(X_train)
    res_c = clf_classical.predict(X_test)
    from ml.evaluate import precision_recall_f1
    results["IsolationForest"] = precision_recall_f1(y_true, res_c.flags)

    # QML (on a subset for speed in demo)
    subset = min(300, len(X_test))
    qml = QMLAnomalyDetector(n_qubits=4, n_layers=3)
    qml.fit(X_train[:500])
    flags_q, _ = qml.predict(X_test[:subset])
    results["QML"] = precision_recall_f1(y_true[:subset], flags_q)

    print("\n── QML vs Classical Benchmark ──────────────────────────")
    print(f"{'Model':20s} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    print("─" * 50)
    for name, m in results.items():
        print(f"{name:20s} {m['accuracy']:7.3f} {m['precision']:7.3f} "
              f"{m['recall']:7.3f} {m['f1']:7.3f}")

    return results


def demo_circuit_on_emissions():
    """Quick demo: encode top emitters and show quantum scores."""
    TOP_EMITTERS = {
        "China_2018": np.array([7215.3, 270.4, 982.1, 831.4]),
        "USA_2018":   np.array([1498.2, 1612.0, 2340.5, 38.1]),
        "India_2018": np.array([1661.0, 44.4, 557.3, 183.2]),
        "Germany":    np.array([194.3, 79.6, 216.6, 19.7]),
        "Anomaly_X":  np.array([9800.0, 3500.0, 4000.0, 2000.0]),  # synthetic
    }

    # Scale to [-π, π]
    vals = np.array(list(TOP_EMITTERS.values()))
    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    vals_s = scaler.fit_transform(vals)

    qc = SimulatedQuantumClassifier(n_qubits=4, n_layers=3)

    print("\n── Quantum Anomaly Scores — Top Emitters ───────────────")
    print(f"{'Country':15s} {'Score':>8}  {'Label':>10}")
    print("─" * 38)
    for (name, _), xs in zip(TOP_EMITTERS.items(), vals_s):
        score = qc.anomaly_score(xs)
        label = "ANOMALY" if score > 0.5 else "normal"
        print(f"{name:15s} {score:8.4f}  {label:>10}")


if __name__ == "__main__":
    print("NeutralCarbon — QML Classifier Demo")
    print("=" * 60)

    demo_circuit_on_emissions()

    print("\n[Loading dataset for full pipeline demo…]")
    df = clean(load_raw())
    df_train, df_test = train_test_split_temporal(df)
    X_train, _, _       = get_feature_matrix(df_train)
    X_test,  meta_test, _ = get_feature_matrix(df_test)

    qml = QMLAnomalyDetector(n_qubits=4, n_layers=3)
    qml.fit(X_train[:400])             # subset for speed
    flags, scores = qml.predict(X_test[:100])
    report = qml.anomaly_report(flags, scores, meta_test.iloc[:100])

    print(f"\nTop flagged records:\n{report.head(10).to_string()}")
