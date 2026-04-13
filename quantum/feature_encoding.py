"""
neutralcarbon/quantum/feature_encoding.py
------------------------------------------
Feature encoding strategies for mapping classical emission data
into quantum circuit rotation angles.

Strategies:
  1. AngleEncoding      — RY(π·x_i) per feature, standard approach
  2. AmplitudeEncoding  — encodes data as quantum state amplitudes
  3. IQPEncoding        — Instantaneous Quantum Polynomial circuits
  4. ZZFeatureMap       — kernel-based map (Havlíček et al., 2019)

Reference:
  Havlíček et al. (2019) — Supervised learning with quantum-enhanced
  feature spaces. IEEE QCE 2019.
"""
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from typing import Literal


class AngleEncoding:
    """
    Angle encoding: maps each feature x_i → RY(π · x_i) rotation.

    - Simple and hardware-efficient
    - Requires feature range in [−1, 1] (scales automatically)
    - 1 qubit per feature dimension
    """

    def __init__(self, n_qubits: int = 4, feature_range: tuple = (-1.0, 1.0)):
        self.n_qubits      = n_qubits
        self.feature_range = feature_range
        self.scaler        = MinMaxScaler(feature_range=feature_range)
        self.pca           = PCA(n_components=n_qubits)
        self._fitted       = False

    def fit(self, X: np.ndarray) -> "AngleEncoding":
        """Fit PCA (reduce to n_qubits dims) then scale."""
        X_pca = self.pca.fit_transform(X)
        self.scaler.fit(X_pca)
        self._fitted = True
        print(f"[AngleEncoding] PCA explained variance: "
              f"{self.pca.explained_variance_ratio_.sum():.3f}")
        return self

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode a single sample → rotation angles (radians).
        Returns array of shape (n_qubits,).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        x_pca    = self.pca.transform(x.reshape(1, -1))
        x_scaled = self.scaler.transform(x_pca).flatten()
        angles   = np.pi * x_scaled       # map [-1,1] → [-π, π]
        return angles

    def encode_batch(self, X: np.ndarray) -> np.ndarray:
        """Encode all rows. Returns (n_samples, n_qubits) angle matrix."""
        X_pca    = self.pca.transform(X)
        X_scaled = self.scaler.transform(X_pca)
        return np.pi * X_scaled

    def feature_importance(self) -> np.ndarray:
        """
        Return per-original-feature importance scores based on
        PCA loadings × explained variance.
        """
        loadings = np.abs(self.pca.components_)        # (n_qubits, n_features)
        ev       = self.pca.explained_variance_ratio_  # (n_qubits,)
        scores   = (loadings * ev[:, None]).sum(axis=0)
        return scores / scores.sum()                   # normalise to [0, 1]


class ZZFeatureMap:
    """
    ZZ feature map (Havlíček et al., 2019).
    Encodes pairwise feature interactions via ZZ gates, giving an
    exponentially large feature space in Hilbert space.

    φ(x) = exp(i · Σ_{j<k} (π - x_j)(π - x_k) · Z_j ⊗ Z_k) ·
            exp(i · Σ_j x_j · Z_j) · H^⊗n

    Simulated classically here using the induced kernel:
        K(x, x') = |⟨φ(x)|φ(x')⟩|²
    """

    def __init__(self, n_qubits: int = 4, reps: int = 2):
        self.n_qubits = n_qubits
        self.reps     = reps
        self.pca      = PCA(n_components=n_qubits)
        self.scaler   = MinMaxScaler(feature_range=(0, np.pi))
        self._fitted  = False

    def fit(self, X: np.ndarray) -> "ZZFeatureMap":
        X_pca = self.pca.fit_transform(X)
        self.scaler.fit(X_pca)
        self._fitted = True
        return self

    def _phi(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the ZZ feature vector for a single encoded sample x.
        Returns a real vector approximating the quantum state.
        """
        n = len(x)
        dim = 2 ** n
        state = np.ones(dim, dtype=complex) / np.sqrt(dim)   # H^⊗n |0⟩

        for _ in range(self.reps):
            # Single-qubit Z rotations: exp(i · x_j · Z_j)
            for j in range(n):
                phase = np.exp(1j * x[j])
                for idx in range(dim):
                    bit = (idx >> (n - 1 - j)) & 1
                    state[idx] *= (phase if bit == 0 else np.conj(phase))

            # Two-qubit ZZ interactions: exp(i · (π-x_j)(π-x_k) · ZZ)
            for j in range(n):
                for k in range(j + 1, n):
                    coeff = (np.pi - x[j]) * (np.pi - x[k])
                    for idx in range(dim):
                        bj = (idx >> (n - 1 - j)) & 1
                        bk = (idx >> (n - 1 - k)) & 1
                        zz = (1 - 2 * bj) * (1 - 2 * bk)
                        state[idx] *= np.exp(1j * coeff * zz)

        return state

    def kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Quantum kernel K(x1, x2) = |⟨φ(x1)|φ(x2)⟩|²"""
        phi1 = self._phi(x1)
        phi2 = self._phi(x2)
        return float(np.abs(np.dot(phi1.conj(), phi2)) ** 2)

    def kernel_matrix(self, X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
        """
        Compute the full quantum kernel matrix K[i,j] = K(X_i, X_j).
        If Y is None, computes symmetric K(X, X).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first")

        X_enc = self.scaler.transform(self.pca.transform(X))
        Y_enc = X_enc if Y is None else self.scaler.transform(self.pca.transform(Y))

        n, m = len(X_enc), len(Y_enc)
        K    = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                K[i, j] = self.kernel(X_enc[i], Y_enc[j])
            if (i + 1) % 10 == 0:
                print(f"  [ZZFeatureMap] Kernel row {i+1}/{n}")
        return K


class IQPEncoding:
    """
    Instantaneous Quantum Polynomial (IQP) encoding.
    Diagonal unitary circuits that are hard to simulate classically.

    Circuit: H^⊗n → D(x) → H^⊗n → D²(x) → … (reps times)
    where D(x) = exp(i · x_j · Z_j) ⊗ exp(i · x_j·x_k · ZZ_{jk})
    """

    def __init__(self, n_qubits: int = 4, reps: int = 1):
        self.n_qubits = n_qubits
        self.reps     = reps
        self.pca      = PCA(n_components=n_qubits)
        self.scaler   = MinMaxScaler(feature_range=(0, 2 * np.pi))
        self._fitted  = False

    def fit(self, X: np.ndarray) -> "IQPEncoding":
        self.pca.fit(X)
        X_pca = self.pca.transform(X)
        self.scaler.fit(X_pca)
        self._fitted = True
        return self

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode one sample → IQP expectation values, shape (n_qubits,)."""
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        x_pca    = self.pca.transform(x.reshape(1, -1))
        x_scaled = self.scaler.transform(x_pca).flatten()

        n   = self.n_qubits
        dim = 2 ** n
        state = np.ones(dim, dtype=complex) / np.sqrt(dim)

        for _ in range(self.reps):
            # D(x): single-qubit and pairwise diagonal gates
            for idx in range(dim):
                phase = 0.0
                bits = [(idx >> (n - 1 - j)) & 1 for j in range(n)]
                for j in range(n):
                    phase += x_scaled[j] * (1 - 2 * bits[j])
                for j in range(n):
                    for k in range(j + 1, n):
                        phase += x_scaled[j] * x_scaled[k] * (1 - 2 * bits[j]) * (1 - 2 * bits[k])
                state[idx] *= np.exp(1j * phase)

            # H^⊗n
            H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            H_n = H
            for _ in range(n - 1):
                H_n = np.kron(H_n, H)
            state = H_n @ state

        # Z-expectation values
        ev = []
        for j in range(n):
            exp_val = sum(
                abs(state[idx]) ** 2 * (1 - 2 * ((idx >> (n - 1 - j)) & 1))
                for idx in range(dim)
            )
            ev.append(float(np.real(exp_val)))

        return np.array(ev)


class FeatureEncodingBenchmark:
    """
    Compare encoding strategies on a small test set.
    Reports reconstruction quality and separation power.
    """

    @staticmethod
    def separation_score(X_normal: np.ndarray, X_anomaly: np.ndarray,
                         encoder) -> float:
        """
        Measures how well the encoding separates normal from anomalous samples
        using the Mahalanobis distance ratio.
        """
        enc_normal  = np.array([encoder.encode(x) for x in X_normal])
        enc_anomaly = np.array([encoder.encode(x) for x in X_anomaly])

        mu_n = enc_normal.mean(axis=0)
        mu_a = enc_anomaly.mean(axis=0)

        cov = np.cov(enc_normal.T) + 1e-6 * np.eye(enc_normal.shape[1])
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.eye(len(mu_n))

        diff  = mu_a - mu_n
        score = float(diff @ cov_inv @ diff)
        return score

    @staticmethod
    def run(X: np.ndarray, n_qubits: int = 4) -> dict:
        """
        Run benchmark on X.
        Injects 5% synthetic anomalies for evaluation.
        """
        rng      = np.random.RandomState(42)
        n        = len(X)
        n_anom   = max(5, int(n * 0.05))

        normal  = X[:n - n_anom]
        anomaly = X[n - n_anom:] * rng.uniform(3, 6, X[n - n_anom:].shape)

        encoders = {
            "AngleEncoding": AngleEncoding(n_qubits=n_qubits),
            "IQPEncoding":   IQPEncoding(n_qubits=n_qubits, reps=1),
        }

        results = {}
        for name, enc in encoders.items():
            enc.fit(normal)
            score = FeatureEncodingBenchmark.separation_score(
                normal[:20], anomaly[:20], enc
            )
            results[name] = round(score, 4)
            print(f"[Benchmark] {name}: separation score = {score:.4f}")

        return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    import os
    from data.preprocess import load_raw, clean, get_feature_matrix

    print("NeutralCarbon — Quantum Feature Encoding Demo")
    print("=" * 55)

    df = clean(load_raw())
    X, meta, _ = get_feature_matrix(df, extra_features=False)

    # Demo: angle encoding
    enc = AngleEncoding(n_qubits=4)
    enc.fit(X)
    angles = enc.encode(X[0])
    print(f"\nAngle encoding (first sample): {np.round(angles, 4)}")

    importance = enc.feature_importance()
    print(f"Feature importances (PCA-weighted): {np.round(importance, 3)}")

    # Demo: ZZ kernel on small subset
    print("\nComputing ZZ kernel matrix (20 samples)…")
    zz = ZZFeatureMap(n_qubits=4, reps=1)
    zz.fit(X[:100])
    K = zz.kernel_matrix(X[:20])
    print(f"Kernel matrix shape: {K.shape}, trace: {np.trace(K):.4f}")

    # Benchmark
    print("\nRunning encoding benchmark…")
    FeatureEncodingBenchmark.run(X[:200], n_qubits=4)
