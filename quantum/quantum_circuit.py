"""
neutralcarbon/quantum/quantum_circuit.py
-----------------------------------------
Quantum circuit definition using Cirq for the NeutralCarbon QML module.

Architecture:
  - 4 qubits (one per primary emission feature group)
  - Angle encoding layer  : encodes classical data as rotation angles
  - Entanglement layer    : CNOT gates for feature correlation
  - Variational layer     : parametrised RZ rotations (trainable)
  - Measurement           : Z-basis measurement → expectation values

Reference:
  Havlíček et al. (2019) — Supervised learning with quantum-enhanced feature spaces
  [IEEE QCE 2019]
"""

import numpy as np
import sympy

try:
    import cirq
    CIRQ_AVAILABLE = True
except ImportError:
    CIRQ_AVAILABLE = False
    print("[warning] cirq not installed — using simulation mode")


class NeutralCarbonCircuit:
    """
    4-qubit variational quantum circuit for carbon emission anomaly detection.

    The circuit maps a 4-dimensional feature vector (after PCA reduction)
    to a single quantum expectation value used as an anomaly score.
    """

    N_QUBITS = 4

    def __init__(self):
        if not CIRQ_AVAILABLE:
            raise ImportError("pip install cirq tensorflow-quantum")

        self.qubits  = cirq.LineQubit.range(self.N_QUBITS)
        self.params  = sympy.symbols(
            " ".join([f"theta_{i}" for i in range(self.N_QUBITS * 3)])
        )
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> "cirq.Circuit":
        """
        Build the variational ansatz.

        Layer 1 — Hadamard (create superposition)
        Layer 2 — Angle encoding RY(x_i) — encodes data
        Layer 3 — Entanglement CNOT chain
        Layer 4 — Variational RZ(θ_i) — trainable
        Layer 5 — Second entanglement layer
        Layer 6 — Variational RY(φ_i) — trainable
        """
        q  = self.qubits
        p  = self.params
        ops = []

        # Layer 1: Hadamard
        ops += [cirq.H(qi) for qi in q]

        # Layer 2: Angle encoding (placeholder — replaced per sample during training)
        ops += [cirq.ry(p[i])(q[i]) for i in range(self.N_QUBITS)]

        # Layer 3: Entanglement (CNOT chain)
        ops += [cirq.CNOT(q[i], q[i + 1]) for i in range(self.N_QUBITS - 1)]
        ops += [cirq.CNOT(q[-1], q[0])]   # circular entanglement

        # Layer 4: Variational RZ
        ops += [cirq.rz(p[self.N_QUBITS + i])(q[i]) for i in range(self.N_QUBITS)]

        # Layer 5: Second entanglement
        ops += [cirq.CNOT(q[i + 1], q[i]) for i in range(self.N_QUBITS - 1)]

        # Layer 6: Variational RY
        ops += [cirq.ry(p[2 * self.N_QUBITS + i])(q[i]) for i in range(self.N_QUBITS)]

        return cirq.Circuit(ops)

    def readout_ops(self) -> list:
        """Z-basis measurement operators on all qubits."""
        return [cirq.Z(qi) for qi in self.qubits]

    def print_circuit(self) -> None:
        print(self.circuit)

    def n_parameters(self) -> int:
        return len(self.params)


class SimulatedQuantumClassifier:
    """
    Simulated quantum classifier — does NOT require Cirq/TFQ.
    Uses classical numpy to approximate quantum feature transformation.

    Implements the same angle-encoding + entanglement logic numerically,
    allowing the pipeline to run without quantum hardware.
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 3, seed: int = 42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.rng      = np.random.RandomState(seed)
        # Random variational parameters (would be trained in real QML)
        self.theta = self.rng.uniform(0, 2 * np.pi, (n_layers, n_qubits))

    def _rx(self, angle: float) -> np.ndarray:
        """Single-qubit RX rotation matrix."""
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        return np.array([[c, -1j * s], [-1j * s, c]])

    def _ry(self, angle: float) -> np.ndarray:
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        return np.array([[c, -s], [s, c]])

    def _rz(self, angle: float) -> np.ndarray:
        return np.diag([np.exp(-1j * angle / 2), np.exp(1j * angle / 2)])

    def encode_and_measure(self, x: np.ndarray) -> np.ndarray:
        """
        Encode a feature vector x (length n_qubits) into expectation values.
        Returns a vector of length n_qubits (simulated Z-expectation values).
        """
        # Start in |0⟩^n, represent as statevector
        dim    = 2 ** self.n_qubits
        state  = np.zeros(dim, dtype=complex)
        state[0] = 1.0

        # Apply Hadamard to each qubit (Kronecker product)
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        H_n = H
        for _ in range(self.n_qubits - 1):
            H_n = np.kron(H_n, H)
        state = H_n @ state

        # Angle encoding: RY(π * x_i) per qubit
        for i in range(self.n_qubits):
            angle = np.pi * np.tanh(x[i % len(x)])
            Ry    = self._ry(angle)
            # Apply to qubit i: kron(I^{i} ⊗ Ry ⊗ I^{n-i-1})
            op = np.eye(1)
            for j in range(self.n_qubits):
                op = np.kron(op, Ry if j == i else np.eye(2))
            state = op @ state

        # Variational layers
        for layer in range(self.n_layers):
            # Entanglement: CNOT chain
            for i in range(self.n_qubits - 1):
                state = self._apply_cnot(state, i, i + 1)

            # Parametrised RZ
            for i in range(self.n_qubits):
                angle = self.theta[layer, i]
                Rz    = self._rz(angle)
                op    = np.eye(1)
                for j in range(self.n_qubits):
                    op = np.kron(op, Rz if j == i else np.eye(2))
                state = op @ state

        # Z-expectation for each qubit
        expectations = []
        for i in range(self.n_qubits):
            exp_val = self._z_expectation(state, i)
            expectations.append(np.real(exp_val))

        return np.array(expectations)

    def _apply_cnot(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        """Apply CNOT gate between control and target qubits."""
        new_state = state.copy()
        dim = len(state)
        for idx in range(dim):
            bits = [(idx >> (self.n_qubits - 1 - k)) & 1
                    for k in range(self.n_qubits)]
            if bits[control] == 1:
                bits[target] ^= 1
                new_idx = sum(b << (self.n_qubits - 1 - k)
                              for k, b in enumerate(bits))
                new_state[new_idx], new_state[idx] = state[idx], state[new_idx]
        return new_state

    def _z_expectation(self, state: np.ndarray, qubit: int) -> complex:
        """⟨Z_i⟩ = Σ |c_k|² * (-1)^{bit_k_i}"""
        ev = 0.0
        for idx, amp in enumerate(state):
            bit = (idx >> (self.n_qubits - 1 - qubit)) & 1
            ev += abs(amp) ** 2 * (1 - 2 * bit)
        return ev

    def anomaly_score(self, x: np.ndarray) -> float:
        """
        Returns a scalar anomaly score in [0, 1].
        Higher → more likely anomalous.
        Uses the mean absolute deviation of expectation values from 0.
        """
        expectations = self.encode_and_measure(x)
        # Anomalies produce more extreme expectation values
        score = float(np.mean(np.abs(expectations)))
        return score

    def batch_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for all rows in X."""
        return np.array([self.anomaly_score(x) for x in X])


if __name__ == "__main__":
    print("NeutralCarbon — Quantum Circuit Demo")
    print("=" * 50)

    # Simulated classifier (no Cirq/TFQ needed)
    qc = SimulatedQuantumClassifier(n_qubits=4, n_layers=3)

    # Example: 4 emission features (scaled)
    normal_sample  = np.array([0.1, -0.2, 0.05, 0.3])
    anomaly_sample = np.array([3.8, 4.1, -3.5, 5.2])

    ns = qc.anomaly_score(normal_sample)
    as_ = qc.anomaly_score(anomaly_sample)

    print(f"Normal sample score:  {ns:.4f}")
    print(f"Anomaly sample score: {as_:.4f}")
    print(f"Separation ratio:     {as_/ns:.2f}x")

    # Show circuit if Cirq is available
    if CIRQ_AVAILABLE:
        print("\nFull Quantum Circuit:")
        nc_circuit = NeutralCarbonCircuit()
        nc_circuit.print_circuit()
        print(f"\nTrainable parameters: {nc_circuit.n_parameters()}")
