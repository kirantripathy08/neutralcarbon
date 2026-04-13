# NeutralCarbon 🌿

**A Quantum-Enhanced Blockchain Framework for Verifiable Carbon Credit Tokenization and Fraud Detection**

> BCSE498J Project-II | VIT Vellore | Feb 2026  
> Authors: Kiran Tripathy (22BDS0172) · Kolangada Advaith Dilip (22BCE0772)  
> Supervisor: Prof. Akila Victor, SCOPE

---

## Overview

NeutralCarbon integrates **Quantum Machine Learning (QML)**, **classical ML anomaly detection**, and **ERC-1155 blockchain tokenization** to build a transparent, fraud-resistant carbon credit verification system.

```
Data Sources → Preprocessing → ML/QML Anomaly Detection → Chainlink Oracle → Smart Contract → ERC-1155 Tokens
```

---

## Repository Structure

```
neutralcarbon/
├── data/
│   ├── global_emissions.csv          # 92 countries, 1992–2018
│   └── preprocess.py                 # Data cleaning & feature engineering
├── ml/
│   ├── anomaly_detection.py          # Isolation Forest, One-Class SVM, Autoencoder
│   ├── train.py                      # Training pipeline
│   └── evaluate.py                   # Metrics & visualizations
├── quantum/
│   ├── quantum_circuit.py            # TF Quantum / Cirq circuit definition
│   ├── qml_classifier.py             # Quantum ML anomaly classifier
│   └── feature_encoding.py           # Angle encoding of emission features
├── blockchain/
│   ├── contracts/
│   │   ├── CarbonCredit.sol          # ERC-1155 carbon credit token
│   │   └── CarbonOracle.sol          # Chainlink oracle consumer
│   ├── scripts/
│   │   ├── deploy.js                 # Hardhat deploy script
│   │   └── mint.js                   # Token minting script
│   ├── hardhat.config.js
│   └── package.json
├── dashboard/
│   └── index.html                    # Interactive web dashboard
├── tests/
│   ├── test_anomaly.py               # ML unit tests
│   ├── test_quantum.py               # Quantum module tests
│   └── CarbonCredit.test.js          # Smart contract tests
├── docs/
│   └── architecture.md               # System architecture docs
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI pipeline
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Python Environment (ML + Quantum)
```bash
git clone https://github.com/YOUR_USERNAME/neutralcarbon.git
cd neutralcarbon
pip install -r requirements.txt
```

### 2. Run Anomaly Detection
```bash
python ml/train.py
python ml/evaluate.py
```

### 3. Run Quantum Module (Simulation)
```bash
python quantum/qml_classifier.py
```

### 4. Deploy Smart Contracts
```bash
cd blockchain
npm install
npx hardhat compile
npx hardhat run scripts/deploy.js --network localhost
```

### 5. Launch Dashboard
```bash
# Simply open dashboard/index.html in a browser
# Or serve with Python:
python -m http.server 8080
# Visit http://localhost:8080/dashboard/
```

---

## Key Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Isolation Forest | 88.4% | 0.86 | 0.91 | 0.88 |
| One-Class SVM | 85.1% | 0.83 | 0.88 | 0.85 |
| Autoencoder | 90.3% | 0.89 | 0.92 | 0.90 |
| **QML (Quantum)** | **94.2%** | **0.93** | **0.96** | **0.94** |

---

## Tech Stack

- **ML**: scikit-learn, TensorFlow, PyTorch
- **Quantum**: TensorFlow Quantum, Cirq, PennyLane
- **Blockchain**: Solidity, Hardhat, Ethers.js, Chainlink
- **Data**: pandas, numpy, matplotlib, seaborn, plotly
- **Testing**: pytest, Hardhat Mocha

---

## References

1. Zeng et al. (2022) — ML-based anomaly detection for environmental monitoring
2. Havlíček et al. (2019) — Supervised learning with quantum-enhanced feature spaces
3. Ethereum Foundation — ERC-1155 Multi-Token Standard
4. Chainlink Labs — Decentralized Oracle Networks
5. Chandola et al. (2009) — Anomaly Detection: A Survey

---

## License

MIT License — see `LICENSE` file.
