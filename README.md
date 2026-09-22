# Blockchain-Enabled Federated AI Framework for Secure Distributed Intelligence

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Modern-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade decentralized, privacy-preserving threat intelligence framework combining **Federated Learning (FL)** with **Permissioned Blockchain Technology** and **Smart Contracts** over the **CICIDS2017** benchmark dataset.

---

## Key Features

1. **Tier 1: Client Silo Layer**
   - **Hospital A** (Healthcare Node): Infiltration, Heartbleed, PortScan threat distribution.
   - **Bank B** (Financial Node): DDoS (LOIC/HOIC), DoS GoldenEye, FTP/SSH-Patator threats.
   - **IoT Node C** (Critical Edge Gateway): Botnet (ARES), DoS Slowloris, Web Attacks (SQLi, XSS, Brute Force).
   - **Differential Privacy (DP-SGD)**: Gradient norm clipping ($L_2$ norm $C$) and calibrated Gaussian noise injection ($\sigma$) providing formal $(\epsilon, \delta)$-DP guarantees without raw telemetry leakage.
   - **Cryptographic Signatures**: SECP256R1/SECP256K1 ECDSA asymmetric key pairs and SHA-256 weight fingerprinting.

2. **Tier 2: Smart Contract & Byzantine Shield**
   - **Hybrid On-Chain / Off-Chain Architecture**: Overcomes EVM gas constraints (solving Ramanan et al. BAFFLE limitations) by committing cryptographic SHA-256 weight hashes and Merkle roots on-chain.
   - **Byzantine Fault Tolerance**: Cosine Distance Shield, Multi-Krum, and Coordinate-wise Median defenses detecting and slashing malicious nodes injecting sign-flipped, Gaussian noise, or backdoor gradients.
   - **Deterministic FedAvg Contract**: Automated aggregation on verified benign updates.
   - **Production Solidity Contract**: `contracts/FederatedAuditLedger.sol`.

3. **Tier 3: Ledger & SOC Dashboard**
   - **Cryptographic Blockchain Ledger**: SHA-256 block chaining, Merkle trees, and Proof-of-Authority validator signatures.
   - **Tamper Evidence**: Built-in cryptographic integrity auditor that flags any altered block, broken pointer, or invalid Merkle root.
   - **Interactive SOC Security Cockpit**: Modern dark mode glassmorphism UI with real-time Chart.js convergence curves, live block inspector, and one-click compliance export (GDPR / HIPAA / SOC 2).

---

## Directory Structure

```
├── core/
│   ├── __init__.py
│   ├── dataset.py            # CICIDS2017 loader & Non-IID partitioner (78 features, 15 classes)
│   ├── model.py              # PyTorch Deep Neural Network (DNN) IDS classifier
│   ├── privacy.py            # DP-SGD engine with (epsilon, delta) privacy accounting
│   ├── crypto.py             # ECDSA key management, SHA-256 hashing, Merkle trees
│   ├── byzantine.py          # Byzantine defense (Cosine Shield, Multi-Krum) & attack simulator
│   ├── blockchain.py         # Permissioned blockchain ledger & tamper detection
│   └── federated_engine.py   # Three-tier coordinator & round orchestrator
├── contracts/
│   └── FederatedAuditLedger.sol # Production Solidity smart contract
├── server/
│   ├── __init__.py
│   └── app.py                # FastAPI backend & REST/SSE endpoints
├── frontend/
│   ├── index.html            # SOC Dashboard markup
│   ├── styles.css            # Dark mode glassmorphism styles
│   └── app.js                # Interactive dashboard controller & Chart.js integration
├── tests/
│   └── test_system.py        # Automated test suite (7 comprehensive test cases)
├── PROJECT_REPORT.md         # Comprehensive academic & engineering project report
└── README.md
```

---

## Quick Start Guide

### Prerequisites
- Python 3.11+
- PyTorch (`torch`, `torchvision`)
- NumPy, Pandas, Scikit-learn
- FastAPI, Uvicorn, Cryptography

### 1. Run Automated Unit Tests
Verify all mathematical, cryptographic, and algorithmic components:
```bash
python -m unittest tests/test_system.py
```
*(All 7 test suites pass in ~3.5 seconds)*

### 2. Launch the Web Application & SOC Cockpit
Start the FastAPI server:
```bash
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```
Open your browser and navigate to:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### 3. Interactive SOC Dashboard Controls
- **Execute Round**: Trains 1 federated round across Hospital A, Bank B, and IoT Node C, computes DP-SGD gradients, verifies ECDSA signatures, aggregates via FedAvg, and mines a new blockchain block.
- **Run 3 Rounds**: Runs 3 rounds sequentially to observe accuracy and loss convergence.
- **Inject Adversarial Attack**: Toggles a rogue node sending poisoned updates; observe the Byzantine Shield identify, slash, and exclude the rogue node on-chain!
- **Audit Chain Integrity**: Runs real-time cryptographic verification of every block and Merkle root.
- **Simulate Block Tamper**: Injects an artificial modification into Block #1 to verify instant tamper detection.
- **Inspect Block**: Click on any block card in the horizontal explorer to view full transactions, Merkle roots, and cryptographic signatures.
- **Audit Report**: Downloads the official JSON forensic audit certificate for regulatory compliance.

---

## REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | Current system state, participant metadata, and baseline metrics |
| `/api/train-round` | POST | Triggers a federated round (epochs, lr, DP toggle, attack toggle) |
| `/api/blockchain` | GET | Returns full blockchain history, blocks, and transactions |
| `/api/blockchain/verify` | POST | Performs cryptographic audit on the entire ledger |
| `/api/blockchain/tamper-test` | POST | Simulates a block tamper to demonstrate alert detection |
| `/api/reset` | POST | Resets orchestrator state back to genesis |
| `/api/export-audit` | GET | Exports GDPR/HIPAA compliant forensic audit JSON report |
