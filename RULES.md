# Agent Instructions & Engineering Rules (RULES.md)

## Project: Blockchain-Enabled Federated AI Framework for Secure Distributed Intelligence
**Document Version:** 1.0.0  
**Scope:** Engineering Invariants, Agent Coding Rules, Security Guardrails, and Clone Implementation Protocol  

---

### 1. Architectural Invariants (Non-Negotiable Rules)

When implementing, refactoring, or cloning this repository, every human engineer and autonomous AI agent **must adhere strictly** to the following invariants:

1. **Zero Raw Telemetry Exfiltration**:
   - Under no circumstances may raw network packet flows, raw feature vectors, or individual sample data leave the Tier 1 organizational boundary (`Hospital A`, `Bank B`, `IoT Node C`).
   - Only aggregated, differentially private, cryptographically fingerprinted parameter vectors ($\mathbf{w}$) and their corresponding SHA-256 hashes ($h_k$) may be shared across the network.
2. **Hybrid On-Chain / Off-Chain Principle (The BAFFLE Rule)**:
   - High-dimensional neural network weights ($21,391$ float32 numbers $\approx 83.56\text{ KB}$) must **never** be stored directly inside Ethereum Virtual Machine (EVM) storage. Storing raw weights on-chain violates EVM block gas limits ($> 30\text{M}$ gas).
   - Only 32-byte SHA-256 cryptographic fingerprints, Binary Merkle roots, participant public keys, and transaction receipts are recorded on-chain.
3. **Deterministic Weight Serialization**:
   - Neural network weights must always be cast explicitly to 32-bit floating point (`np.float32`) before serialization.
   - Parameter vectors must be flattened in deterministic order across layers:
     `[net.0.weight, net.0.bias, net.1.weight, net.1.bias, ..., net.11.weight, net.11.bias]`
   - Hashing is defined as `hashlib.sha256(weights.astype(np.float32).tobytes()).hexdigest()`.
4. **Byzantine Slashing Immutability**:
   - When the Byzantine Shield flags an update ($\mathcal{S}_k < \tau$), the malicious client must be formally excluded from the FedAvg calculation pool.
   - The transaction must still be recorded in the mined block with status `"SLASHED_BYZANTINE"` and `tx_type = "ADVERSARIAL_ATTACK_REJECTED"` to preserve an audit trail of the attack attempt.
5. **Continuous Cryptographic Tamper-Evidence**:
   - Every block in the ledger must cryptographically seal the hash of the preceding block (`previous_hash == chain[i-1].block_hash`).
   - Any break in the chain, altered Merkle root, or modified parameter hash must immediately cause `verify_ledger_integrity()` to return `is_valid: False` with the exact tampered index.

---

### 2. Directory Structure & File Responsibilities

When cloning this project, the filesystem structure must match the following layout exactly:

```
├── core/
│   ├── __init__.py           # Package export
│   ├── dataset.py            # CICIDS2017 feature definitions (78 feats, 15 classes), Non-IID generator
│   ├── model.py              # PyTorch DNN IDS (78->128->64->32->15) + multi-class evaluation
│   ├── privacy.py            # DP-SGD engine (L2 norm clipping + calibrated Gaussian noise)
│   ├── crypto.py             # SECP256R1 ECDSA key manager, SHA-256 weight hash, Merkle tree
│   ├── byzantine.py          # Cosine Distance Shield, Multi-Krum, Median, Attack Simulator
│   ├── blockchain.py         # Permissioned PoA ledger, block chaining, tamper auditor
│   └── federated_engine.py   # Tier 1/2/3 orchestrator, round coordinator, client node state
├── contracts/
│   └── FederatedAuditLedger.sol # Production Solidity smart contract (v0.8.20)
├── server/
│   ├── __init__.py           # Package export
│   └── app.py                # FastAPI REST API, static asset mounting, CORS middleware
├── frontend/
│   ├── index.html            # High-tech SOC Cockpit markup, glassmorphism dashboard
│   ├── styles.css            # Dark-mode styling, ambient glows, CSS variables
│   └── app.js                # Single-page controller, Chart.js dual-axis curves, block inspector
├── tests/
│   └── test_system.py        # 7 comprehensive unit & integration test suites
├── PRD.md                    # Product Requirements Document
├── ARCHITECTURE.md           # System Architecture & Technical Specifications
├── DATA_MODEL.md             # Backend Schema, Smart Contract State & API Payloads
├── RULES.md                  # Agent Instructions & Engineering Invariants (This File)
└── README.md                 # Project Overview & Execution Guide
```

---

### 3. Coding Standards & Conventions

#### 3.1 Python Backend (`core/`, `server/`, `tests/`)
- **Python Version**: `>= 3.11`.
- **Typing**: Use static type annotations (`typing.Dict`, `typing.List`, `typing.Tuple`, `typing.Any`, `typing.Optional`) on all public function and method signatures.
- **Random Seeding**: Always initialize pseudo-random generators (`np.random.seed(42)`, `torch.manual_seed(42)`) to ensure reproducible benchmarks and deterministic test execution.
- **Deep Neural Network Device Management**: Ensure all tensors and models are placed consistently on `self.device` (default `torch.device("cpu")` for zero-setup portability, with seamless CUDA capability if available).
- **Error Handling**: Use standard FastAPI `HTTPException` with explicit status codes (400 for bad client state, 500 for internal errors). Do not swallow cryptographic verification exceptions silently.

#### 3.2 Solidity Smart Contracts (`contracts/`)
- **Compiler Version**: `pragma solidity ^0.8.20;`.
- **License Identifier**: `// SPDX-License-Identifier: MIT` at top of every contract.
- **Access Control**: Enforce `onlyAdmin` and `onlyActiveParticipant` function modifiers.
- **Checks-Effects-Interactions**: Follow CEI pattern to eliminate reentrancy vulnerabilities.
- **Events**: Emit indexed events for all state-changing actions (`ParticipantRegistered`, `RoundStarted`, `CommitmentSubmitted`, `ByzantineClientSlashed`, `RoundFinalized`).

#### 3.3 Frontend Development (`frontend/`)
- **Zero-Build Requirement**: Use vanilla HTML5, modern CSS3, and standard ES6+ JavaScript. Do not introduce complex build steps (e.g., Webpack/Vite) unless requested; keep it instantly runnable via FastAPI static mounting.
- **Aesthetic Excellence**: High-tech SOC cockpit dark mode. Use curated color palettes (Deep Space `#0a0e17`, Cyan `#00f2fe`, Emerald `#10b981`, Rose `#f43f5e`), glassmorphism cards (`backdrop-filter: blur(16px)`), and modern typography (Plus Jakarta Sans & JetBrains Mono).
- **Chart.js Performance**: Destroy and re-instantiate or properly update Chart.js instances on successive training rounds to prevent canvas memory leaks.

---

### 4. Cryptographic & Privacy Guardrails

```
CRITICAL IMPLEMENTATION CHECKLIST:

[ ] SECP256R1 Keypairs: Use cryptography.hazmat.primitives.asymmetric.ec.SECP256R1
[ ] Signature Scheme: ECDSA with SHA-256 digest
[ ] Public Key Export: SubjectPublicKeyInfo in PEM format serialized as Hex
[ ] Weight Fingerprint: SHA-256 of IEEE 754 float32 byte array (64-char hex string)
[ ] Merkle Tree:
      - Empty leaf list returns SHA-256 of b"empty_tree"
      - Single leaf returns itself
      - Odd count leaf list duplicates the last leaf on that level
[ ] DP-SGD Sequence:
      Step 1: Compute global L2 norm of gradients
      Step 2: Compute clip factor: min(1.0, C / (norm + 1e-6))
      Step 3: Multiply gradients by clip factor
      Step 4: Add Gaussian noise: N(0, (sigma * C / sqrt(B))^2 * I)
      Step 5: Call optimizer.step()
[ ] Byzantine Cosine Shield:
      - Coordinate-wise median vector: v_median = median(updates, axis=0)
      - Alignment: dot(v, v_median) / (||v|| * ||v_median||)
      - Threshold: Accept if >= 0.20, Reject/Slash if < 0.20
```

---

### 5. Step-by-Step Clean-Room Clone Protocol

Follow this exact sequential protocol when cloning or rebuilding this system from scratch:

#### Step 1: Environment & Dependencies
Create a virtual environment and install core dependencies:
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install torch torchvision numpy pandas scikit-learn fastapi uvicorn cryptography
```

#### Step 2: Build `core/crypto.py`
Implement `hash_sha256()`, `hash_weights()`, `compute_merkle_root()`, and `NodeKeyManager` using Python's `cryptography` library.
*Verification*: Test key generation, signing, signature verification, and Merkle root calculation.

#### Step 3: Build `core/dataset.py`
Define `CICIDS_CLASSES` (15 classes) and `FEATURE_NAMES` (78 features). Implement `CICIDSDataEngine` with `generate_synthetic_benchmark()` using domain-specific feature profiles for `Hospital A`, `Bank B`, and `IoT Node C`. Wrap in PyTorch `Dataset` and `DataLoader`.
*Verification*: Validate batch shapes `(BatchSize, 78)` and integer labels in range `[0, 14]`.

#### Step 4: Build `core/model.py` & `core/privacy.py`
Implement `IntrusionDetectionDNN` with sequential linear layers, batch norm, LeakyReLU, and dropout. Implement `get_flat_parameters()` and `set_flat_parameters()`. Build `DifferentialPrivacyEngine` with gradient norm clipping and noise injection.
*Verification*: Verify parameter vector has length $21,391$, and DP accountant increments steps.

#### Step 5: Build `core/byzantine.py` & `core/blockchain.py`
Implement `ByzantineDefenseEngine` (Cosine Distance Shield, Multi-Krum, Coordinate-wise Median) and `AdversarialAttackSimulator`. Implement `BlockchainTransaction`, `BlockchainBlock`, and `PermissionedAuditLedger` with `verify_ledger_integrity()`.
*Verification*: Verify that simulated sign-flipped attacks are rejected, and tampering with block previous hashes triggers `is_valid: False`.

#### Step 6: Build `core/federated_engine.py`
Implement `FederatedClientNode` and `FederatedOrchestrator`. Connect Tier 1 client local training, Tier 2 signature verification and Byzantine anomaly defense, and Tier 3 blockchain block creation.
*Verification*: Execute 1 complete round; assert that block 1 is mined and ledger integrity is verified.

#### Step 7: Write Smart Contract `contracts/FederatedAuditLedger.sol`
Write the Solidity contract capturing participant registration, round progression, commitment hashes, Byzantine slashing, and finalization.
*Verification*: Compile with Solidity 0.8.20 compiler (e.g., via `solc` or Hardhat/Foundry).

#### Step 8: Build `server/app.py` & `frontend/`
Expose the FastAPI endpoints (`/api/status`, `/api/train-round`, `/api/blockchain`, `/api/blockchain/verify`, `/api/blockchain/tamper-test`, `/api/reset`, `/api/export-audit`). Implement `frontend/index.html`, `frontend/styles.css`, and `frontend/app.js` with Chart.js visualization.
*Verification*: Launch server via `python -m uvicorn server.app:app --port 8000` and test UI in browser.

#### Step 9: Validate with Full Test Suite
Run the 7 automated unit and integration tests:
```bash
python -m unittest tests/test_system.py
```
**Acceptance Criteria**: All 7 test cases must pass in **< 4.0 seconds**.

---

### 6. Automated Test Suite Verification Matrix

Every test in `tests/test_system.py` must pass with zero failures:

| Test Case | Method Name | Invariant Verified | Expected Outcome |
|---|---|---|---|
| `test_01` | `test_01_dataset_and_partitioning` | CICIDS2017 loader creates 3 Non-IID silos with 78 features across 15 classes. | `Hospital A`, `Bank B`, `IoT Node C` loaders present; batch shape `(32, 78)`. |
| `test_02` | `test_02_model_architecture_and_eval` | Model flattening and parameter loading are mathematically bijective ($w = \text{set}(\text{get}(w))$). | Retrieved weights match modified weights with `atol=1e-5`; output shape `(10, 15)`. |
| `test_03` | `test_03_differential_privacy_engine` | Gradients are clipped to $C \le 1.0$ and noise is injected; privacy budget $\epsilon > 0$. | `initial_norm > 0.0`, `epsilon > 0.0`. |
| `test_04` | `test_04_crypto_and_merkle` | SECP256R1 signature verification succeeds for authentic data and fails for tampered data. | Valid signature `True`; corrupted payload `False`; 64-char Merkle root. |
| `test_05` | `test_05_byzantine_defense` | Cosine Distance Shield detects and slashes sign-flipped adversarial attacks. | `Attacker X` in `rejected`; `Hospital A`, `Bank B`, `IoT Node C` in `accepted`. |
| `test_06` | `test_06_blockchain_ledger_and_tamper_detection` | Ledger validates genesis + round 1 block, and instantly detects artificial hash mutation. | Valid ledger `True`; corrupted previous hash triggers `is_valid: False`. |
| `test_07` | `test_07_end_to_end_orchestrator_round` | Full end-to-end integration round across all 3 tiers with adversarial injection. | `round_number == 1`, adversary rejected, `ledger_verified: True`. |

---

### 7. Common Pitfalls & Troubleshooting Guide

1. **PyTorch BatchNorm on Batch Size 1**:
   - *Problem*: PyTorch `nn.BatchNorm1d` raises an error if an evaluation batch contains only 1 sample while the model is in training mode.
   - *Fix*: In `IntrusionDetectionDNN.forward()`, check `if x.size(0) == 1 and self.training: self.eval(); out = self.net(x); self.train(); return out`.
2. **StandardScaler Zero-Variance Features**:
   - *Problem*: In synthetic flow generation, invariant features have zero standard deviation, resulting in `NaN` during division `(X - mean) / std`.
   - *Fix*: In `dataset.py`, enforce `std[std == 0] = 1.0` before normalization.
3. **ECDSA Signature Verification Failures**:
   - *Problem*: Passing string representations instead of raw byte digests or passing un-hexed PEM strings causes deserialization exceptions.
   - *Fix*: Use `bytes.fromhex(public_key_hex)` to load PEM, and `bytes.fromhex(signature_hex)` for verification inside a `try...except` block.
4. **Merkle Tree Odd Leaf Count**:
   - *Problem*: Merkle tree generator crashes or produces mismatched roots when an odd number of transactions (e.g. 3 clients) is provided.
   - *Fix*: In `compute_merkle_root()`, duplicate the odd leaf: `right = current_level[i + 1] if i + 1 < len(current_level) else left`.
5. **Chart.js Canvas Reuse Error**:
   - *Problem*: `"Canvas is already in use. Chart with ID '0' must be destroyed before the canvas can be reused."`
   - *Fix*: In `frontend/app.js`, keep global references to chart instances (`window.convergenceChart`) and call `.destroy()` before instantiating a new chart, or update `.data.datasets` directly and call `.update()`.
