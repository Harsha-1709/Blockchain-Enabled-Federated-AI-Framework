# Data Model & Schema Specification

## Project: Blockchain-Enabled Federated AI Framework for Secure Distributed Intelligence
**Document Version:** 1.0.0  
**Target Solution:** Clone & Clean-Room Reproduction Specification  
**Scope:** Backend State, Cryptographic Schemas, Smart Contract Storage, Tensors & REST Payloads  

---

### 1. Dataset & Threat Feature Schema

The system ingests and processes the **CICIDS2017** benchmark, comprising **78 continuous numerical features** extracted from bidirectional network flows via CICFlowMeter, categorized across **15 distinct classes**.

#### 1.1 Multi-Class Classification Enumeration (`CICIDS_CLASSES`)

| Class ID | Class Label (`string`) | Threat Category | Typical Signature / Flow Anomaly |
|---|---|---|---|
| `0` | `BENIGN` | Baseline Traffic | Standard flow durations, normal payload ratios, low SYN flags. |
| `1` | `Bot` | ARES Botnet | Elevated subflow packet counts, static window sizes (`8192`). |
| `2` | `DDoS` | Distributed DoS (LOIC/HOIC) | Massive packet volumes ($> 200$), high flow packets/sec ($> 500$), SYN flood flags. |
| `3` | `DoS GoldenEye` | Application-layer DoS | High HTTP keep-alive requests, variable inter-arrival times. |
| `4` | `DoS Hulk` | Volumetric Web DoS | Massive randomized user-agent requests, high byte throughput. |
| `5` | `DoS Slowhttptest` | Low-and-Slow Attack | Fragmented HTTP headers, slow transmission rates. |
| `6` | `DoS slowloris` | Connection Exhaustion | Elongated flow duration ($> 800\text{s}$), high Inter-Arrival Time (IAT) mean. |
| `7` | `FTP-Patator` | Protocol Brute Force | Repetitive small payloads, fixed port 21 connections. |
| `8` | `Heartbleed` | OpenSSL Buffer Over-read | Oversized backward payload length leaks ($> 4000\text{ bytes}$). |
| `9` | `Infiltration` | Lateral Movement / C2 | Large forward packet length ($> 1500\text{ bytes}$), high active data packets. |
| `10` | `PortScan` | Reconnaissance | Rapid sequential queries across varied destination ports, SYN flags. |
| `11` | `SSH-Patator` | Protocol Brute Force | Repetitive auth handshakes on port 22, fixed packet counts. |
| `12` | `Web Attack - Brute Force` | Web Authentication Attack | Repeated HTTP POST attempts, port 80/443, short duration. |
| `13` | `Web Attack - Sql Injection` | Database Injection | Anomalous URL parameter lengths, special character payload sizes. |
| `14` | `Web Attack - XSS` | Cross-Site Scripting | High payload length variances, script tag byte patterns. |

#### 1.2 The 78 Continuous Numerical Flow Features

```
Index | Feature Name                 | Data Type | Description
------+------------------------------+-----------+---------------------------------------------
0     | Destination Port             | float32   | Target transport port number
1     | Flow Duration                | float32   | Total duration of bidirectional flow in μs
2     | Total Fwd Packets            | float32   | Total packets in the forward direction
3     | Total Backward Packets       | float32   | Total packets in the backward direction
4     | Total Length of Fwd Packets  | float32   | Total size of payload in forward direction
5     | Total Length of Bwd Packets  | float32   | Total size of payload in backward direction
6-9   | Fwd Packet Length (Max/Min/Mean/Std)     | Statistical packet size moments (fwd)
10-13 | Bwd Packet Length (Max/Min/Mean/Std)     | Statistical packet size moments (bwd)
14    | Flow Bytes/s                 | float32   | Flow transfer rate in bytes per second
15    | Flow Packets/s               | float32   | Flow transfer rate in packets per second
16-19 | Flow IAT (Mean/Std/Max/Min)  | float32   | Inter-Arrival Time between flow packets
20-24 | Fwd IAT (Total/Mean/Std/Max/Min)         | Forward direction inter-arrival times
25-29 | Bwd IAT (Total/Mean/Std/Max/Min)         | Backward direction inter-arrival times
30-33 | Flags (Fwd/Bwd PSH, URG)     | float32   | TCP flag presence indicators
34-35 | Header Length (Fwd, Bwd)     | float32   | Header bytes in forward/backward directions
36-37 | Packets/s (Fwd, Bwd)         | float32   | Unidirectional transmission rates
38-42 | Packet Length (Min/Max/Mean/Std/Variance)| Overall bidirectional packet size moments
43-50 | TCP Flags (FIN, SYN, RST, PSH, ACK, URG, CWE, ECE) | TCP control bit counters
51    | Down/Up Ratio                | float32   | Ratio of download to upload packets
52-54 | Avg Packet/Segment Sizes     | float32   | Mean size metrics of frames and segments
55-61 | Bulk Transfer Rates & Flags  | float32   | Bulk data transfer characteristics
62-65 | Subflow Packets & Bytes      | float32   | Subflow partition metrics
66-67 | Init_Win_bytes (Fwd, Bwd)    | float32   | Initial TCP window byte sizes
68-69 | act_data_pkt_fwd, min_seg    | float32   | Active data packet counts and minimum segment
70-73 | Active (Mean/Std/Max/Min)    | float32   | Time flow remained active before going idle
74-77 | Idle (Mean/Std/Max/Min)      | float32   | Time flow remained idle before becoming active
```

#### 1.3 Non-IID Client Data Allocations

```json
{
  "Hospital A": {
    "organization_type": "Healthcare Consortium",
    "distribution": {
      "BENIGN": 0.50,
      "Infiltration": 0.20,
      "Heartbleed": 0.15,
      "PortScan": 0.15
    },
    "default_samples": 1200
  },
  "Bank B": {
    "organization_type": "Financial Banking Silo",
    "distribution": {
      "BENIGN": 0.45,
      "DDoS": 0.20,
      "DoS GoldenEye": 0.15,
      "FTP-Patator": 0.10,
      "SSH-Patator": 0.10
    },
    "default_samples": 1200
  },
  "IoT Node C": {
    "organization_type": "Critical Edge Gateway",
    "distribution": {
      "BENIGN": 0.40,
      "Bot": 0.25,
      "DoS slowloris": 0.15,
      "Web Attack - Brute Force": 0.08,
      "Web Attack - Sql Injection": 0.06,
      "Web Attack - XSS": 0.06
    },
    "default_samples": 1200
  }
}
```

---

### 2. Neural Network Model Tensor Schema

#### 2.1 Layer-by-Layer Dimension Breakdown

The model `IntrusionDetectionDNN` accepts a batch tensor of shape `(BatchSize, 78)` and outputs logits of shape `(BatchSize, 15)`.

| Layer Name | PyTorch Module | Input Shape | Output Shape | Parameter Count Calculation | Parameter Count |
|---|---|---|---|---|---|
| `net.0` | `nn.Linear` | `(B, 78)` | `(B, 128)` | $(78 \times 128) + 128$ | $9,984 + 128 = 10,112$ |
| `net.1` | `nn.BatchNorm1d` | `(B, 128)` | `(B, 128)` | $\gamma: 128, \beta: 128$ | $256$ |
| `net.2` | `nn.LeakyReLU(0.1)`| `(B, 128)` | `(B, 128)` | Non-parametric | $0$ |
| `net.3` | `nn.Dropout(0.25)` | `(B, 128)` | `(B, 128)` | Non-parametric | $0$ |
| `net.4` | `nn.Linear` | `(B, 128)` | `(B, 64)` | $(128 \times 64) + 64$ | $8,192 + 64 = 8,256$ |
| `net.5` | `nn.BatchNorm1d` | `(B, 64)` | `(B, 64)` | $\gamma: 64, \beta: 64$ | $128$ |
| `net.6` | `nn.LeakyReLU(0.1)`| `(B, 64)` | `(B, 64)` | Non-parametric | $0$ |
| `net.7` | `nn.Dropout(0.20)` | `(B, 64)` | `(B, 64)` | Non-parametric | $0$ |
| `net.8` | `nn.Linear` | `(B, 64)` | `(B, 32)` | $(64 \times 32) + 32$ | $2,048 + 32 = 2,080$ |
| `net.9` | `nn.BatchNorm1d` | `(B, 32)` | `(B, 32)` | $\gamma: 32, \beta: 32$ | $64$ |
| `net.10` | `nn.LeakyReLU(0.1)`| `(B, 32)` | `(B, 32)` | Non-parametric | $0$ |
| `net.11` | `nn.Linear` | `(B, 32)` | `(B, 15)` | $(32 \times 15) + 15$ | $480 + 15 = 495$ |
| **Total** | | | | | **21,391 parameters** |

#### 2.2 1D Flattened Parameter Array Schema
- **Data Type**: `numpy.float32` (4 bytes per parameter).
- **Total Elements**: $21,391$.
- **Contiguous Byte Length**: $21,391 \times 4 = 85,564\text{ bytes}$ (~83.56 KB).
- **Fingerprint Formula**:
  $$h = \text{SHA-256}(\mathbf{w}_{\text{flat}}.\text{tobytes}()) \quad \implies 64\text{ hex characters}$$

---

### 3. Cryptographic Data Models

#### 3.1 SECP256R1 Public Key Model
```typescript
interface PublicKeyPEM {
  format: "X.509 SubjectPublicKeyInfo";
  curve: "SECP256R1" | "prime256v1";
  encoding: "PEM_HEX"; // Hex-encoded string of standard PEM byte representation
  sample_value: "3059301306072a8648ce3d020106082a8648ce3d03010703420004..."
}
```

#### 3.2 Digital Signature Model
```typescript
interface ECDSASignature {
  algorithm: "ECDSA-SHA256";
  encoding: "DER_HEX"; // Hex-encoded DER sequence (r, s integers)
  sample_value: "3045022100e4b8a1c9...022031a7f0e9..."
}
```

#### 3.3 Binary Merkle Tree Data Structure
```typescript
interface MerkleNode {
  hash: string; // 64-char hex SHA-256
  left?: MerkleNode;
  right?: MerkleNode;
}

interface MerkleTreeCommitment {
  leaf_hashes: string[]; // List of transaction hashes
  depth: number;
  root_hash: string; // 64-char hex string
}
```

---

### 4. Blockchain Ledger Data Structures

#### 4.1 `BlockchainTransaction` Schema

```json
{
  "tx_hash": "a4f89d3c52e891b7d60e7f41c305b8214f2e96d741029c78ea3021bc984210ab",
  "round_number": 1,
  "client_id": "Hospital A",
  "weight_hash": "3e9b1d7f884210eac93120fb7412ca68019ab4215890eac1274bc91024eac891",
  "signature": "3045022100e4b8a1c93a01...",
  "sample_count": 1200,
  "status": "ACCEPTED",
  "anomaly_score": 0.9421,
  "tx_type": "WEIGHT_COMMITMENT",
  "timestamp": 1727041885.124,
  "time_formatted": "2026-09-23 03:21:25"
}
```

| Field Name | Type | Constraints / Enum | Description |
|---|---|---|---|
| `tx_hash` | `string` | 64-char SHA-256 Hex | SHA-256 of `{round}:{client_id}:{weight_hash}:{sig}:{status}:{timestamp}` |
| `round_number` | `integer` | $\ge 0$ | The communication round index |
| `client_id` | `string` | e.g., `"Hospital A"`, `"Bank B"` | Organization node identifier |
| `weight_hash` | `string` | 64-char SHA-256 Hex | Cryptographic fingerprint of the client's local parameters |
| `signature` | `string` | Hex string | ECDSA digital signature over `weight_hash` |
| `sample_count` | `integer` | $> 0$ | Local training dataset sample count used for FedAvg weighting |
| `status` | `string` | `"ACCEPTED"` \| `"SLASHED_BYZANTINE"` \| `"GENESIS_CONFIRMED"` | Byzantine evaluation verdict |
| `anomaly_score`| `float` | Range $[-1.0, 1.0]$ | Cosine similarity $\mathcal{S}_k$ or Multi-Krum distance score |
| `tx_type` | `string` | `"WEIGHT_COMMITMENT"` \| `"ADVERSARIAL_ATTACK_REJECTED"` \| `"SYSTEM_INITIALIZATION"` | Transaction categorization |
| `timestamp` | `float` | POSIX Epoch (sec) | Exact client submission timestamp |

#### 4.2 `BlockchainBlock` Schema

```json
{
  "index": 1,
  "round_number": 1,
  "previous_hash": "0000000000000000000000000000000000000000000000000000000000000000",
  "merkle_root": "b5a938c2ef091a4571bc84019ebf421980ac5170d10382901bce4710291a0c82",
  "block_hash": "e81249fa0218bc47190348acdf109247810471bade091248019a4bc10290192e",
  "global_model_hash": "f471092bacd81029471bc091248eac1902481029baecdf810291024bac019284",
  "validator_id": "SmartContractConsortiumValidator",
  "validator_signature": "3046022100fa...",
  "metrics": {
    "accuracy": 82.5,
    "loss": 0.4521,
    "f1_score": 81.9,
    "precision": 83.1,
    "recall": 82.5,
    "accepted_participants": 3,
    "rejected_participants": 0,
    "dp_enabled": true
  },
  "transaction_count": 3,
  "transactions": [ "...list of BlockchainTransaction..." ],
  "timestamp": 1727041886.582,
  "time_formatted": "2026-09-23 03:21:26"
}
```

| Field Name | Type | Hash Formula / Constraint |
|---|---|---|
| `block_hash` | `string` | $\text{SHA-256}(\text{Index} \parallel \text{Round} \parallel H_{\text{prev}} \parallel \text{MerkleRoot} \parallel H_{\text{global}} \parallel \text{Timestamp})$ |
| `previous_hash` | `string` | Must equal `chain[index - 1].block_hash` |
| `merkle_root` | `string` | Binary Merkle root derived from `[tx.tx_hash for tx in transactions]` |
| `validator_signature` | `string` | SECP256R1 signature of `block_hash` using validator private key |

#### 4.3 Ledger Integrity Verification Result Schema
```json
{
  "is_valid": true,
  "total_blocks": 4,
  "status": "Ledger verified. All hashes, signatures, and Merkle roots are cryptographically intact."
}
```
*(On Tamper Detection)*:
```json
{
  "is_valid": false,
  "tampered_block_index": 1,
  "reason": "Broken chain pointer at index 1"
}
```

---

### 5. Smart Contract Data Structures (`FederatedAuditLedger.sol`)

#### 5.1 Enums & Structs

```solidity
enum ParticipantStatus { Pending, Active, Suspended, Slashed }

struct Participant {
    string name;                 // e.g. "Hospital A"
    string organizationType;     // Healthcare, Financial, IoT Gateway
    string publicKeyPem;         // SECP256R1 Public Key in PEM format
    ParticipantStatus status;    // Active / Slashed / Suspended
    uint256 registrationTimestamp;
    uint256 totalAcceptedRounds;
    uint256 totalSlashedRounds;
}

struct ClientCommitment {
    address clientAddress;       // Ethereum address of consortium node
    string weightHash;           // SHA-256 fingerprint (64-char string)
    bytes signature;             // Cryptographic ECDSA signature
    uint32 sampleCount;          // Local sample volume
    bool isAccepted;             // True if passed Byzantine filter, false if slashed
    uint256 submissionTimestamp; // Block timestamp
}

struct RoundAuditRecord {
    uint256 roundNumber;
    string previousGlobalModelHash;
    string updatedGlobalModelHash;
    string merkleRoot;
    uint16 accuracyBps;          // Basis points (e.g., 8950 = 89.50%)
    uint16 lossBps;              // Basis points (e.g., 125 = 0.0125)
    uint256 completedTimestamp;
    uint256 participantCount;
    uint256 slashedCount;
    bool isFinalized;
}
```

#### 5.2 Solidity State Storage Layout

```solidity
address public consortiumAdmin;
uint256 public currentRound;
uint256 public totalRoundsCompleted;

mapping(address => Participant) public participants;
address[] public participantAddresses;

mapping(uint256 => RoundAuditRecord) public rounds;
mapping(uint256 => ClientCommitment[]) public roundCommitments;
mapping(uint256 => mapping(address => bool)) public hasCommitted;
```

#### 5.3 Smart Contract Events

```solidity
event ParticipantRegistered(address indexed clientAddress, string name, string organizationType);
event ParticipantStatusUpdated(address indexed clientAddress, ParticipantStatus status);
event RoundStarted(uint256 indexed roundNumber, string previousGlobalModelHash, uint256 timestamp);
event CommitmentSubmitted(uint256 indexed roundNumber, address indexed clientAddress, string weightHash);
event ByzantineClientSlashed(uint256 indexed roundNumber, address indexed clientAddress, string reason);
event RoundFinalized(
    uint256 indexed roundNumber,
    string updatedGlobalModelHash,
    string merkleRoot,
    uint16 accuracyBps,
    uint16 lossBps,
    uint256 acceptedParticipants
);
```

---

### 6. REST API Payloads & Request/Response Schemas

#### 6.1 `POST /api/train-round`

**Request Body (`TrainRoundRequest`)**:
```json
{
  "epochs": 1,
  "lr": 0.008,
  "dp_enabled": true,
  "simulate_attack": false,
  "attack_type": "sign_flip",
  "byzantine_method": "cosine_shield"
}
```

| Field | Type | Default | Validation |
|---|---|---|---|
| `epochs` | `integer` | `1` | $\ge 1, \le 10$ |
| `lr` | `float` | `0.008` | $> 0, \le 0.1$ |
| `dp_enabled` | `boolean` | `true` | Boolean flag |
| `simulate_attack`| `boolean` | `false` | Injects malicious rogue update |
| `attack_type` | `string` | `"sign_flip"` | `"sign_flip"` \| `"gaussian_noise"` \| `"backdoor"` |
| `byzantine_method`| `string`| `"cosine_shield"`| `"cosine_shield"` \| `"multi_krum"` \| `"trimmed_mean"` |

**Response Body**:
```json
{
  "success": true,
  "round": {
    "round_number": 1,
    "duration_sec": 3.12,
    "base_model_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "updated_model_hash": "4a591f0d7aa32104921bca9810472190ac47102910481029bac1092840192834",
    "metrics": {
      "loss": 0.6214,
      "accuracy": 82.5,
      "precision": 83.1,
      "recall": 82.5,
      "f1_score": 81.9,
      "sample_count": 1200,
      "per_class_accuracy": {
        "BENIGN": 0.94,
        "DDoS": 0.88,
        "Infiltration": 0.72
      }
    },
    "byzantine_defense": {
      "method": "Cosine Distance Shield",
      "accepted_clients": ["Hospital A", "Bank B", "IoT Node C"],
      "rejected_clients": [],
      "anomaly_scores": {
        "Hospital A": 0.9512,
        "Bank B": 0.9104,
        "IoT Node C": 0.9321
      }
    },
    "client_losses": {
      "Hospital A": 0.5412,
      "Bank B": 0.6120,
      "IoT Node C": 0.5891
    },
    "privacy_accounting": {
      "epsilon": 3.665,
      "delta": 0.00001,
      "steps": 19,
      "status": "ε = 3.67, δ = 1e-05"
    },
    "block": { "...BlockchainBlock JSON..." },
    "ledger_verified": true
  }
}
```

#### 6.2 `GET /api/status`

**Response Body**:
```json
{
  "success": true,
  "data": {
    "current_round": 2,
    "clients": {
      "Hospital A": {
        "name": "Hospital A",
        "org_type": "Healthcare Consortium",
        "sample_count": 1200,
        "attack_profile": ["Infiltration", "Heartbleed", "PortScan"],
        "last_loss": 0.421,
        "rounds_participated": 2,
        "public_key_preview": "3059301306072a8648ce3d02..."
      },
      "Bank B": { "...Bank B metadata..." },
      "IoT Node C": { "...IoT Node C metadata..." }
    },
    "metrics": {
      "accuracy": 86.4,
      "loss": 0.381,
      "f1_score": 85.9,
      "precision": 87.1,
      "recall": 86.4
    },
    "total_blocks": 3,
    "latest_block": { "...latest BlockchainBlock..." },
    "chain_validity": {
      "is_valid": true,
      "total_blocks": 3,
      "status": "Ledger verified..."
    },
    "classes": [ "BENIGN", "Bot", "DDoS", "...15 classes..." ]
  }
}
```

#### 6.3 `GET /api/export-audit` (Compliance Certificate Schema)

```json
{
  "consortium_name": "Distributed Cybersecurity Intelligence Consortium",
  "benchmark_dataset": "CICIDS2017 (Canadian Institute for Cybersecurity)",
  "report_generated_at": "2026-09-23 03:25:10 UTC",
  "total_rounds_completed": 3,
  "ledger_cryptographically_valid": true,
  "validation_details": {
    "is_valid": true,
    "total_blocks": 4,
    "status": "Ledger verified. All hashes, signatures, and Merkle roots are cryptographically intact."
  },
  "participating_silos": {
    "Hospital A": { "org_type": "Healthcare Consortium", "sample_count": 1200 },
    "Bank B": { "org_type": "Financial Banking Silo", "sample_count": 1200 },
    "IoT Node C": { "org_type": "Critical Edge Gateway", "sample_count": 1200 }
  },
  "final_model_performance": {
    "accuracy": 89.2,
    "loss": 0.312,
    "f1_score": 88.9,
    "precision": 89.5,
    "recall": 89.2
  },
  "blockchain_audit_blocks": [
    { "index": 0, "round_number": 0, "...genesis..." },
    { "index": 1, "round_number": 1, "...round 1..." },
    { "index": 2, "round_number": 2, "...round 2..." },
    { "index": 3, "round_number": 3, "...round 3..." }
  ]
}
```
