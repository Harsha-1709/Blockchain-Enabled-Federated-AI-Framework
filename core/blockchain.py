"""
Permissioned Blockchain Ledger Engine
Implements:
- Cryptographic Block Chaining (SHA-256 header hashing, previous block linking)
- Merkle Root verification for transaction payloads
- Consortium Proof-of-Authority (PoA) validation & signatures
- Immutable audit log tracking training rounds, participant contributions,
  weight fingerprints, and Byzantine slash events
- Full ledger integrity auditing & tamper detection
"""

import time
import json
import numpy as np
from typing import Dict, List, Any, Optional
from core.crypto import hash_sha256, compute_merkle_root, NodeKeyManager

class BlockchainTransaction:
    """Represents a client contribution or governance event in a federated round"""
    def __init__(
        self,
        round_number: int,
        client_id: str,
        weight_hash: str,
        signature: str,
        sample_count: int,
        status: str = "ACCEPTED",
        anomaly_score: float = 1.0,
        tx_type: str = "WEIGHT_UPDATE",
        timestamp: Optional[float] = None
    ):
        self.round_number = round_number
        self.client_id = client_id
        self.weight_hash = weight_hash
        self.signature = signature
        self.sample_count = sample_count
        self.status = status  # ACCEPTED or SLASHED_BYZANTINE
        self.anomaly_score = anomaly_score
        self.tx_type = tx_type
        self.timestamp = timestamp or time.time()
        self.tx_hash = self.compute_tx_hash()

    def compute_tx_hash(self) -> str:
        payload = f"{self.round_number}:{self.client_id}:{self.weight_hash}:{self.signature}:{self.status}:{self.timestamp}"
        return hash_sha256(payload.encode("utf-8"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tx_hash": self.tx_hash,
            "round_number": self.round_number,
            "client_id": self.client_id,
            "weight_hash": self.weight_hash,
            "signature": self.signature,
            "sample_count": self.sample_count,
            "status": self.status,
            "anomaly_score": self.anomaly_score,
            "tx_type": self.tx_type,
            "timestamp": self.timestamp,
            "time_formatted": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        }

class BlockchainBlock:
    """Cryptographically sealed block in the consortium audit chain"""
    def __init__(
        self,
        index: int,
        round_number: int,
        previous_hash: str,
        transactions: List[BlockchainTransaction],
        global_model_hash: str,
        metrics: Dict[str, Any],
        validator_id: str = "SmartContractConsortiumValidator",
        timestamp: Optional[float] = None
    ):
        self.index = index
        self.round_number = round_number
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.global_model_hash = global_model_hash
        self.metrics = metrics
        self.validator_id = validator_id
        self.timestamp = timestamp or time.time()

        # Compute Merkle Root of all transactions
        tx_hashes = [tx.tx_hash for tx in transactions]
        self.merkle_root = compute_merkle_root(tx_hashes)

        self.block_hash = self.compute_block_hash()
        self.validator_signature = ""

    def compute_block_hash(self) -> str:
        header = f"{self.index}:{self.round_number}:{self.previous_hash}:{self.merkle_root}:{self.global_model_hash}:{self.timestamp}"
        return hash_sha256(header.encode("utf-8"))

    def seal_block(self, validator_key_mgr: NodeKeyManager):
        """Signs the block hash using the validator private key"""
        self.validator_signature = validator_key_mgr.sign_payload(self.block_hash)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "round_number": self.round_number,
            "previous_hash": self.previous_hash,
            "merkle_root": self.merkle_root,
            "block_hash": self.block_hash,
            "global_model_hash": self.global_model_hash,
            "validator_id": self.validator_id,
            "validator_signature": self.validator_signature,
            "metrics": self.metrics,
            "transaction_count": len(self.transactions),
            "transactions": [tx.to_dict() for tx in self.transactions],
            "timestamp": self.timestamp,
            "time_formatted": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        }

class PermissionedAuditLedger:
    """
    Maintains the consortium blockchain ledger, validating state transitions
    and storing immutable records of model evolution.
    """
    def __init__(self, validator_id: str = "ConsortiumSmartContract"):
        self.validator_key_mgr = NodeKeyManager(validator_id)
        self.chain: List[BlockchainBlock] = []
        self._create_genesis_block()

    def _create_genesis_block(self):
        genesis_tx = BlockchainTransaction(
            round_number=0,
            client_id="SystemInitializer",
            weight_hash=hash_sha256(b"GENESIS_ZERO_WEIGHTS"),
            signature="GENESIS_SIG",
            sample_count=0,
            status="GENESIS_CONFIRMED",
            tx_type="SYSTEM_INITIALIZATION"
        )
        genesis_block = BlockchainBlock(
            index=0,
            round_number=0,
            previous_hash="0000000000000000000000000000000000000000000000000000000000000000",
            transactions=[genesis_tx],
            global_model_hash=hash_sha256(b"INITIAL_GLOBAL_IDS_MODEL"),
            metrics={"accuracy": 0.0, "loss": 0.0, "status": "Consortium Genesis Activated"}
        )
        genesis_block.seal_block(self.validator_key_mgr)
        self.chain.append(genesis_block)

    def get_latest_block(self) -> BlockchainBlock:
        return self.chain[-1]

    def add_round_block(
        self,
        round_number: int,
        transactions: List[BlockchainTransaction],
        global_model_hash: str,
        metrics: Dict[str, Any]
    ) -> BlockchainBlock:
        """Mines and appends a verified round block to the audit ledger"""
        latest = self.get_latest_block()
        new_block = BlockchainBlock(
            index=len(self.chain),
            round_number=round_number,
            previous_hash=latest.block_hash,
            transactions=transactions,
            global_model_hash=global_model_hash,
            metrics=metrics,
            validator_id=self.validator_key_mgr.node_id
        )
        new_block.seal_block(self.validator_key_mgr)
        self.chain.append(new_block)
        return new_block

    def verify_ledger_integrity(self) -> Dict[str, Any]:
        """
        Cryptographically validates every block from genesis to current head:
        1. Checks previous_hash chaining
        2. Recalculates Merkle roots
        3. Re-computes SHA-256 block hashes
        4. Verifies validator digital signatures
        """
        for i in range(len(self.chain)):
            block = self.chain[i]

            # 1. Verify recalculation of block hash
            expected_hash = block.compute_block_hash()
            if block.block_hash != expected_hash:
                return {
                    "is_valid": False,
                    "tampered_block_index": block.index,
                    "reason": f"Block hash mismatch at index {block.index}"
                }

            # 2. Check previous hash linking
            if i > 0:
                prev_block = self.chain[i - 1]
                if block.previous_hash != prev_block.block_hash:
                    return {
                        "is_valid": False,
                        "tampered_block_index": block.index,
                        "reason": f"Broken chain pointer at index {block.index}"
                    }

            # 3. Verify Merkle Root
            tx_hashes = [tx.tx_hash for tx in block.transactions]
            expected_merkle = compute_merkle_root(tx_hashes)
            if block.merkle_root != expected_merkle:
                return {
                    "is_valid": False,
                    "tampered_block_index": block.index,
                    "reason": f"Invalid Merkle root at block {block.index}"
                }

            # 4. Verify validator signature
            is_sig_valid = NodeKeyManager.verify_signature(
                self.validator_key_mgr.get_public_key_hex(),
                block.block_hash,
                block.validator_signature
            )
            if not is_sig_valid:
                return {
                    "is_valid": False,
                    "tampered_block_index": block.index,
                    "reason": f"Invalid validator signature at block {block.index}"
                }

        return {
            "is_valid": True,
            "total_blocks": len(self.chain),
            "status": "Ledger verified. All hashes, signatures, and Merkle roots are cryptographically intact."
        }

    def export_ledger_json(self) -> List[Dict[str, Any]]:
        return [block.to_dict() for block in self.chain]
