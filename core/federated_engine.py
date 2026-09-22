"""
Federated AI Engine Orchestrator
Coordinates Tier 1 Clients, Tier 2 Smart Contract & Byzantine Shield, and Tier 3 Audit Ledger.
Executes end-to-end decentralized training, DP-SGD privacy, ECDSA cryptographic verification,
Byzantine poisoning defense, and immutable blockchain block mining.
"""

import time
import copy
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from core.dataset import CICIDSDataEngine, NUM_FEATURES, NUM_CLASSES, CICIDS_CLASSES
from core.model import IntrusionDetectionDNN, evaluate_model
from core.privacy import DifferentialPrivacyEngine
from core.crypto import NodeKeyManager, hash_weights, hash_sha256
from core.byzantine import ByzantineDefenseEngine, AdversarialAttackSimulator
from core.blockchain import PermissionedAuditLedger, BlockchainTransaction, BlockchainBlock

class FederatedClientNode:
    """Represents a consortium enterprise participant (Tier 1)"""
    def __init__(
        self,
        node_id: str,
        name: str,
        org_type: str,
        data_loader: torch.utils.data.DataLoader,
        sample_count: int,
        attack_profile: List[str],
        device: torch.device = torch.device("cpu")
    ):
        self.node_id = node_id
        self.name = name
        self.org_type = org_type
        self.data_loader = data_loader
        self.sample_count = sample_count
        self.attack_profile = attack_profile
        self.device = device

        self.key_manager = NodeKeyManager(node_id)
        self.local_model = IntrusionDetectionDNN(NUM_FEATURES, NUM_CLASSES).to(device)
        self.privacy_engine = DifferentialPrivacyEngine(clip_norm=1.0, noise_multiplier=0.3, enabled=True)

        self.last_train_loss = 0.0
        self.last_weight_hash = ""
        self.total_rounds_participated = 0

    def local_train(
        self,
        global_flat_weights: np.ndarray,
        epochs: int = 1,
        lr: float = 0.01,
        dp_enabled: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Loads global model, trains on local CICIDS2017 partition with DP-SGD,
        and returns updated flat parameter vector.
        """
        self.local_model.set_flat_parameters(global_flat_weights)
        self.local_model.train()
        self.privacy_engine.enabled = dp_enabled

        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        steps = 0

        for epoch in range(epochs):
            for features, labels in self.data_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.local_model(features)
                loss = criterion(outputs, labels)
                loss.backward()

                # Apply DP-SGD gradient clipping and Gaussian noise
                if dp_enabled:
                    self.privacy_engine.clip_and_add_noise(
                        self.local_model,
                        batch_size=len(labels),
                        dataset_size=self.sample_count
                    )

                optimizer.step()
                total_loss += loss.item()
                steps += 1

        self.last_train_loss = total_loss / max(1, steps)
        self.total_rounds_participated += 1

        updated_weights = self.local_model.get_flat_parameters()
        self.last_weight_hash = hash_weights(updated_weights)
        return updated_weights, self.last_train_loss

    def sign_update(self, weight_hash: str) -> str:
        """Signs the weight update hash using the node's ECDSA private key"""
        return self.key_manager.sign_payload(weight_hash)

class FederatedOrchestrator:
    """
    Consortium Orchestrator managing Tier 1, 2, and 3.
    """
    def __init__(
        self,
        samples_per_client: int = 1200,
        byzantine_method: str = "cosine_shield",
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.data_engine = CICIDSDataEngine(random_seed=42)
        self.byzantine_engine = ByzantineDefenseEngine(method=byzantine_method)
        self.ledger = PermissionedAuditLedger(validator_id="SmartContractConsortiumValidator")

        # Global Intrusion Detection Model
        self.global_model = IntrusionDetectionDNN(NUM_FEATURES, NUM_CLASSES).to(self.device)
        self.current_round = 0
        self.round_history: List[Dict[str, Any]] = []

        # Load CICIDS2017 partitioned data
        client_loaders, self.test_loader, self.partition_stats = self.data_engine.get_data_loaders(
            batch_size=64, samples_per_client=samples_per_client
        )

        # Initialize Tier 1 Client Nodes
        self.clients: Dict[str, FederatedClientNode] = {
            "Hospital A": FederatedClientNode(
                node_id="hospital_a_node",
                name="Hospital A",
                org_type="Healthcare Consortium",
                data_loader=client_loaders["Hospital A"],
                sample_count=self.partition_stats["Hospital A"]["total_samples"],
                attack_profile=self.partition_stats["Hospital A"]["top_attacks"],
                device=self.device
            ),
            "Bank B": FederatedClientNode(
                node_id="bank_b_node",
                name="Bank B",
                org_type="Financial Banking Silo",
                data_loader=client_loaders["Bank B"],
                sample_count=self.partition_stats["Bank B"]["total_samples"],
                attack_profile=self.partition_stats["Bank B"]["top_attacks"],
                device=self.device
            ),
            "IoT Node C": FederatedClientNode(
                node_id="iot_c_node",
                name="IoT Node C",
                org_type="Critical Edge Gateway",
                data_loader=client_loaders["IoT Node C"],
                sample_count=self.partition_stats["IoT Node C"]["total_samples"],
                attack_profile=self.partition_stats["IoT Node C"]["top_attacks"],
                device=self.device
            )
        }

        # Initial baseline evaluation
        initial_eval = evaluate_model(self.global_model, self.test_loader, self.device)
        self.baseline_accuracy = initial_eval["accuracy"]
        self.baseline_loss = initial_eval["loss"]

    def run_federated_round(
        self,
        epochs: int = 1,
        lr: float = 0.008,
        dp_enabled: bool = True,
        simulate_attack: bool = False,
        attack_type: str = "sign_flip"
    ) -> Dict[str, Any]:
        """
        Executes a complete decentralized federated round:
        1. Local Client Training (DP-SGD)
        2. Cryptographic Fingerprinting & ECDSA Signing
        3. Smart Contract Verification & Byzantine Anomaly Filtering
        4. Deterministic Aggregation (FedAvg)
        5. Global Evaluation (15 Classes)
        6. Blockchain Block Sealing & Ledger Appending
        """
        self.current_round += 1
        round_start_time = time.time()
        global_weights = self.global_model.get_flat_parameters()
        base_model_hash = hash_weights(global_weights)

        client_updates = {}
        client_sample_weights = {}
        client_signatures = {}
        client_weight_hashes = {}
        client_train_losses = {}

        # 1. Tier 1: Client Local Training
        for cid, client in self.clients.items():
            updated_w, loss = client.local_train(
                global_flat_weights=global_weights,
                epochs=epochs,
                lr=lr,
                dp_enabled=dp_enabled
            )
            w_hash = hash_weights(updated_w)
            sig = client.sign_update(w_hash)

            client_updates[cid] = updated_w
            client_sample_weights[cid] = client.sample_count
            client_signatures[cid] = sig
            client_weight_hashes[cid] = w_hash
            client_train_losses[cid] = float(round(loss, 4))

        # 2. Inject Simulated Attack if enabled
        simulated_attacker_id = "Adversary Node X"
        if simulate_attack:
            # Create poisoned update
            attacker_key_mgr = NodeKeyManager("adversary_x")
            poisoned_w = AdversarialAttackSimulator.apply_attack(
                client_updates["Hospital A"],
                attack_type=attack_type,
                intensity=4.0
            )
            p_hash = hash_weights(poisoned_w)
            p_sig = attacker_key_mgr.sign_payload(p_hash)

            client_updates[simulated_attacker_id] = poisoned_w
            client_sample_weights[simulated_attacker_id] = 1000
            client_signatures[simulated_attacker_id] = p_sig
            client_weight_hashes[simulated_attacker_id] = p_hash
            client_train_losses[simulated_attacker_id] = 9.99

        # 3. Tier 2: Cryptographic Signature Verification
        verified_clients = {}
        for cid, w in client_updates.items():
            w_hash = client_weight_hashes[cid]
            sig = client_signatures[cid]

            if cid in self.clients:
                pub_key = self.clients[cid].key_manager.get_public_key_hex()
                is_valid = NodeKeyManager.verify_signature(pub_key, w_hash, sig)
            else:
                # Malicious or unregistered node
                is_valid = True # Passes signature format, must be caught by Byzantine Shield

            if is_valid:
                verified_clients[cid] = w

        # 4. Tier 2: Byzantine Anomaly Filtering & FedAvg
        aggregated_weights, accepted_cids, rejected_cids, byz_metrics = (
            self.byzantine_engine.filter_and_aggregate(verified_clients, client_sample_weights)
        )

        # 5. Update Global Model
        self.global_model.set_flat_parameters(aggregated_weights)
        updated_global_model_hash = hash_weights(aggregated_weights)

        # 6. Evaluate Global Model on CICIDS2017 Test Set
        eval_metrics = evaluate_model(self.global_model, self.test_loader, self.device)

        # 7. Tier 3: Create Blockchain Transactions & Block
        transactions: List[BlockchainTransaction] = []
        for cid in client_updates.keys():
            status = "ACCEPTED" if cid in accepted_cids else "SLASHED_BYZANTINE"
            score = byz_metrics.get("scores", {}).get(cid, 1.0)
            tx = BlockchainTransaction(
                round_number=self.current_round,
                client_id=cid,
                weight_hash=client_weight_hashes[cid],
                signature=client_signatures[cid][:24] + "...",
                sample_count=client_sample_weights[cid],
                status=status,
                anomaly_score=score,
                tx_type="ADVERSARIAL_ATTACK_REJECTED" if status == "SLASHED_BYZANTINE" else "WEIGHT_COMMITMENT"
            )
            transactions.append(tx)

        # Mine new block
        round_metrics = {
            "accuracy": eval_metrics["accuracy"],
            "loss": eval_metrics["loss"],
            "f1_score": eval_metrics["f1_score"],
            "precision": eval_metrics["precision"],
            "recall": eval_metrics["recall"],
            "accepted_participants": len(accepted_cids),
            "rejected_participants": len(rejected_cids),
            "dp_enabled": dp_enabled
        }
        new_block = self.ledger.add_round_block(
            round_number=self.current_round,
            transactions=transactions,
            global_model_hash=updated_global_model_hash,
            metrics=round_metrics
        )

        # Differential privacy budget summary
        privacy_status = self.clients["Hospital A"].privacy_engine.compute_privacy_spent()

        round_summary = {
            "round_number": self.current_round,
            "duration_sec": round(time.time() - round_start_time, 2),
            "base_model_hash": base_model_hash,
            "updated_model_hash": updated_global_model_hash,
            "metrics": eval_metrics,
            "byzantine_defense": {
                "method": byz_metrics.get("method"),
                "accepted_clients": accepted_cids,
                "rejected_clients": rejected_cids,
                "anomaly_scores": byz_metrics.get("scores")
            },
            "client_losses": client_train_losses,
            "privacy_accounting": privacy_status,
            "block": new_block.to_dict(),
            "ledger_verified": self.ledger.verify_ledger_integrity()["is_valid"]
        }

        self.round_history.append(round_summary)
        return round_summary

    def get_system_state(self) -> Dict[str, Any]:
        """Returns snapshot of current orchestrator state, clients, and blockchain"""
        latest_eval = evaluate_model(self.global_model, self.test_loader, self.device)
        return {
            "current_round": self.current_round,
            "clients": {
                cid: {
                    "name": c.name,
                    "org_type": c.org_type,
                    "sample_count": c.sample_count,
                    "attack_profile": c.attack_profile,
                    "last_loss": c.last_train_loss,
                    "rounds_participated": c.total_rounds_participated,
                    "public_key_preview": c.key_manager.get_public_key_hex()[:24] + "..."
                }
                for cid, c in self.clients.items()
            },
            "metrics": latest_eval,
            "total_blocks": len(self.ledger.chain),
            "latest_block": self.ledger.get_latest_block().to_dict(),
            "chain_validity": self.ledger.verify_ledger_integrity(),
            "classes": CICIDS_CLASSES
        }
