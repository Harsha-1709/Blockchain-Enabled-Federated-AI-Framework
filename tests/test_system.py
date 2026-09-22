"""
Comprehensive Unit & Integration Test Suite
Validates:
1. Dataset generation and Non-IID client distributions
2. PyTorch DNN model parameter setting, retrieval, and forward pass
3. Differential privacy (DP-SGD) gradient clipping and noise injection
4. Cryptographic signatures, SHA-256 weight fingerprinting, and Merkle tree root
5. Byzantine fault tolerance & adversarial attack rejection
6. Blockchain ledger integrity and tamper detection
7. End-to-end federated training round
"""

import unittest
import numpy as np
import torch
import copy

from core.dataset import CICIDSDataEngine, NUM_FEATURES, NUM_CLASSES
from core.model import IntrusionDetectionDNN, evaluate_model
from core.privacy import DifferentialPrivacyEngine
from core.crypto import NodeKeyManager, hash_weights, compute_merkle_root, hash_sha256
from core.byzantine import ByzantineDefenseEngine, AdversarialAttackSimulator
from core.blockchain import PermissionedAuditLedger, BlockchainTransaction
from core.federated_engine import FederatedOrchestrator

class TestBlockchainFederatedAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_engine = CICIDSDataEngine(random_seed=42)

    def test_01_dataset_and_partitioning(self):
        client_loaders, test_loader, stats = self.data_engine.get_data_loaders(
            batch_size=32, samples_per_client=100
        )
        self.assertIn("Hospital A", client_loaders)
        self.assertIn("Bank B", client_loaders)
        self.assertIn("IoT Node C", client_loaders)

        # Check sample batches
        for cid, loader in client_loaders.items():
            features, labels = next(iter(loader))
            self.assertEqual(features.shape[1], NUM_FEATURES)
            self.assertTrue(len(labels) <= 32)
            self.assertTrue(all(0 <= l < NUM_CLASSES for l in labels))

    def test_02_model_architecture_and_eval(self):
        model = IntrusionDetectionDNN(NUM_FEATURES, NUM_CLASSES)
        flat = model.get_flat_parameters()
        self.assertGreater(len(flat), 1000)

        # Perturb flat weights and re-set
        flat_modified = flat + 0.05
        model.set_flat_parameters(flat_modified)
        retrieved = model.get_flat_parameters()
        np.testing.assert_allclose(flat_modified, retrieved, atol=1e-5)

        # Forward pass
        dummy_input = torch.randn(10, NUM_FEATURES)
        out = model(dummy_input)
        self.assertEqual(out.shape, (10, NUM_CLASSES))

    def test_03_differential_privacy_engine(self):
        model = IntrusionDetectionDNN(NUM_FEATURES, NUM_CLASSES)
        dp = DifferentialPrivacyEngine(clip_norm=1.0, noise_multiplier=0.5, enabled=True)

        # Synthesize gradients
        dummy_input = torch.randn(16, NUM_FEATURES)
        dummy_targets = torch.randint(0, NUM_CLASSES, (16,))
        criterion = torch.nn.CrossEntropyLoss()

        out = model(dummy_input)
        loss = criterion(out, dummy_targets)
        loss.backward()

        initial_norm = dp.clip_and_add_noise(model, batch_size=16, dataset_size=100)
        self.assertGreater(initial_norm, 0.0)

        privacy_spent = dp.compute_privacy_spent()
        self.assertIn("epsilon", privacy_spent)
        self.assertGreater(privacy_spent["epsilon"], 0.0)

    def test_04_crypto_and_merkle(self):
        node = NodeKeyManager("test_node")
        pub_hex = node.get_public_key_hex()

        msg = "0x4a591f0d7aa_weight_hash_verification"
        sig = node.sign_payload(msg)

        # Verify valid signature
        is_valid = NodeKeyManager.verify_signature(pub_hex, msg, sig)
        self.assertTrue(is_valid)

        # Verify altered message fails
        self.assertFalse(NodeKeyManager.verify_signature(pub_hex, msg + "_corrupted", sig))

        # Test Merkle Root
        leaves = [hash_sha256(f"leaf_{i}".encode()) for i in range(4)]
        merkle_root = compute_merkle_root(leaves)
        self.assertEqual(len(merkle_root), 64)

    def test_05_byzantine_defense(self):
        engine = ByzantineDefenseEngine(method="cosine_shield")

        # Create benign updates with similar direction
        base = np.random.randn(500)
        u1 = base + np.random.randn(500) * 0.1
        u2 = base + np.random.randn(500) * 0.1
        u3 = base + np.random.randn(500) * 0.1

        # Poisoned update (sign flipping & massive magnitude)
        u_malicious = AdversarialAttackSimulator.apply_attack(base, attack_type="sign_flip", intensity=4.0)

        updates = {
            "Hospital A": u1,
            "Bank B": u2,
            "IoT Node C": u3,
            "Attacker X": u_malicious
        }
        weights = {"Hospital A": 100, "Bank B": 100, "IoT Node C": 100, "Attacker X": 100}

        agg, accepted, rejected, metrics = engine.filter_and_aggregate(updates, weights)

        self.assertIn("Attacker X", rejected)
        self.assertIn("Hospital A", accepted)
        self.assertIn("Bank B", accepted)
        self.assertIn("IoT Node C", accepted)

    def test_06_blockchain_ledger_and_tamper_detection(self):
        ledger = PermissionedAuditLedger("Validator_A")

        # Initial genesis block
        self.assertEqual(len(ledger.chain), 1)
        self.assertTrue(ledger.verify_ledger_integrity()["is_valid"])

        # Add round 1 block
        tx = BlockchainTransaction(
            round_number=1,
            client_id="Hospital A",
            weight_hash=hash_sha256(b"w1"),
            signature="sig1",
            sample_count=500
        )
        block1 = ledger.add_round_block(
            round_number=1,
            transactions=[tx],
            global_model_hash=hash_sha256(b"global_1"),
            metrics={"accuracy": 75.0, "loss": 0.4}
        )
        self.assertEqual(len(ledger.chain), 2)
        self.assertTrue(ledger.verify_ledger_integrity()["is_valid"])

        # Simulate tampering with block 1 previous hash
        block1.previous_hash = "tampered_fake_hash_1234567890abcdef"
        tamper_check = ledger.verify_ledger_integrity()
        self.assertFalse(tamper_check["is_valid"])

    def test_07_end_to_end_orchestrator_round(self):
        orchestrator = FederatedOrchestrator(samples_per_client=100)
        round_res = orchestrator.run_federated_round(
            epochs=1,
            lr=0.01,
            dp_enabled=True,
            simulate_attack=True,
            attack_type="sign_flip"
        )
        self.assertEqual(round_res["round_number"], 1)
        self.assertIn("Adversary Node X", round_res["byzantine_defense"]["rejected_clients"])
        self.assertTrue(round_res["ledger_verified"])

if __name__ == "__main__":
    unittest.main()
