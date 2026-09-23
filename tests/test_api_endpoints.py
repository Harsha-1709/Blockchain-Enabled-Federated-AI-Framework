"""
End-to-End API Integration Test Suite
Validates all FastAPI REST endpoints, static mounting, tamper detection,
adversarial attack injection, and forensic JSON audit export.
"""

import unittest
from fastapi.testclient import TestClient
from server.app import app

class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_root_and_static_files(self):
        # 1. Test index page
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("Blockchain-Enabled", res.text)
        self.assertIn("Federated AI Framework", res.text)

        # 2. Test static CSS
        res_css = self.client.get("/static/styles.css")
        self.assertEqual(res_css.status_code, 200)
        self.assertIn("glassmorphism", res_css.text)

        # 3. Test static JS
        res_js = self.client.get("/static/app.js")
        self.assertEqual(res_js.status_code, 200)
        self.assertIn("initChart", res_js.text)

    def test_02_status_endpoint(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertIn("Hospital A", data["data"]["clients"])
        self.assertIn("Bank B", data["data"]["clients"])
        self.assertIn("IoT Node C", data["data"]["clients"])
        self.assertEqual(len(data["data"]["classes"]), 15)

    def test_03_train_clean_round(self):
        payload = {
            "epochs": 1,
            "lr": 0.01,
            "dp_enabled": True,
            "simulate_attack": False,
            "byzantine_method": "cosine_shield"
        }
        res = self.client.post("/api/train-round", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        round_data = data["round"]
        self.assertGreaterEqual(round_data["round_number"], 1)
        self.assertTrue(round_data["ledger_verified"])
        self.assertEqual(len(round_data["byzantine_defense"]["accepted_clients"]), 3)
        self.assertEqual(len(round_data["byzantine_defense"]["rejected_clients"]), 0)

    def test_04_train_adversarial_round(self):
        payload = {
            "epochs": 1,
            "lr": 0.01,
            "dp_enabled": True,
            "simulate_attack": True,
            "attack_type": "sign_flip",
            "byzantine_method": "cosine_shield"
        }
        res = self.client.post("/api/train-round", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        round_data = data["round"]
        self.assertIn("Adversary Node X", round_data["byzantine_defense"]["rejected_clients"])
        self.assertTrue(round_data["ledger_verified"])

    def test_05_blockchain_query_and_verify(self):
        # Query chain
        res = self.client.get("/api/blockchain")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertGreaterEqual(data["total_blocks"], 3) # Genesis + Round 1 + Round 2
        self.assertTrue(data["chain_valid"])

        # Cryptographic verification endpoint
        res_v = self.client.post("/api/blockchain/verify")
        self.assertEqual(res_v.status_code, 200)
        data_v = res_v.json()
        self.assertTrue(data_v["success"])
        self.assertTrue(data_v["report"]["is_valid"])

    def test_06_tamper_detection(self):
        # Inject artificial tamper into Block 1
        res = self.client.post("/api/blockchain/tamper-test")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertFalse(data["verification_result"]["is_valid"])
        self.assertEqual(data["verification_result"]["tampered_block_index"], 1)

    def test_07_confusion_matrix_and_audit_export(self):
        # Reset system to restore valid chain first
        res_reset = self.client.post("/api/reset")
        self.assertEqual(res_reset.status_code, 200)
        self.assertTrue(res_reset.json()["success"])

        # Train 1 round to have populated metrics
        self.client.post("/api/train-round", json={"epochs": 1, "lr": 0.01})

        # Test confusion matrix endpoint
        res_cm = self.client.get("/api/confusion-matrix")
        self.assertEqual(res_cm.status_code, 200)
        cm_data = res_cm.json()
        self.assertTrue(cm_data["success"])
        self.assertEqual(len(cm_data["metrics"]["confusion_matrix"]), 15)

        # Test export audit report
        res_audit = self.client.get("/api/export-audit")
        self.assertEqual(res_audit.status_code, 200)
        audit_data = res_audit.json()
        self.assertEqual(audit_data["consortium_name"], "Distributed Cybersecurity Intelligence Consortium")
        self.assertTrue(audit_data["ledger_cryptographically_valid"])
        self.assertIn("Hospital A", audit_data["participating_silos"])
        self.assertGreaterEqual(len(audit_data["blockchain_audit_blocks"]), 2)

if __name__ == "__main__":
    unittest.main()
