"""
FastAPI Backend Server for Blockchain-Enabled Federated AI Framework
Serves:
- REST API for training orchestration, client node telemetry, and blockchain queries
- Live round execution with DP-SGD, ECDSA verification, and Byzantine anomaly defense
- Blockchain explorer endpoints with real-time tamper-detection
- Static frontend web dashboard
"""

import os
import json
import time
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from core.federated_engine import FederatedOrchestrator
from core.dataset import CICIDS_CLASSES

app = FastAPI(
    title="Blockchain-Enabled Federated AI Framework",
    description="Decentralized Cybersecurity Threat Intelligence over CICIDS2017 Benchmark",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Orchestrator instance
orchestrator = FederatedOrchestrator(samples_per_client=1000)

class TrainRoundRequest(BaseModel):
    epochs: int = 1
    lr: float = 0.008
    dp_enabled: bool = True
    simulate_attack: bool = False
    attack_type: str = "sign_flip" # sign_flip, gaussian_noise, backdoor
    byzantine_method: str = "cosine_shield" # cosine_shield, multi_krum, trimmed_mean

@app.get("/api/status")
def get_status():
    """Returns current system architecture state, clients, and blockchain metrics"""
    state = orchestrator.get_system_state()
    return {
        "success": True,
        "data": state
    }

@app.post("/api/train-round")
def run_round(req: TrainRoundRequest):
    """Executes a single federated learning round governed by smart contract rules"""
    try:
        orchestrator.byzantine_engine.method = req.byzantine_method
        round_result = orchestrator.run_federated_round(
            epochs=req.epochs,
            lr=req.lr,
            dp_enabled=req.dp_enabled,
            simulate_attack=req.simulate_attack,
            attack_type=req.attack_type
        )
        return {
            "success": True,
            "round": round_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/blockchain")
def get_blockchain():
    """Returns the full permissioned ledger of blocks and transactions"""
    blocks = orchestrator.ledger.export_ledger_json()
    validity = orchestrator.ledger.verify_ledger_integrity()
    return {
        "success": True,
        "total_blocks": len(blocks),
        "chain_valid": validity["is_valid"],
        "validation_report": validity,
        "blocks": blocks
    }

@app.post("/api/blockchain/verify")
def verify_blockchain():
    """Validates every block, previous hash, Merkle root, and digital signature"""
    report = orchestrator.ledger.verify_ledger_integrity()
    return {
        "success": True,
        "report": report
    }

@app.post("/api/blockchain/tamper-test")
def simulate_tamper():
    """
    Demonstrates tamper-evidence by altering data in a mined block
    to prove cryptographic consensus detection.
    """
    if len(orchestrator.ledger.chain) <= 1:
        raise HTTPException(status_code=400, detail="Run at least 1 federated round before testing tamper detection.")

    # Tamper with block 1's previous hash
    target_block = orchestrator.ledger.chain[1]
    target_block.previous_hash = "000000000000000000TAMPERED_HASH_ALERT"
    verification = orchestrator.ledger.verify_ledger_integrity()
    return {
        "success": True,
        "message": "Simulated tamper payload injected into Block #1",
        "verification_result": verification
    }

@app.post("/api/reset")
def reset_system():
    """Resets the orchestrator, clearing models and rebuilding genesis ledger"""
    global orchestrator
    orchestrator = FederatedOrchestrator(samples_per_client=1000)
    return {
        "success": True,
        "message": "Consortium and Ledger re-initialized to Genesis State"
    }

@app.get("/api/confusion-matrix")
def get_confusion_matrix():
    """Returns the multi-class confusion matrix on the CICIDS2017 15-class test set"""
    state = orchestrator.get_system_state()
    return {
        "success": True,
        "classes": CICIDS_CLASSES,
        "metrics": state["metrics"]
    }

@app.get("/api/export-audit")
def export_audit_report():
    """Exports an enterprise forensic audit log for compliance (GDPR, HIPAA, PCI-DSS, SOC2)"""
    state = orchestrator.get_system_state()
    ledger_history = orchestrator.ledger.export_ledger_json()
    validity = orchestrator.ledger.verify_ledger_integrity()

    audit_payload = {
        "consortium_name": "Distributed Cybersecurity Intelligence Consortium",
        "benchmark_dataset": "CICIDS2017 (Canadian Institute for Cybersecurity)",
        "report_generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_rounds_completed": orchestrator.current_round,
        "ledger_cryptographically_valid": validity["is_valid"],
        "validation_details": validity,
        "participating_silos": state["clients"],
        "final_model_performance": state["metrics"],
        "blockchain_audit_blocks": ledger_history
    }
    return JSONResponse(
        content=audit_payload,
        headers={"Content-Disposition": "attachment; filename=consortium_audit_report.json"}
    )

# Static Frontend mounting
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))
