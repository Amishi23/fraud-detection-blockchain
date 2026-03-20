"""
Fraud Detection Blockchain — API
─────────────────────────────────
FastAPI backend wiring the ML detector and blockchain together.
"""

import time
import random
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from blockchain import Blockchain
from detector  import FraudDetector, TransactionGenerator

app = FastAPI(title="Fraud Detection Blockchain API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup ───────────────────────────────────────────────────────────────────

print("[STARTUP] Training fraud detection model...")
detector   = FraudDetector()
blockchain = Blockchain()
generator  = TransactionGenerator()
print("[STARTUP] Ready.")

# ── Request/Response models ───────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    amount:                  float = Field(..., gt=0, example=249.99)
    hour:                    int   = Field(..., ge=0, le=23, example=14)
    day_of_week:             int   = Field(..., ge=0, le=6, example=2)
    merchant_category:       str   = Field(..., example="online_retail")
    location:                str   = Field(..., example="domestic")
    transactions_last_hour:  int   = Field(0, ge=0, example=1)
    amount_deviation:        float = Field(0.0, example=0.5)
    card_present:            int   = Field(1, ge=0, le=1, example=1)
    is_recurring:            int   = Field(0, ge=0, le=1, example=0)

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": time.time(), "chain_length": len(blockchain.chain)}


@app.post("/api/transaction/analyze")
def analyze_transaction(req: TransactionRequest):
    """
    Submit a transaction for fraud analysis.
    Result is recorded on the blockchain regardless of outcome.
    """
    transaction = req.dict()
    transaction["transaction_id"] = str(uuid.uuid4())[:8].upper()
    transaction["submitted_at"]   = time.time()

    # Run ML model
    result = detector.predict(transaction)

    fraud_result = {
        "is_fraud":   result.is_fraud,
        "confidence": result.confidence,
        "risk_level": result.risk_level,
        "risk_score": result.risk_score,
        "flags":      result.flags,
        "model":      result.model_version,
    }

    # Record on blockchain
    block = blockchain.add_block(transaction, fraud_result)

    return {
        "transaction_id": transaction["transaction_id"],
        "block_index":    block.index,
        "block_hash":     block.hash[:16] + "...",
        "fraud_result":   fraud_result,
        "mined_in_ms":    round((time.time() - transaction["submitted_at"]) * 1000, 1),
    }


@app.get("/api/blockchain")
def get_blockchain():
    """Return the full blockchain."""
    return {
        "chain": blockchain.to_list(),
        "stats": blockchain.get_stats(),
    }


@app.get("/api/blockchain/validate")
def validate_chain():
    """Cryptographically validate the entire chain."""
    valid, reason = blockchain.is_valid()
    return {
        "is_valid":    valid,
        "reason":      reason,
        "chain_length": len(blockchain.chain),
        "checked_at":  time.time(),
    }


@app.get("/api/blockchain/fraud-alerts")
def get_fraud_alerts():
    """Return only fraud blocks from the chain."""
    return {
        "fraud_blocks": blockchain.get_fraud_blocks(),
        "total":        len(blockchain.get_fraud_blocks()),
    }


@app.get("/api/model/metrics")
def get_model_metrics():
    """Return ML model performance metrics."""
    return detector.get_metrics()


@app.post("/api/simulate")
def simulate_transactions(count: int = 20):
    """
    Auto-simulate a batch of transactions (mix of fraud and legitimate).
    Great for demo/testing purposes.
    """
    results = []
    for _ in range(count):
        # 10% chance of fraud in simulation
        is_fraud_sim = random.random() < 0.10
        raw = generator.generate_fraud() if is_fraud_sim else generator.generate_legitimate()

        req = TransactionRequest(**raw)
        result = analyze_transaction(req)
        results.append(result)

    stats = blockchain.get_stats()
    return {
        "simulated": count,
        "results":   results,
        "chain_stats": stats,
    }


@app.get("/api/stats")
def get_stats():
    return {
        "blockchain": blockchain.get_stats(),
        "model":      detector.get_metrics(),
    }
