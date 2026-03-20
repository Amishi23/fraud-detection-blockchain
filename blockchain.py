"""
Blockchain — Built from Scratch
SHA-256 hashing, Proof of Work, Chain validation, Tamper detection
"""

import hashlib
import json
import time
from dataclasses import dataclass, field


def make_serializable(obj):
    """Recursively convert numpy/bool types to native Python types."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif type(obj).__name__ in ('bool_', 'bool8'):
        return bool(obj)
    elif type(obj).__name__ in ('int64', 'int32', 'int16', 'int8'):
        return int(obj)
    elif type(obj).__name__ in ('float64', 'float32', 'float16'):
        return float(obj)
    elif isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, (int, float, str)) or obj is None:
        return obj
    else:
        return str(obj)


@dataclass
class Block:
    index:          int
    timestamp:      float
    transaction:    dict
    fraud_result:   dict
    previous_hash:  str
    nonce:          int  = 0
    hash:           str  = field(default="", init=False)

    def __post_init__(self):
        # Sanitize inputs
        self.transaction  = make_serializable(self.transaction)
        self.fraud_result = make_serializable(self.fraud_result)
        self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        block_data = {
            "index":         self.index,
            "timestamp":     self.timestamp,
            "transaction":   self.transaction,
            "fraud_result":  self.fraud_result,
            "previous_hash": self.previous_hash,
            "nonce":         self.nonce,
        }
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine(self, difficulty: int):
        target = "0" * difficulty
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.compute_hash()

    def to_dict(self) -> dict:
        return {
            "index":         self.index,
            "timestamp":     self.timestamp,
            "transaction":   self.transaction,
            "fraud_result":  self.fraud_result,
            "previous_hash": self.previous_hash,
            "nonce":         self.nonce,
            "hash":          self.hash,
        }


class Blockchain:
    DIFFICULTY = 3

    def __init__(self):
        self.chain: list[Block] = []
        self._create_genesis_block()

    def _create_genesis_block(self):
        genesis = Block(
            index=0,
            timestamp=time.time(),
            transaction={"type": "GENESIS", "message": "Fraud Detection Chain Initialized"},
            fraud_result={"is_fraud": False, "confidence": 0.0, "risk_level": "none"},
            previous_hash="0" * 64,
        )
        genesis.mine(self.DIFFICULTY)
        self.chain.append(genesis)

    @property
    def last_block(self) -> Block:
        return self.chain[-1]

    def add_block(self, transaction: dict, fraud_result: dict) -> Block:
        block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            transaction=transaction,
            fraud_result=fraud_result,
            previous_hash=self.last_block.hash,
        )
        block.mine(self.DIFFICULTY)
        self.chain.append(block)
        return block

    def is_valid(self) -> tuple[bool, str]:
        for i in range(1, len(self.chain)):
            current  = self.chain[i]
            previous = self.chain[i - 1]
            if current.hash != current.compute_hash():
                return False, f"Block {i} hash is corrupted"
            if current.previous_hash != previous.hash:
                return False, f"Block {i} is not linked to block {i-1}"
        return True, "Chain is valid"

    def get_fraud_blocks(self) -> list[dict]:
        return [
            b.to_dict() for b in self.chain[1:]
            if b.fraud_result.get("is_fraud")
        ]

    def get_stats(self) -> dict:
        blocks = self.chain[1:]
        fraud_blocks = [b for b in blocks if b.fraud_result.get("is_fraud")]
        total_amount = sum(b.transaction.get("amount", 0) for b in blocks)
        fraud_amount = sum(b.transaction.get("amount", 0) for b in fraud_blocks)
        valid, _ = self.is_valid()

        return {
            "total_transactions": len(blocks),
            "total_fraud":        len(fraud_blocks),
            "total_legitimate":   len(blocks) - len(fraud_blocks),
            "fraud_rate":         round(len(fraud_blocks) / max(len(blocks), 1) * 100, 1),
            "total_amount":       round(float(total_amount), 2),
            "fraud_amount":       round(float(fraud_amount), 2),
            "chain_length":       len(self.chain),
            "chain_valid":        bool(valid),
            "difficulty":         self.DIFFICULTY,
        }

    def to_list(self) -> list[dict]:
        return [b.to_dict() for b in self.chain]
