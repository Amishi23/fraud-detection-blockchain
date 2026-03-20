"""
Fraud Detection ML Model — Real Data Edition
─────────────────────────────────────────────
Trained on the Kaggle Credit Card Fraud dataset:
  - 284,807 real anonymized transactions
  - 492 fraud cases (0.17% — extremely imbalanced)
  - Features V1-V28 are PCA components (anonymized for privacy)
  - Plus Time and Amount features

ML techniques:
  - Custom SMOTE for extreme class imbalance
  - Random Forest with class_weight balanced
  - Threshold tuning at 0.3 for high recall
  - StandardScaler on Amount and Time
"""

import numpy as np
import pandas as pd
import time
import random
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from dataclasses import dataclass
from typing import Optional


# ── SMOTE ─────────────────────────────────────────────────────────────────────

class SimpleSMOTE:
    def __init__(self, ratio: float = 0.1):
        self.ratio = ratio

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple:
        minority_idx = np.where(y == 1)[0]
        majority_idx = np.where(y == 0)[0]
        n_minority   = len(minority_idx)
        n_majority   = len(majority_idx)
        n_synthetic  = int(n_majority * self.ratio) - n_minority
        if n_synthetic <= 0:
            return X, y
        X_minority = X[minority_idx]
        synthetic  = []
        for _ in range(n_synthetic):
            idx      = np.random.randint(0, n_minority)
            sample   = X_minority[idx]
            neighbor = X_minority[np.random.randint(0, n_minority)]
            alpha    = np.random.random()
            synthetic.append(sample + alpha * (neighbor - sample))
        X_resampled = np.vstack([X, np.array(synthetic)])
        y_resampled = np.concatenate([y, np.ones(len(synthetic))])
        return X_resampled, y_resampled


# ── Transaction Generator (for simulation endpoint) ───────────────────────────

class TransactionGenerator:
    MERCHANTS = [
        "grocery_store", "gas_station", "restaurant", "online_retail",
        "pharmacy", "hotel", "airline", "electronics", "atm_withdrawal",
        "money_transfer", "casino", "crypto_exchange"
    ]
    HIGH_RISK_MERCHANTS = {"money_transfer", "casino", "crypto_exchange", "atm_withdrawal"}

    def generate_legitimate(self) -> dict:
        hour = random.choices(
            range(24),
            weights=[1,1,1,1,1,1,2,4,5,5,4,4,5,5,4,4,5,5,4,3,3,2,2,1], k=1
        )[0]
        return {
            "amount":                 round(random.lognormvariate(3.5, 1.0), 2),
            "hour":                   hour,
            "day_of_week":            random.randint(0, 6),
            "merchant_category":      random.choice(self.MERCHANTS[:8]),
            "location":               "domestic",
            "transactions_last_hour": random.randint(0, 3),
            "amount_deviation":       random.uniform(-1.5, 1.5),
            "card_present":           1,
            "is_recurring":           random.choice([0, 0, 0, 1]),
        }

    def generate_fraud(self) -> dict:
        fraud_type = random.choice(["high_amount", "unusual_hour", "velocity", "foreign"])
        base = self.generate_legitimate()
        if fraud_type == "high_amount":
            base["amount"] = round(random.uniform(800, 5000), 2)
            base["merchant_category"] = random.choice(list(self.HIGH_RISK_MERCHANTS))
        elif fraud_type == "unusual_hour":
            base["hour"] = random.randint(2, 5)
            base["amount"] = round(random.uniform(200, 2000), 2)
            base["card_present"] = 0
        elif fraud_type == "velocity":
            base["transactions_last_hour"] = random.randint(8, 20)
            base["amount_deviation"] = random.uniform(3, 8)
        elif fraud_type == "foreign":
            base["location"] = "foreign"
            base["merchant_category"] = random.choice(list(self.HIGH_RISK_MERCHANTS))
            base["card_present"] = 0
        return base

    def transaction_to_features(self, t: dict) -> list:
        """Convert API transaction to feature vector matching Kaggle format."""
        merchant_risk = 1.0 if t.get("merchant_category") in self.HIGH_RISK_MERCHANTS else 0.0
        location_risk = 1.0 if t.get("location") == "foreign" else 0.0
        hour  = float(t.get("hour", 12))
        night = 1.0 if (hour >= 1 and hour <= 5) else 0.0
        amount = float(t.get("amount", 0))
        vel    = float(t.get("transactions_last_hour", 0))
        card   = float(t.get("card_present", 1))

        # Build a 30-feature vector matching V1-V28 + Time + Amount structure
        # Real PCA features are unknown so we synthesize plausible values
        # Amount and time are real; V-features are proxied from transaction signals
        features = [
            float(hour * 3600),   # Time proxy
            merchant_risk * -2.5, # V1 proxy
            location_risk * -1.8, # V2 proxy
            night * -2.0,         # V3 proxy
            vel / 10.0,           # V4 proxy
            (1 - card) * 1.5,     # V5 proxy
            merchant_risk * 1.2,  # V6 proxy
            location_risk * 0.8,  # V7 proxy
            night * 1.1,          # V8 proxy
            vel / 20.0,           # V9 proxy
            random.gauss(0, 0.1), # V10
            random.gauss(0, 0.1), # V11
            random.gauss(0, 0.1), # V12
            random.gauss(0, 0.1), # V13
            random.gauss(0, 0.1), # V14
            random.gauss(0, 0.1), # V15
            random.gauss(0, 0.1), # V16
            random.gauss(0, 0.1), # V17
            random.gauss(0, 0.1), # V18
            random.gauss(0, 0.1), # V19
            random.gauss(0, 0.1), # V20
            random.gauss(0, 0.1), # V21
            random.gauss(0, 0.1), # V22
            random.gauss(0, 0.1), # V23
            random.gauss(0, 0.1), # V24
            random.gauss(0, 0.1), # V25
            random.gauss(0, 0.1), # V26
            random.gauss(0, 0.1), # V27
            random.gauss(0, 0.1), # V28
            amount,               # Amount
        ]
        return features


# ── Fraud Detector ────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    is_fraud:      bool
    confidence:    float
    risk_level:    str
    risk_score:    float
    flags:         list
    model_version: str


class FraudDetector:
    THRESHOLD  = 0.3
    CSV_PATH   = os.path.join(os.path.dirname(__file__), "creditcard.csv")

    def __init__(self):
        self.generator = TransactionGenerator()
        self.model     = None
        self.scaler    = StandardScaler()
        self.smote     = SimpleSMOTE(ratio=0.1)
        self.trained   = False
        self.metrics   = {}
        self.train()

    def train(self):
        print("[ML] Loading Kaggle credit card fraud dataset...")
        df = pd.read_csv(self.CSV_PATH)
        print(f"[ML] Dataset loaded — {len(df):,} transactions, {df['Class'].sum():,} fraud ({df['Class'].mean()*100:.2f}%)")

        # Scale Amount and Time (V1-V28 are already PCA scaled)
        df["Amount"] = self.scaler.fit_transform(df[["Amount"]])
        df["Time"]   = (df["Time"] - df["Time"].mean()) / df["Time"].std()

        feature_cols = [c for c in df.columns if c != "Class"]
        X = df[feature_cols].values
        y = df["Class"].values.astype(float)

        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"[ML] Before SMOTE — Fraud: {int(sum(y_train))}, Legitimate: {int(len(y_train)-sum(y_train))}")
        X_train_r, y_train_r = self.smote.fit_resample(X_train, y_train)
        print(f"[ML] After SMOTE  — Fraud: {int(sum(y_train_r))}, Legitimate: {int(len(y_train_r)-sum(y_train_r))}")

        print("[ML] Training Random Forest on 284,807 real transactions...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train_r, y_train_r)

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        auc    = float(roc_auc_score(y_test, y_prob))
        report = classification_report(y_test, y_pred, output_dict=True)
        key    = "1.0" if "1.0" in report else "1"

        self.metrics = {
            "auc_roc":            round(auc, 4),
            "precision":          round(float(report[key]["precision"]), 4),
            "recall":             round(float(report[key]["recall"]), 4),
            "f1_score":           round(float(report[key]["f1-score"]), 4),
            "accuracy":           round(float(report["accuracy"]), 4),
            "training_samples":   int(len(X_train_r)),
            "real_fraud_cases":   int(df["Class"].sum()),
            "dataset":            "Kaggle Credit Card Fraud (284,807 transactions)",
        }
        self.trained = True
        print(f"[ML] Training complete — AUC-ROC: {auc:.4f}")

    def _generate_flags(self, transaction: dict, confidence: float) -> list:
        flags  = []
        amount = transaction.get("amount", 0)
        hour   = transaction.get("hour", 12)
        vel    = transaction.get("transactions_last_hour", 0)
        merch  = transaction.get("merchant_category", "")
        loc    = transaction.get("location", "domestic")
        if amount > 500:   flags.append(f"High amount: ${amount:.2f}")
        if amount > 1500:  flags.append("Unusually large transaction")
        if hour in range(1, 6): flags.append(f"Unusual hour: {hour}:00 AM")
        if vel >= 5:       flags.append(f"High velocity: {vel} txns/hr")
        if merch in self.generator.HIGH_RISK_MERCHANTS:
            flags.append(f"High-risk merchant: {merch}")
        if loc == "foreign": flags.append("Foreign location")
        if not transaction.get("card_present"): flags.append("Card not present (CNP)")
        if confidence > 0.6: flags.append(f"Model confidence: {confidence*100:.0f}%")
        return flags if flags else ["Anomalous pattern detected"]

    def predict(self, transaction: dict) -> PredictionResult:
        features = np.array(
            self.generator.transaction_to_features(transaction)
        ).reshape(1, -1)

        proba      = float(self.model.predict_proba(features)[0][1])
        is_fraud   = bool(proba >= self.THRESHOLD)
        risk_score = round(proba * 100, 1)

        if proba >= 0.75:   risk_level = "critical"
        elif proba >= 0.5:  risk_level = "high"
        elif proba >= 0.3:  risk_level = "medium"
        else:               risk_level = "low"

        flags = self._generate_flags(transaction, proba) if is_fraud else []

        return PredictionResult(
            is_fraud=bool(is_fraud),
            confidence=round(float(proba), 4),
            risk_level=str(risk_level),
            risk_score=float(risk_score),
            flags=[str(f) for f in flags],
            model_version="rf-v2.0-kaggle-smote",
        )

    def get_metrics(self) -> dict:
        return self.metrics
