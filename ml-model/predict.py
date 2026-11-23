from pathlib import Path
from typing import Any

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "fraud_model.pkl"
DEFAULT_FEATURES = [
    "amount",
    "device_new",
    "location_unknown",
    "hour",
    "is_night",
    "spending_deviation",
]


def load_fraud_model(model_path: Path = MODEL_PATH) -> tuple[Any, list[str]]:
    """Load trained model and expected feature order from fraud_model.pkl."""
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        feature_cols = artifact.get("feature_cols", DEFAULT_FEATURES)
    else:
        model = artifact
        feature_cols = DEFAULT_FEATURES
    return model, list(feature_cols)


def _extract_hour(transaction: dict[str, Any]) -> int:
    """Extract hour (0-23) from 'hour', 'transaction_time', or 'timestamp'."""
    if "hour" in transaction and transaction["hour"] is not None:
        hour_val = pd.to_numeric(transaction["hour"], errors="coerce")
        if pd.notna(hour_val):
            return int(max(0, min(23, hour_val)))

    for key in ("transaction_time", "timestamp"):
        raw = transaction.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if ":" in text and len(text) <= 8:
            part = text.split(":", 1)[0]
            num = pd.to_numeric(part, errors="coerce")
            if pd.notna(num):
                return int(max(0, min(23, num)))
        ts = pd.to_datetime(text, errors="coerce")
        if pd.notna(ts):
            return int(ts.hour)

    return 0


def build_features(
    transaction: dict[str, Any],
    customer_avg_amount: float | None = None,
) -> dict[str, float | int]:
    """Build engineered features for a single incoming transaction."""
    amount = float(pd.to_numeric(transaction.get("amount", 0.0), errors="coerce") or 0.0)
    hour = _extract_hour(transaction)

    device_val = str(transaction.get("device", "")).strip().lower()
    location_val = str(transaction.get("location", "")).strip().lower()

    device_new = int(bool(transaction.get("device_new")) or device_val == "new")
    location_unknown = int(
        bool(transaction.get("location_unknown")) or location_val == "unknown"
    )
    is_night = int(0 <= hour < 5)

    if customer_avg_amount is not None and customer_avg_amount > 0:
        spending_deviation = amount / float(customer_avg_amount)
    elif transaction.get("spending_deviation") is not None:
        spending_deviation = float(transaction["spending_deviation"])
    else:
        spending_deviation = 1.0

    return {
        "amount": amount,
        "device_new": device_new,
        "location_unknown": location_unknown,
        "hour": hour,
        "is_night": is_night,
        "spending_deviation": spending_deviation,
    }


def score_transaction(
    transaction: dict[str, Any],
    customer_avg_amount: float | None = None,
    model: Any | None = None,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Score a single transaction and map fraud probability to a decision."""
    if model is None or feature_cols is None:
        model, feature_cols = load_fraud_model()

    feats = build_features(transaction, customer_avg_amount=customer_avg_amount)
    row = pd.DataFrame([{col: feats.get(col, 0) for col in feature_cols}])[feature_cols]

    if hasattr(model, "predict_proba"):
        fraud_prob = float(model.predict_proba(row)[0][1])
    else:
        fraud_prob = float(model.predict(row)[0])

    if fraud_prob >= 0.75:
        decision = "BLOCK"
        is_fraud = 1
    elif fraud_prob >= 0.40:
        decision = "REVIEW"
        is_fraud = 1
    else:
        decision = "APPROVE"
        is_fraud = 0

    return {
        "customer_id": transaction.get("customer_id"),
        "amount": feats["amount"],
        "customer_avg_amount": customer_avg_amount,
        "spending_deviation": round(float(feats["spending_deviation"]), 4),
        "fraud_probability": round(fraud_prob, 4),
        "is_fraud": is_fraud,
        "decision": decision,
        "features": feats,
    }


def format_predictions_table(results: list[dict[str, Any]]) -> str:
    """Format scoring results into a clean, easy-to-read column table."""
    if not results:
        return "No transactions scored."

    rows = [
        {
            "Customer ID": r.get("customer_id") or "-",
            "Amount": f"{float(r.get('amount', 0.0)):,.2f}",
            "Avg Spend": (
                f"{float(r['customer_avg_amount']):,.2f}"
                if r.get("customer_avg_amount") is not None
                else "-"
            ),
            "Deviation": f"{float(r.get('spending_deviation', 1.0)):.2f}x",
            "Fraud Risk": f"{float(r.get('fraud_probability', 0.0)) * 100:.1f}%",
            "Flag": "YES (1)" if int(r.get("is_fraud", 0)) == 1 else "NO (0)",
            "Decision": str(r.get("decision", "APPROVE")),
        }
        for r in results
    ]

    return pd.DataFrame(rows).to_string(index=False)


if __name__ == "__main__":
    sample_normal = {
        "customer_id": "C1036",
        "amount": 110000,
        "transaction_time": "14:30",
        "device": "known",
        "location": "Bengaluru",
    }
    sample_risky = {
        "customer_id": "C1188",
        "amount": 320000,
        "transaction_time": "02:15",
        "device": "new",
        "location": "unknown",
    }

    results = [
        score_transaction(sample_normal, customer_avg_amount=118979.0),
        score_transaction(sample_risky, customer_avg_amount=76640.33),
    ]

    print("\nREAL-TIME TRANSACTION SCORING SUMMARY")
    print("=" * 78)
    print(format_predictions_table(results))
    print("=" * 78)