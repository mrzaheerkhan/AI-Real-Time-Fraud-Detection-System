import sqlite3
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_MODEL_DIR = PROJECT_ROOT / "ml-model"
DB_PATH = PROJECT_ROOT / "database" / "fraud_detection.db"

if str(ML_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(ML_MODEL_DIR))

from predict import load_fraud_model, score_transaction  # noqa: E402

MODEL: Any = None
FEATURE_COLS: list[str] = []


class TransactionInput(BaseModel):
    customer_id: str = Field(..., min_length=1, examples=["C1188"])
    amount: float = Field(..., gt=0, examples=[320000])
    transaction_time: str | None = Field(default=None, examples=["02:15"])
    device: str = Field(default="known", examples=["new"])
    location: str = Field(default="unknown", examples=["Bengaluru"])


def get_db_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                amount REAL NOT NULL,
                customer_avg_amount REAL,
                spending_deviation REAL NOT NULL,
                fraud_probability REAL NOT NULL,
                is_fraud INTEGER NOT NULL,
                decision TEXT NOT NULL,
                device TEXT,
                location TEXT,
                transaction_time TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def get_customer_avg_amount(customer_id: str) -> float | None:
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT AVG(amount) AS avg_amt FROM transactions WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()
    if row and row["avg_amt"] is not None:
        return float(row["avg_amt"])
    return None


def save_transaction_record(payload: TransactionInput, result: dict[str, Any]) -> int:
    created_at = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO transactions (
                customer_id,
                amount,
                customer_avg_amount,
                spending_deviation,
                fraud_probability,
                is_fraud,
                decision,
                device,
                location,
                transaction_time,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.customer_id,
                float(result["amount"]),
                result.get("customer_avg_amount"),
                float(result["spending_deviation"]),
                float(result["fraud_probability"]),
                int(result["is_fraud"]),
                str(result["decision"]),
                payload.device,
                payload.location,
                payload.transaction_time,
                created_at,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, FEATURE_COLS
    init_db()
    try:
        MODEL, FEATURE_COLS = load_fraud_model()
    except Exception as exc:
        raise RuntimeError(f"Failed to load fraud_model.pkl: {exc}") from exc
    yield


app = FastAPI(title="AI Real-Time Fraud Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "features": FEATURE_COLS,
    }


@app.post("/api/transactions/score")
def score_incoming_transaction(payload: TransactionInput) -> dict[str, Any]:
    try:
        avg_amount = get_customer_avg_amount(payload.customer_id)
        result = score_transaction(
            transaction=payload.model_dump(),
            customer_avg_amount=avg_amount,
            model=MODEL,
            feature_cols=FEATURE_COLS,
        )
        record_id = save_transaction_record(payload, result)
        return {
            "id": record_id,
            **result,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/transactions")
def list_transactions(limit: int = 20) -> list[dict[str, Any]]:
    safe_limit = max(1, min(limit, 100))
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, customer_id, amount, customer_avg_amount,
                   spending_deviation, fraud_probability, is_fraud,
                   decision, device, location, transaction_time, created_at
            FROM transactions
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
    return [dict(row) for row in rows]

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)