"""FastAPI scoring service.   Run from the repo root:  uvicorn api.app:app --reload"""
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src import config as C
from src.scoring import FraudScorer

state = {}
audit = logging.getLogger("fraud_audit")


def setup_audit_logger():
    C.LOG_DIR.mkdir(exist_ok=True)
    if not audit.handlers:
        h = RotatingFileHandler(C.AUDIT_LOG, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        h.setFormatter(logging.Formatter("%(message)s"))
        audit.addHandler(h); audit.setLevel(logging.INFO); audit.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["scorer"] = FraudScorer()   # heavy models loaded ONCE at startup, not per request
    setup_audit_logger()
    yield
    state.clear()


app = FastAPI(title="Hybrid Credit Card Fraud Detection API", version="1.0.0", lifespan=lifespan)


class Transaction(BaseModel):
    user_id: int = Field(..., ge=1, examples=[42])
    amount: float = Field(..., gt=0, le=10_000_000, examples=[1500])
    card_type: Literal["VISA", "MASTER", "RUPAY"] = "VISA"
    merchant_category: str = Field("grocery", examples=["grocery"])
    location: str = Field(..., min_length=2, examples=["Mumbai"])
    device_type: Literal["pos_terminal", "web", "mobile_app", "atm"] = "mobile_app"
    timestamp: Optional[datetime] = Field(None, description="Defaults to server time")


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": "scorer" in state}


# NOTE: plain `def` (not `async def`): model inference is CPU-bound & blocking, so FastAPI
# runs it in a worker thread pool instead of freezing the event loop.
@app.post("/score_transaction")
def score_transaction(tx: Transaction):
    scorer: FraudScorer = state["scorer"]
    if tx.merchant_category not in scorer.meta["merchant_categories"]:
        raise HTTPException(422, f"Unknown merchant_category. Allowed: {scorer.meta['merchant_categories']}")
    t0 = time.perf_counter()
    payload = tx.model_dump()
    result = scorer.score_one(payload)
    result.update(request_id=str(uuid.uuid4()), latency_ms=round((time.perf_counter() - t0) * 1000, 1))
    audit.info(json.dumps({"logged_at": datetime.now().isoformat(timespec="seconds"),
                           "request_id": result["request_id"],
                           "input": {**payload, "timestamp": str(payload["timestamp"])},
                           "risk_score": result["risk_score"], "decision": result["decision"],
                           "reasons": [r["text"] for r in result["reasons"]]}))
    return result


@app.get("/audit_log")
def audit_log(limit: int = Query(50, ge=1, le=1000)):
    if not C.AUDIT_LOG.exists():
        return []
    lines = C.AUDIT_LOG.read_text(encoding="utf-8").strip().splitlines()[-limit:]
    return [json.loads(x) for x in reversed(lines)]


@app.get("/meta")
def meta():
    return state["scorer"].meta
