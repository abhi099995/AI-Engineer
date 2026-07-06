"""
P1: Credit Risk Scorer — FastAPI Inference Service

Endpoints:
    GET  /health         — health check
    POST /predict        — single prediction
    POST /predict/batch  — batch predictions

Usage:
    uvicorn api.main:app --reload --port 8000
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "credit_risk_model.pkl"

app = FastAPI(
    title="Credit Risk Scorer",
    description="Predicts probability of loan default for a given applicant.",
    version="0.1.0",
)

# Load model at startup — fail fast if missing
_artifact: Optional[dict] = None

@app.on_event("startup")
async def load_model() -> None:
    global _artifact
    if MODEL_PATH.exists():
        _artifact = joblib.load(MODEL_PATH)
    # If not found, /predict will return 503


# ──────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────

class ApplicantFeatures(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    balance: float = Field(..., ge=0, description="Current account balance")
    credit_limit: float = Field(..., ge=0, description="Total credit limit")
    payment_amount: float = Field(..., ge=0, description="Last payment amount")
    pay_status: int = Field(..., ge=-2, le=9, description="Payment status (-1=paid duly, 0=no consumption, 1-9=months delayed)")

    @field_validator("credit_limit")
    @classmethod
    def credit_limit_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("credit_limit must be positive")
        return v


class PredictionResponse(BaseModel):
    default_probability: float = Field(..., description="Probability of default (0-1)")
    risk_tier: str = Field(..., description="LOW / MEDIUM / HIGH risk tier")
    approved: bool = Field(..., description="Approval recommendation at default threshold 0.4")


class BatchPredictionRequest(BaseModel):
    applicants: list[ApplicantFeatures] = Field(..., min_items=1, max_items=100)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

DEFAULT_THRESHOLD = 0.4

def _to_dataframe(features: ApplicantFeatures) -> pd.DataFrame:
    return pd.DataFrame([features.model_dump()])


def _risk_tier(prob: float) -> str:
    if prob < 0.2:
        return "LOW"
    if prob < DEFAULT_THRESHOLD:
        return "MEDIUM"
    return "HIGH"


def _predict_single(features: ApplicantFeatures) -> PredictionResponse:
    if _artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    model = _artifact["model"]
    pipeline = _artifact["pipeline"]

    df = _to_dataframe(features)
    X = pipeline.transform(df)
    prob = float(model.predict_proba(X)[0, 1])

    return PredictionResponse(
        default_probability=round(prob, 4),
        risk_tier=_risk_tier(prob),
        approved=prob < DEFAULT_THRESHOLD,
    )


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _artifact is not None,
        "version": "0.1.0",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: ApplicantFeatures) -> PredictionResponse:
    return _predict_single(features)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    predictions = [_predict_single(applicant) for applicant in request.applicants]
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))
