"""
P1: Credit Risk Scorer — Tests

Tests use synthetic data via numpy.random.seed(42) — no external data dependencies.

Usage:
    pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ──────────────────────────────────────────────
# Feature pipeline tests
# ──────────────────────────────────────────────

def _synthetic_credit_df(n: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.randint(21, 70, n),
        "balance": np.random.exponential(10000, n),
        "credit_limit": np.random.choice([5000, 10000, 20000, 50000], n),
        "payment_amount": np.random.exponential(500, n),
        "pay_status": np.random.choice([-1, 0, 1, 2, 3], n, p=[0.3, 0.5, 0.1, 0.05, 0.05]),
    })


def test_feature_pipeline_output_shape() -> None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from features import build_credit_pipeline

    df = _synthetic_credit_df(50)
    pipeline = build_credit_pipeline()
    X = pipeline.fit_transform(df)

    assert X.shape[0] == 50, "Row count must be preserved"
    assert X.shape[1] > df.shape[1], "Feature engineering should add columns"


def test_feature_pipeline_no_nulls() -> None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from features import build_credit_pipeline

    df = _synthetic_credit_df(50)
    df.loc[0, "balance"] = np.nan  # inject null
    df.loc[5, "payment_amount"] = np.nan

    pipeline = build_credit_pipeline()
    X = pipeline.fit_transform(df)

    assert not np.isnan(X).any(), "Pipeline must handle nulls — no NaN in output"


def test_feature_pipeline_reproducible() -> None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from features import build_credit_pipeline

    df = _synthetic_credit_df(50)
    pipeline = build_credit_pipeline()
    X1 = pipeline.fit_transform(df)
    X2 = pipeline.transform(df)

    np.testing.assert_array_almost_equal(X1, X2, decimal=5)


# ──────────────────────────────────────────────
# API tests (no model loaded — 503 expected)
# ──────────────────────────────────────────────

def _get_test_client() -> TestClient:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from api.main import app
    return TestClient(app)


def _force_model_unloaded() -> None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1]))
    import api.main as main

    main._artifact = None


def test_health_endpoint() -> None:
    client = _get_test_client()
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "version" in data


def test_predict_invalid_age_raises_422() -> None:
    client = _get_test_client()
    response = client.post("/predict", json={
        "age": 15,  # below minimum of 18
        "balance": 5000,
        "credit_limit": 10000,
        "payment_amount": 300,
        "pay_status": 0,
    })
    assert response.status_code == 422


def test_predict_invalid_credit_limit_raises_422() -> None:
    client = _get_test_client()
    response = client.post("/predict", json={
        "age": 30,
        "balance": 5000,
        "credit_limit": -1000,  # negative
        "payment_amount": 300,
        "pay_status": 0,
    })
    assert response.status_code == 422


def test_batch_predict_empty_list_raises_422() -> None:
    client = _get_test_client()
    response = client.post("/predict/batch", json={"applicants": []})
    assert response.status_code == 422


def test_predict_returns_503_when_model_unavailable() -> None:
    client = _get_test_client()
    _force_model_unloaded()
    response = client.post("/predict", json={
        "age": 30,
        "balance": 5000,
        "credit_limit": 10000,
        "payment_amount": 300,
        "pay_status": 0,
    })
    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]


def test_batch_predict_returns_503_when_model_unavailable() -> None:
    client = _get_test_client()
    _force_model_unloaded()
    response = client.post("/predict/batch", json={
        "applicants": [{
            "age": 30,
            "balance": 5000,
            "credit_limit": 10000,
            "payment_amount": 300,
            "pay_status": 0,
        }]
    })
    assert response.status_code == 503
