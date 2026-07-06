"""
P1: Credit Risk Scorer — Training Script
Trains XGBoost model with MLflow experiment tracking.

Usage:
    python3 src/train.py

Outputs:
    - Trained model logged to MLflow
    - Model saved to models/credit_risk_model.pkl
"""

import os
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

from features import build_credit_pipeline

try:
    import mlflow
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "credit_features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load processed feature data. Falls back to synthetic data if file not found."""
    if path.exists():
        df = pd.read_csv(path)
        log.info("Loaded data from %s — %d rows", path, len(df))
    else:
        log.warning("Data file not found at %s — generating synthetic data", path)
        df = _generate_synthetic_data()

    X = df.drop("default", axis=1)
    y = df["default"]
    return X, y


def _generate_synthetic_data(n: int = 2000) -> pd.DataFrame:
    """Generate synthetic credit risk data for development/testing."""
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(21, 70, n),
        "balance": np.random.exponential(10000, n),
        "credit_limit": np.random.choice([5000, 10000, 20000, 50000], n),
        "payment_amount": np.random.exponential(500, n),
        "pay_status": np.random.choice([-1, 0, 1, 2, 3], n, p=[0.3, 0.5, 0.1, 0.05, 0.05]),
    })
    default_prob = (
        0.15
        + 0.06 * (df["pay_status"] > 0).astype(float)
        - 0.001 * df["payment_amount"].clip(0, 2000)
    ).clip(0.05, 0.80)
    df["default"] = np.random.binomial(1, default_prob)
    return df


def train(
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.05,
) -> None:
    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_credit_pipeline()
    X_train_t = pipeline.fit_transform(X_train)
    X_test_t = pipeline.transform(X_test)

    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="auc",
        verbosity=0,
    )

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train_t, y_train, cv=5, scoring="roc_auc")
    log.info("CV ROC-AUC: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    model.fit(X_train_t, y_train)

    test_proba = model.predict_proba(X_test_t)[:, 1]
    test_auc = roc_auc_score(y_test, test_proba)
    log.info("Test ROC-AUC: %.4f", test_auc)

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "git_sha": os.environ.get("GITHUB_SHA", "local"),
    }
    metrics = {
        "cv_auc_mean": float(cv_scores.mean()),
        "cv_auc_std": float(cv_scores.std()),
        "test_auc": float(test_auc),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    if MLFLOW_AVAILABLE:
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(model, "model")
            log.info("Experiment logged to MLflow")
    else:
        log.info("MLflow not available — skipping experiment tracking")
        log.info("Params: %s", params)
        log.info("Metrics: %s", metrics)

    model_path = MODELS_DIR / "credit_risk_model.pkl"
    joblib.dump({"model": model, "pipeline": pipeline}, model_path)
    log.info("Model saved to %s", model_path)


if __name__ == "__main__":
    train()
