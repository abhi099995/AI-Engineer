"""
P1: Credit Risk Scorer — Training Script
Trains a Logistic Regression baseline and XGBoost model with MLflow tracking.

Usage:
    python3 src/train.py

Outputs:
    - Trained model logged to MLflow
    - Model saved to models/credit_risk_model.pkl
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
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


def _compute_scale_pos_weight(y: pd.Series) -> float:
    """Compute a stable scale_pos_weight for XGBoost.

    Falls back to 1.0 when a class is missing to avoid divide-by-zero.
    """
    negative_count = int((y == 0).sum())
    positive_count = int((y == 1).sum())

    if positive_count == 0 or negative_count == 0:
        log.warning(
            "Detected single-class training labels (neg=%d, pos=%d); using scale_pos_weight=1.0",
            negative_count,
            positive_count,
        )
        return 1.0

    return float(negative_count) / float(positive_count)


def _classification_metrics(y_true: pd.Series, y_proba: np.ndarray, threshold: float = 0.4) -> dict:
    """Compute core binary classification metrics at a fixed threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


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

    lr_model = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
    )
    lr_cv_scores = cross_val_score(lr_model, X_train_t, y_train, cv=5, scoring="roc_auc")
    lr_model.fit(X_train_t, y_train)
    lr_test_proba = lr_model.predict_proba(X_test_t)[:, 1]
    lr_metrics = _classification_metrics(y_test, lr_test_proba)

    scale_pos_weight = _compute_scale_pos_weight(y_train)

    xgb_model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="auc",
        verbosity=0,
    )

    xgb_cv_scores = cross_val_score(xgb_model, X_train_t, y_train, cv=5, scoring="roc_auc")
    xgb_model.fit(X_train_t, y_train)

    xgb_test_proba = xgb_model.predict_proba(X_test_t)[:, 1]
    xgb_test_pred = (xgb_test_proba >= 0.4).astype(int)
    xgb_metrics = _classification_metrics(y_test, xgb_test_proba)

    log.info("Model comparison on test split (threshold=0.40):")
    log.info(
        "LogReg  | AUC=%.4f Precision=%.4f Recall=%.4f F1=%.4f",
        lr_metrics["auc"],
        lr_metrics["precision"],
        lr_metrics["recall"],
        lr_metrics["f1"],
    )
    log.info(
        "XGBoost | AUC=%.4f Precision=%.4f Recall=%.4f F1=%.4f",
        xgb_metrics["auc"],
        xgb_metrics["precision"],
        xgb_metrics["recall"],
        xgb_metrics["f1"],
    )
    log.info(
        "CV ROC-AUC | LogReg: %.4f ± %.4f | XGBoost: %.4f ± %.4f",
        lr_cv_scores.mean(),
        lr_cv_scores.std(),
        xgb_cv_scores.mean(),
        xgb_cv_scores.std(),
    )

    log.info(
        "XGBoost classification report (threshold=0.40):\n%s",
        classification_report(y_test, xgb_test_pred, digits=4),
    )

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "baseline_model": "logistic_regression",
        "candidate_model": "xgboost",
        "git_sha": os.environ.get("GITHUB_SHA", "local"),
    }
    metrics = {
        "logreg_cv_auc_mean": float(lr_cv_scores.mean()),
        "logreg_cv_auc_std": float(lr_cv_scores.std()),
        "logreg_test_auc": float(lr_metrics["auc"]),
        "logreg_test_precision": float(lr_metrics["precision"]),
        "logreg_test_recall": float(lr_metrics["recall"]),
        "logreg_test_f1": float(lr_metrics["f1"]),
        "xgb_cv_auc_mean": float(xgb_cv_scores.mean()),
        "xgb_cv_auc_std": float(xgb_cv_scores.std()),
        "xgb_test_auc": float(xgb_metrics["auc"]),
        "xgb_test_precision": float(xgb_metrics["precision"]),
        "xgb_test_recall": float(xgb_metrics["recall"]),
        "xgb_test_f1": float(xgb_metrics["f1"]),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    if MLFLOW_AVAILABLE:
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(xgb_model, "model")
            log.info("Experiment logged to MLflow")
    else:
        log.info("MLflow not available — skipping experiment tracking")
        log.info("Params: %s", params)
        log.info("Metrics: %s", metrics)

    # Persist XGBoost artifact for serving while logging baseline comparison metrics.
    model_path = MODELS_DIR / "credit_risk_model.pkl"
    joblib.dump({"model": xgb_model, "pipeline": pipeline}, model_path)
    log.info("Model saved to %s", model_path)


if __name__ == "__main__":
    train()
