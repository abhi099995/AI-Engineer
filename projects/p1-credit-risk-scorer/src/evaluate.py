"""
P1: Credit Risk Scorer — Evaluation Script
Evaluates a trained model with full metrics and threshold optimization.

Usage:
    python3 src/evaluate.py

Outputs:
    - Classification report
    - ROC-AUC score
    - Optimal threshold based on F1 score
    - Confusion matrix
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "credit_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "credit_risk_model.pkl"


def _generate_synthetic_data(n: int = 2000) -> pd.DataFrame:
    """Synthetic fallback — same seed as train.py for consistency."""
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


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Find threshold that maximizes F1 score on the positive (default) class."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx])


def evaluate(threshold: Optional[float] = None) -> dict:
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Run train.py first.")
        return {}

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    pipeline = artifact["pipeline"]

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = _generate_synthetic_data()

    X = df.drop("default", axis=1)
    y = df["default"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_test_t = pipeline.transform(X_test)
    y_proba = model.predict_proba(X_test_t)[:, 1]

    # Threshold optimization
    optimal_threshold = threshold or find_optimal_threshold(y_test.values, y_proba)
    y_pred = (y_proba >= optimal_threshold).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 55)
    print("EVALUATION REPORT — P1 Credit Risk Scorer")
    print("=" * 55)
    print(f"\nOptimal threshold: {optimal_threshold:.3f}")
    print(f"ROC-AUC:           {auc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))
    print("Confusion Matrix:")
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    return {
        "roc_auc": auc,
        "threshold": optimal_threshold,
        "f1_default": f1_score(y_test, y_pred),
        "confusion_matrix": cm.tolist(),
    }


if __name__ == "__main__":
    evaluate()
