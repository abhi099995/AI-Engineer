"""
P1: Credit Risk Scorer — Feature Engineering Pipeline
Phase: 1 Classical ML → P1 Project
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for credit risk feature engineering.
    Inherits from sklearn base classes so it works inside a Pipeline.
    """

    def fit(self, X: pd.DataFrame, y=None):
        # Compute statistics on training data only (no leakage)
        self.median_payment_ = X["payment_amount"].median() if "payment_amount" in X.columns else None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Utilization ratio: how much of the credit limit is used
        if "balance" in X.columns and "credit_limit" in X.columns:
            X["utilization_ratio"] = X["balance"] / (X["credit_limit"] + 1)

        # Payment behavior flags
        if "pay_status" in X.columns:
            X["is_delinquent"] = (X["pay_status"] > 0).astype(int)
            X["max_delinquency"] = X["pay_status"].clip(0, 9)

        # Log-transform skewed financial features
        for col in ["balance", "credit_limit", "payment_amount"]:
            if col in X.columns:
                X[f"log_{col}"] = np.log1p(X[col].clip(0))

        return X


def build_credit_pipeline() -> Pipeline:
    """Build the full preprocessing + feature engineering pipeline."""
    return Pipeline([
        ("feature_engineer", CreditFeatureEngineer()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


if __name__ == "__main__":
    # Smoke test with synthetic data
    np.random.seed(42)
    n = 500

    sample_data = pd.DataFrame({
        "age": np.random.randint(21, 70, n),
        "balance": np.random.exponential(10000, n),
        "credit_limit": np.random.choice([5000, 10000, 20000, 50000], n),
        "payment_amount": np.random.exponential(500, n),
        "pay_status": np.random.choice([-1, 0, 1, 2, 3], n, p=[0.3, 0.5, 0.1, 0.05, 0.05]),
    })

    pipeline = build_credit_pipeline()
    X_transformed = pipeline.fit_transform(sample_data)

    print(f"Input shape:  {sample_data.shape}")
    print(f"Output shape: {X_transformed.shape}")
    print("✓ Feature pipeline smoke test passed")
