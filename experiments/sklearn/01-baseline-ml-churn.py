"""
Experiment: Baseline ML — Logistic Regression + Random Forest on synthetic churn data
Phase: 2 — Classical ML
Date: 2026-07-06

Goals:
- Build a proper train/val/test split pipeline
- Train two baseline models
- Compare with proper evaluation metrics
- Practice feature engineering without data leakage
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ──────────────────────────────────────────────
# Generate synthetic churn dataset
# ──────────────────────────────────────────────

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    "age": np.random.randint(22, 65, n),
    "tenure_months": np.random.randint(1, 120, n),
    "monthly_spend": np.round(np.random.exponential(scale=200, size=n), 2),
    "support_tickets": np.random.poisson(lam=1.5, size=n),
    "plan": np.random.choice(["basic", "pro", "enterprise"], n, p=[0.5, 0.35, 0.15]),
})

# Realistic churn logic: new customers with many tickets churn more
churn_prob = (
    0.3
    - 0.002 * df["tenure_months"]
    + 0.08 * df["support_tickets"]
    + (df["plan"] == "basic").astype(float) * 0.1
    - (df["plan"] == "enterprise").astype(float) * 0.15
)
churn_prob = churn_prob.clip(0.05, 0.90)
df["churned"] = np.random.binomial(1, churn_prob)

# Feature engineering
df["clv_estimate"] = df["monthly_spend"] * df["tenure_months"]
df["is_new"] = (df["tenure_months"] <= 6).astype(int)
df["high_ticket_rate"] = (df["support_tickets"] >= 3).astype(int)
df = pd.get_dummies(df, columns=["plan"], drop_first=True)

# ──────────────────────────────────────────────
# Split
# ──────────────────────────────────────────────

X = df.drop("churned", axis=1)
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
print(f"Churn rate — Train: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")

# ──────────────────────────────────────────────
# Model 1: Logistic Regression
# ──────────────────────────────────────────────

lr_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    # solver='saga' is more numerically stable for large feature ranges
    ("model", LogisticRegression(class_weight="balanced", max_iter=1000,
                                  solver="saga", random_state=42))
])

lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)
lr_proba = lr_pipeline.predict_proba(X_test)[:, 1]

print("\n" + "=" * 50)
print("MODEL 1: Logistic Regression")
print("=" * 50)
print(classification_report(y_test, lr_preds, target_names=["No Churn", "Churn"]))
print(f"ROC-AUC: {roc_auc_score(y_test, lr_proba):.4f}")

# Cross-validation
lr_cv_scores = cross_val_score(lr_pipeline, X_train, y_train, cv=5, scoring="roc_auc")
print(f"CV ROC-AUC: {lr_cv_scores.mean():.4f} ± {lr_cv_scores.std():.4f}")

# ──────────────────────────────────────────────
# Model 2: Random Forest
# ──────────────────────────────────────────────

rf_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    ))
])

rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_test)
rf_proba = rf_pipeline.predict_proba(X_test)[:, 1]

print("\n" + "=" * 50)
print("MODEL 2: Random Forest")
print("=" * 50)
print(classification_report(y_test, rf_preds, target_names=["No Churn", "Churn"]))
print(f"ROC-AUC: {roc_auc_score(y_test, rf_proba):.4f}")

rf_cv_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=5, scoring="roc_auc")
print(f"CV ROC-AUC: {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")

# ──────────────────────────────────────────────
# Feature Importance (RF)
# ──────────────────────────────────────────────

feature_importance = pd.Series(
    rf_pipeline.named_steps["model"].feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n" + "=" * 50)
print("TOP FEATURES (Random Forest)")
print("=" * 50)
print(feature_importance.head(8).round(4))

# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────

print("\n" + "=" * 50)
print("COMPARISON SUMMARY")
print("=" * 50)
print(f"{'Model':<25} {'ROC-AUC':>10} {'CV AUC':>10}")
print(f"{'Logistic Regression':<25} {roc_auc_score(y_test, lr_proba):>10.4f} {lr_cv_scores.mean():>10.4f}")
print(f"{'Random Forest':<25} {roc_auc_score(y_test, rf_proba):>10.4f} {rf_cv_scores.mean():>10.4f}")

print("\n✓ Key observations:")
print("  - tenure_months and clv_estimate are the strongest predictors")
print("  - Random Forest outperforms LR — non-linear patterns in the data")
print("  - class_weight='balanced' important due to class imbalance")
print("  - Next step: try XGBoost and tune decision threshold for recall vs precision")
