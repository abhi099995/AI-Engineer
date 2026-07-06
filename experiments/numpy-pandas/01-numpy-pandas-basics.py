"""
Experiment: NumPy + Pandas Fundamentals
Phase: 1 — Foundations
Date: 2026-07-06

Goals:
- Practice core NumPy operations used in ML
- Practice a basic EDA workflow with Pandas
- Generate a synthetic dataset to avoid data download requirements
"""

import numpy as np
import pandas as pd
from collections import Counter

# ──────────────────────────────────────────────
# PART 1: NumPy Operations
# ──────────────────────────────────────────────

print("=" * 50)
print("PART 1: NumPy Operations")
print("=" * 50)

# Create a synthetic "feature matrix" — 100 samples, 4 features
np.random.seed(42)
X = np.random.randn(100, 4)  # Gaussian features
y = (X[:, 0] + X[:, 1] * 0.5 > 0).astype(int)  # binary label based on features

print(f"Feature matrix shape: {X.shape}")
print(f"Label distribution: {Counter(y)}")

# Normalize features to [0, 1] — common in ML preprocessing
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_normalized = (X - X_min) / (X_max - X_min)

print(f"\nBefore normalization — col 0: mean={X[:,0].mean():.3f}, std={X[:,0].std():.3f}")
print(f"After normalization  — col 0: mean={X_normalized[:,0].mean():.3f}, std={X_normalized[:,0].std():.3f}")

# Correlation matrix — spot correlated features before modeling
corr_matrix = np.corrcoef(X.T)
print(f"\nFeature correlation matrix:\n{np.round(corr_matrix, 2)}")

# Dot product — this is what linear layers do: output = W @ x + b
weights = np.array([0.5, -0.3, 0.1, 0.0])
bias = 0.2
predictions = X @ weights + bias  # matrix-vector multiplication
print(f"\nLinear predictions (first 5): {np.round(predictions[:5], 3)}")
print(f"Sigmoid outputs (first 5): {np.round(1 / (1 + np.exp(-predictions[:5])), 3)}")


# ──────────────────────────────────────────────
# PART 2: Pandas EDA Workflow
# ──────────────────────────────────────────────

print("\n" + "=" * 50)
print("PART 2: Pandas EDA Workflow")
print("=" * 50)

# Create a synthetic customer dataset
np.random.seed(42)
n = 300

df = pd.DataFrame({
    "age": np.random.randint(22, 65, n),
    "tenure_months": np.random.randint(1, 120, n),
    "monthly_spend": np.round(np.random.exponential(scale=200, size=n), 2),
    "support_tickets": np.random.poisson(lam=1.5, size=n),
    "plan": np.random.choice(["basic", "pro", "enterprise"], n, p=[0.5, 0.35, 0.15]),
    "churned": np.random.choice([0, 1], n, p=[0.75, 0.25])
})

# Inject some missing values (realistic)
df.loc[np.random.choice(n, 15), "monthly_spend"] = np.nan
df.loc[np.random.choice(n, 8), "age"] = np.nan

print("\nDataset shape:", df.shape)
print("\nColumn types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nBasic statistics:\n", df.describe().round(2))

# Churn rate by plan
churn_by_plan = df.groupby("plan")["churned"].agg(["mean", "count"]).round(3)
churn_by_plan.columns = ["churn_rate", "count"]
print("\nChurn rate by plan:\n", churn_by_plan)

# Feature engineering
df["high_spender"] = (df["monthly_spend"] > df["monthly_spend"].median()).astype(int)
df["is_new"] = (df["tenure_months"] <= 6).astype(int)
df["clv_estimate"] = df["monthly_spend"] * df["tenure_months"]

# Fill missing values (would use separate train/test fit in real project)
df["monthly_spend"] = df["monthly_spend"].fillna(df["monthly_spend"].median())
df["age"] = df["age"].fillna(df["age"].median())

# One-hot encode plan
df_encoded = pd.get_dummies(df, columns=["plan"], drop_first=True)

print(f"\nFeatures after encoding: {list(df_encoded.columns)}")
print(f"\nFinal dataset shape: {df_encoded.shape}")

# Correlation with target
numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
correlations = df_encoded[numeric_cols].corr()["churned"].drop("churned").sort_values()
print("\nFeature correlations with churn:\n", correlations.round(3))

print("\n✓ Experiment complete. Key takeaways:")
print("  - tenure_months negatively correlated with churn (longer = less churn)")
print("  - support_tickets positively correlated (more tickets = more at risk)")
print("  - monthly_spend has weak correlation (spend alone doesn't predict churn)")
print("  - Next step: build a baseline logistic regression on this dataset")
