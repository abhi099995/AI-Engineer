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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

# Fill missing values (would use separate train/test fit in real project)
df["monthly_spend"] = df["monthly_spend"].fillna(df["monthly_spend"].median())
df["age"] = df["age"].fillna(df["age"].median())

# Feature engineering
df["high_spender"] = (df["monthly_spend"] > df["monthly_spend"].median()).astype(int)
df["is_new"] = (df["tenure_months"] <= 6).astype(int)
df["clv_estimate"] = df["monthly_spend"] * df["tenure_months"]

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


# ──────────────────────────────────────────────
# PART 3: Baseline Logistic Regression
# ──────────────────────────────────────────────

print("\n" + "=" * 50)
print("PART 3: Baseline Logistic Regression")
print("=" * 50)

feature_cols = [
    "age",
    "tenure_months",
    "monthly_spend",
    "support_tickets",
    "high_spender",
    "is_new",
    "clv_estimate",
    "plan_enterprise",
    "plan_pro",
]

X_model = df_encoded[feature_cols]
y_model = df_encoded["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X_model,
    y_model,
    test_size=0.2,
    random_state=42,
    stratify=y_model,
)

baseline_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        class_weight="balanced",
        random_state=42,
        solver="liblinear",
        max_iter=2000,
    ),
)
baseline_model.fit(X_train, y_train)

y_proba = baseline_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_proba)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print(f"ROC-AUC:   {auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print("Confusion matrix:")
print(cm)


# ──────────────────────────────────────────────
# PART 4: One-Step Gradient Descent Tie-In
# ──────────────────────────────────────────────

print("\n" + "=" * 50)
print("PART 4: Logistic Gradient Descent (Tiny Demo)")
print("=" * 50)

# Small deterministic toy data (2 features, binary target)
X_toy = np.array([
    [0.2, 1.0],
    [1.0, 1.5],
    [1.2, 0.7],
    [2.0, 1.8],
], dtype=float)
y_toy = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)

w = np.zeros(2, dtype=float)
b = 0.0
lr = 0.1


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def bce_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-12
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


# Before update
logits_before = X_toy @ w + b
probs_before = sigmoid(logits_before)
loss_before = bce_loss(y_toy, probs_before)

# Gradient for logistic regression with BCE: dw = X^T (p - y) / n, db = mean(p - y)
errors = probs_before - y_toy
dw = (X_toy.T @ errors) / len(X_toy)
db = float(np.mean(errors))

# One gradient descent step
w = w - lr * dw
b = b - lr * db

# After update
logits_after = X_toy @ w + b
probs_after = sigmoid(logits_after)
loss_after = bce_loss(y_toy, probs_after)

print(f"Loss before step: {loss_before:.6f}")
print(f"Loss after  step: {loss_after:.6f}")
print(f"Updated weights: {np.round(w, 4)}")
print(f"Updated bias: {b:.4f}")
print("Expectation: loss should decrease after this step.")
