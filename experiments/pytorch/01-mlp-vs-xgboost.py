"""
Experiment: PyTorch Neural Network vs XGBoost on synthetic churn data
Phase: 3 — Deep Learning
Date: 2026-07-06

Goals:
- Build a simple MLP in PyTorch for binary classification
- Compare against XGBoost baseline
- Practice the full PyTorch training loop
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


# ──────────────────────────────────────────────
# Generate synthetic churn data
# ──────────────────────────────────────────────

np.random.seed(42)
n = 2000

df = pd.DataFrame({
    "age": np.random.randint(22, 65, n),
    "tenure_months": np.random.randint(1, 120, n),
    "monthly_spend": np.round(np.random.exponential(scale=200, size=n), 2),
    "support_tickets": np.random.poisson(lam=1.5, size=n),
    "plan": np.random.choice(["basic", "pro", "enterprise"], n, p=[0.5, 0.35, 0.15]),
})

churn_prob = (
    0.3
    - 0.002 * df["tenure_months"]
    + 0.08 * df["support_tickets"]
    + (df["plan"] == "basic").astype(float) * 0.1
    - (df["plan"] == "enterprise").astype(float) * 0.15
).clip(0.05, 0.90)
df["churned"] = np.random.binomial(1, churn_prob)

df["clv_estimate"] = df["monthly_spend"] * df["tenure_months"]
df["is_new"] = (df["tenure_months"] <= 6).astype(int)
df["high_ticket_rate"] = (df["support_tickets"] >= 3).astype(int)
df = pd.get_dummies(df, columns=["plan"], drop_first=True)

X = df.drop("churned", axis=1).values.astype(np.float32)
y = df["churned"].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocess
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
X_train = scaler.fit_transform(imputer.fit_transform(X_train))
X_test = scaler.transform(imputer.transform(X_test))


# ──────────────────────────────────────────────
# PyTorch MLP
# ──────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"

class ChurnMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Compute class weight for imbalance
pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)

model = ChurnMLP(input_dim=X_train.shape[1]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# DataLoader
train_dataset = TensorDataset(
    torch.FloatTensor(X_train).to(device),
    torch.FloatTensor(y_train).unsqueeze(1).to(device)
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training loop
print("Training PyTorch MLP...")
for epoch in range(30):
    model.train()
    total_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            logits = model(X_test_t).squeeze().cpu().numpy()
            proba = 1 / (1 + np.exp(-logits))
            val_auc = roc_auc_score(y_test, proba)
        print(f"  Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")

# Final MLP AUC
model.eval()
with torch.no_grad():
    X_test_t = torch.FloatTensor(X_test).to(device)
    logits = model(X_test_t).squeeze().cpu().numpy()
    mlp_proba = 1 / (1 + np.exp(-logits))
mlp_auc = roc_auc_score(y_test, mlp_proba)


# ──────────────────────────────────────────────
# XGBoost baseline
# ──────────────────────────────────────────────

print("\nTraining XGBoost...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="auc",
    verbosity=0
)
xgb.fit(X_train, y_train)
xgb_proba = xgb.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_proba)


# ──────────────────────────────────────────────
# Comparison
# ──────────────────────────────────────────────

print("\n" + "=" * 45)
print("COMPARISON SUMMARY")
print("=" * 45)
print(f"{'Model':<20} {'ROC-AUC':>10}")
print(f"{'PyTorch MLP':<20} {mlp_auc:>10.4f}")
print(f"{'XGBoost':<20} {xgb_auc:>10.4f}")

print("\n✓ Key observations:")
print("  - XGBoost typically wins on tabular data with small datasets")
print("  - MLP needs more data and tuning to compete")
print("  - PyTorch overhead: preprocessing must be done manually outside the model")
print("  - Next: try on image/text data where MLP/transformers shine")
