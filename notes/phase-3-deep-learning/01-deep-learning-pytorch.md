# Deep Learning + PyTorch Concepts

## Why Deep Learning?

Classical ML (sklearn, XGBoost) works well on tabular data.
Deep Learning handles: text, images, audio, sequences, and high-dimensional data where features cannot be manually engineered.

---

## 1. Neural Network Fundamentals

### Anatomy of a Layer
```
Input x → Linear transformation → Activation → Output
           z = Wx + b              a = σ(z)
```

- **W** — weight matrix (learned)
- **b** — bias vector (learned)
- **σ** — activation function (non-linearity)

### Why Activations?
Without activations, stacking linear layers collapses to one linear layer — no expressive power.

| Activation | Formula | Use case |
|-----------|---------|----------|
| ReLU | max(0, x) | Hidden layers (default) |
| Sigmoid | 1/(1+e⁻ˣ) | Binary output |
| Softmax | eˣⁱ / Σeˣʲ | Multi-class output |
| GELU | x·Φ(x) | Transformers |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | RNNs, normalized range |

---

## 2. Training Loop

```python
for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:

        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        # Backward pass
        optimizer.zero_grad()   # clear previous gradients
        loss.backward()         # compute gradients (backprop)
        optimizer.step()        # update weights
```

### Loss Functions
| Task | Loss |
|------|------|
| Binary classification | `BCEWithLogitsLoss` |
| Multi-class | `CrossEntropyLoss` |
| Regression | `MSELoss`, `HuberLoss` |

### Optimizers
| Optimizer | Notes |
|-----------|-------|
| SGD | Simple, needs tuned LR |
| Adam | Default choice, adaptive LR per parameter |
| AdamW | Adam + decoupled weight decay (used in transformers) |

---

## 3. PyTorch Basics

### Tensors
```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
x = torch.zeros(3, 4)
x = torch.randn(32, 128)   # batch of 32, 128 features

# GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)

# Shapes
x.shape         # torch.Size([32, 128])
x.dtype         # torch.float32
x.device        # device(type='cpu')
```

### Defining a Model
```python
import torch.nn as nn

class CreditRiskNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)   # binary output (logit)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

model = CreditRiskNet(input_dim=15).to(device)
print(model)
```

### DataLoader Pattern
```python
from torch.utils.data import DataLoader, TensorDataset

X_tensor = torch.FloatTensor(X_train.values)
y_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

---

## 4. Transformers (Conceptual)

The architecture behind GPT, BERT, T5, LLaMA.

### Self-Attention (core mechanism)
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- **Q, K, V** — Query, Key, Value projections of the input
- Each token attends to every other token
- Captures long-range dependencies efficiently (vs RNN's sequential bottleneck)

### Transformer Block
```
Input tokens
    ↓
Token embeddings + Positional embeddings
    ↓
[Multi-head Self-Attention] → residual + LayerNorm
    ↓
[Feed-Forward Network] → residual + LayerNorm
    ↓
Repeat N times (layers)
    ↓
Output
```

### Key insight for GenAI engineering
- LLMs are transformers trained on next-token prediction at massive scale
- Fine-tuning adjusts weights; prompting changes the input context
- RAG adds retrieved knowledge to the context window at inference time

---

## 5. Regularization Techniques

| Technique | How | When |
|-----------|-----|------|
| Dropout | Randomly zero activations during training | Overfitting in large networks |
| Weight decay (L2) | Penalize large weights in loss | General regularization |
| Batch Normalization | Normalize layer inputs per batch | Deep networks, faster training |
| Early stopping | Stop when val loss stops improving | Any network |
| Data augmentation | Artificially expand training data | Images, text |

---

## Personal Notes
- PyTorch is dynamic graph (define-by-run) — easier to debug than TF1 static graphs
- `model.eval()` + `torch.no_grad()` are required during inference — forgetting causes memory leaks and wrong BatchNorm behavior
- TODO: Build a binary classification net on the churn dataset, compare with XGBoost
