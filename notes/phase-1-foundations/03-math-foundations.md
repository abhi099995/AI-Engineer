# Math Foundations for ML

## 1. Linear Algebra

### Vectors
- A vector is a list of numbers representing a point or direction in space
- In ML: a single data sample is often a vector of features

```
x = [age=25, salary=70000, tenure=3]  →  feature vector in R³
```

### Matrices
- A matrix is a 2D array; a dataset is a matrix where rows = samples, columns = features
- Weight matrix in neural nets: W ∈ R^(m×n)

### Dot Product
$$a \cdot b = \sum_{i} a_i b_i$$

- Measures similarity/alignment between vectors
- Core of every linear layer in neural networks: `output = W · x + b`

### Matrix Multiplication
$$C = AB \quad \text{where} \quad C_{ij} = \sum_k A_{ik} B_{kj}$$

### Key operations used in ML
| Operation | ML use |
|-----------|--------|
| Transpose (Aᵀ) | Shape alignment in matmul |
| Inverse (A⁻¹) | Analytical solutions like linear regression |
| Eigendecomposition | PCA, covariance analysis |
| SVD | Dimensionality reduction |
| Norm ‖x‖ | Regularization, distance metrics |

---

## 2. Probability

### Fundamentals
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$
$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

### Bayes' Theorem
$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

- Used in Naive Bayes classifiers
- Foundation of probabilistic ML

### Distributions used in ML
| Distribution | Use case |
|-------------|----------|
| Gaussian/Normal | Features, noise, VAEs |
| Bernoulli | Binary classification outputs |
| Categorical | Multi-class outputs (softmax) |
| Uniform | Initialization, random sampling |

---

## 3. Statistics

### Key Metrics
$$\mu = \frac{1}{n}\sum x_i \quad \text{(mean)}$$
$$\sigma^2 = \frac{1}{n}\sum (x_i - \mu)^2 \quad \text{(variance)}$$

### Correlation vs Causation
- Correlation: two variables move together
- Causation: one causes the other
- Feature correlation ≠ predictive power

### Hypothesis Testing
- Null hypothesis: "no effect"
- p-value < 0.05 → reject null hypothesis
- Used in A/B testing AI model variants

---

## 4. Calculus (Gradient-Focused)

### Derivative
$$\frac{d}{dx} f(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

- Tells you: how much does the output change if I change the input a tiny bit?

### Gradient (multi-variable extension)
$$\nabla_\theta L = \left[\frac{\partial L}{\partial \theta_1}, \frac{\partial L}{\partial \theta_2}, \ldots\right]$$

- Gradient descent: move parameters in the direction that reduces loss
$$\theta \leftarrow \theta - \alpha \cdot \nabla_\theta L$$

### Chain Rule (backpropagation)
$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

- Every backward pass in a neural net applies the chain rule recursively

### Logistic Regression Gradient Intuition (Practical)

- For binary classification, logistic regression predicts:
$$\hat{y} = \sigma(z), \quad z = w^T x + b, \quad \sigma(z)=\frac{1}{1+e^{-z}}$$
- With binary cross-entropy loss:
$$L = -\frac{1}{n}\sum_{i=1}^{n} \left[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$
- A useful simplification from the chain rule:
$$\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$$
- So gradients become:
$$\nabla_w L = \frac{1}{n}X^T(\hat{y}-y), \quad \frac{\partial L}{\partial b} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)$$
- Update rule:
$$w \leftarrow w - \alpha\nabla_w L, \quad b \leftarrow b - \alpha\frac{\partial L}{\partial b}$$

Interpretation:
- If prediction is too high for a negative class sample, gradient pushes weights down.
- If prediction is too low for a positive class sample, gradient pushes weights up.
- Over many samples, this nudges the decision boundary toward lower classification error.

---

---

## 5. Classification Thresholds & Precision-Recall Intuition

- A classifier outputs a **probability score** (0–1); a **threshold** converts it to a binary label (default/no-default)
- **Default threshold is 0.5**, but this is almost never optimal on imbalanced data — you must tune it
- Lowering the threshold → more positives predicted → **recall ↑, precision ↓** (catch more true defaults, but also flag more good borrowers)
- Raising the threshold → fewer positives predicted → **precision ↑, recall ↓** (very confident before flagging, but miss more real defaults)
- The **Precision-Recall curve** shows this tradeoff across all thresholds; the **area under it (PR-AUC)** is a better metric than ROC-AUC when positives are rare
- **Optimal threshold** depends on the business cost: in credit risk, a false negative (missed default) is far more expensive than a false positive — so you deliberately lower the threshold to recover recall
- Use `sklearn.metrics.precision_recall_curve` to sweep thresholds, then pick the one that maximises the metric that matters (e.g. F1, or a custom cost function)
- Rule of thumb: **check the PR curve first, tune threshold second, report both precision and recall** — never report accuracy alone on imbalanced data

---

## Personal Notes
- The math is not required to be mastered before building — learn it alongside building
- Priority order for ML: linear algebra → calculus basics → probability → statistics

## See Also
- [NumPy Basics](01-numpy-basics.md)
- [Pandas for ML](02-pandas-for-ml.md)
- [P1 Credit Risk Scorer README](../../projects/p1-credit-risk-scorer/README.md)
