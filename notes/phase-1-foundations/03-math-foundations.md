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

---

## Personal Notes
- The math is not required to be mastered before building — learn it alongside building
- Priority order for ML: linear algebra → calculus basics → probability → statistics
