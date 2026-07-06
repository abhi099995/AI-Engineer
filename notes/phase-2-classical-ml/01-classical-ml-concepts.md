# Classical ML Concepts

## The ML Workflow

```
Raw Data → EDA → Feature Engineering → Model Training → Evaluation → Deployment
```

## 1. Types of Learning

| Type | Definition | Example |
|------|-----------|---------|
| Supervised | Learn from labeled examples | Credit risk scoring |
| Unsupervised | Find structure without labels | Customer segmentation |
| Semi-supervised | Small labeled + large unlabeled | Document classification |
| Reinforcement | Learn from reward signals | Game AI, robotics |

---

## 2. Regression

### Linear Regression
$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \ldots + w_n x_n$$

Objective: minimize MSE:
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### Key variants
- Ridge (L2 regularization): penalizes large weights, handles multicollinearity
- Lasso (L1 regularization): drives some weights to zero → feature selection
- ElasticNet: combination of both

---

## 3. Classification

### Logistic Regression
$$P(y=1 \mid x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-z}}$$

- Binary classification despite the name
- Output is a probability, not a class — threshold at 0.5 by default

### Decision Trees
- Split data at each node to maximize information gain
- Prone to overfitting; fix with max_depth or min_samples_leaf
- Very interpretable

### Random Forests
- Ensemble of decision trees with bootstrap sampling
- Reduces variance through averaging predictions
- Feature importance built-in

### Gradient Boosting (XGBoost, LightGBM)
- Builds trees sequentially; each tree corrects previous errors
- Often best out-of-the-box on tabular data
- XGBoost is a go-to for Kaggle-style problems

### Support Vector Machines (SVM)
- Finds the maximum-margin hyperplane
- Effective in high-dimensional spaces
- Slower on large datasets

---

## 4. Key Evaluation Metrics

### Classification Metrics
| Metric | Formula | Use when |
|--------|---------|----------|
| Accuracy | TP+TN / total | Balanced classes |
| Precision | TP / (TP+FP) | Cost of false positive is high |
| Recall | TP / (TP+FN) | Cost of false negative is high (fraud, medical) |
| F1 Score | 2 * P*R / (P+R) | Imbalanced classes |
| ROC-AUC | Area under ROC curve | General discriminative performance |

### Regression Metrics
| Metric | Use when |
|--------|----------|
| MAE | Interpretable, robust to outliers |
| RMSE | Penalizes large errors more |
| R² | Overall fit quality (0 to 1) |

---

## 5. Bias-Variance Tradeoff

$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

| Problem | Cause | Fix |
|---------|-------|-----|
| High bias (underfitting) | Model too simple | More features, complex model |
| High variance (overfitting) | Model too complex | Regularization, more data, simpler model |

---

## 6. Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
print(scores.mean(), scores.std())
```

- k-fold: split data into k parts; train on k-1, test on 1; rotate k times
- Stratified k-fold: preserves class ratio in each fold (use for imbalanced)

---

## Personal Notes
- Always baseline with a simple model (logistic regression) before trying complex ones
- Feature engineering beats algorithm selection most of the time on tabular data
- TODO: Build a comparison notebook: LogReg vs RF vs XGBoost on same dataset
