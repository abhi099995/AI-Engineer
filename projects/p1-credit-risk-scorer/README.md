# P1: Credit Risk Scorer

> **Phase:** Classical ML | **Status:** 🔄 In Progress

## Problem Statement

Banks and fintechs need to predict whether a loan applicant will default.
This project builds a production-style credit risk scoring API backed by a trained classification model.

## Business Context

- A false negative (predicting "good" for a bad borrower) costs the bank the loan amount
- A false positive (predicting "bad" for a good borrower) loses a customer and revenue
- Threshold tuning and business cost matrices matter here — not just accuracy

## Learning Goals

- [ ] End-to-end ML pipeline: data → features → train → evaluate → serve
- [ ] Handle class imbalance (defaults are rare)
- [ ] Compare Logistic Regression, Random Forest, XGBoost
- [ ] Tune decision threshold based on business cost
- [ ] Serve predictions via FastAPI
- [ ] Write model card

## Dataset

Use: [UCI Credit Card Default Dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)  
Or: Kaggle's "Give Me Some Credit" dataset

## Architecture

```
CSV data
    ↓
EDA notebook
    ↓
Feature engineering pipeline (sklearn Pipeline)
    ↓
Model training + cross-validation
    ↓
Threshold optimization
    ↓
FastAPI endpoint: POST /predict
    ↓
Docker container
```

## Tech Stack

- `pandas`, `numpy` — data manipulation
- `scikit-learn` — pipeline, models, evaluation
- `xgboost` — primary model
- `matplotlib`, `seaborn` — visualization
- `FastAPI` — serving
- `joblib` — model serialization
- `Docker` — containerization

## Project Structure

```
p1-credit-risk-scorer/
├── data/
│   └── .gitkeep           # data not committed, instructions in README
├── notebooks/
│   ├── 01-eda.ipynb
│   ├── 02-feature-engineering.ipynb
│   └── 03-model-comparison.ipynb
├── src/
│   ├── features.py         # feature engineering functions
│   ├── train.py            # training script
│   ├── evaluate.py         # evaluation and threshold tuning
│   └── predict.py          # inference logic
├── api/
│   └── main.py             # FastAPI app
├── tests/
│   └── test_predict.py
├── Dockerfile
├── requirements.txt
├── model_card.md
└── README.md
```

## Key Metrics to Hit

| Metric | Target |
|--------|--------|
| ROC-AUC | > 0.75 |
| Recall (default class) | > 0.65 |
| Precision (default class) | > 0.50 |
| API p99 latency | < 100ms |

## Milestones

- [ ] Week 1: EDA, clean data, baseline logistic regression
- [ ] Week 2: Feature engineering, model comparison
- [ ] Week 3: Threshold tuning, FastAPI, Dockerize
- [ ] Week 4: Tests, model card, final README

## Tradeoffs and Decisions

| Decision | Option A | Option B | Chosen | Reason |
|----------|----------|----------|--------|--------|
| Model | Random Forest | XGBoost | XGBoost | Better performance on tabular + handles missing values |
| Imbalance | SMOTE | Class weights | Class weights | Less risk of overfitting synthetic samples |
| Threshold | Default 0.5 | Cost-optimized | Cost-optimized | Business cost matters more than F1 |
