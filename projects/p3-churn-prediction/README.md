# P3: Churn Prediction Engine

> **Phase:** Classical ML + MLOps | **Status:** ⏳ Upcoming (Month 2-3)

## Problem Statement

SaaS companies lose significant revenue to customer churn. This project builds a full churn prediction system:
from data pipeline to model training, experiment tracking, serving API, and drift monitoring.

This is the MLOps-heavy project. The goal is not just a model — it is a repeatable, monitored ML system.

## Business Context

- Churn rate of 5% monthly = ~46% annual customer loss
- Predicting churn 30 days in advance gives the retention team time to act
- Not all churned customers are worth saving — combine with CLV for targeting

## Learning Goals

- [ ] Build reproducible training pipeline with versioned data
- [ ] Track experiments with MLflow
- [ ] Build model comparison report
- [ ] Deploy as versioned API with rollback support
- [ ] Add drift monitoring with Evidently
- [ ] Automate retraining trigger based on drift alerts

## Dataset

Options:
- [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [IBM Watson Churn Dataset](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)

## Architecture

```
Data pipeline (raw → cleaned → features)
    ↓
Feature store (CSV/Parquet with versioning)
    ↓
Training script → MLflow experiment tracker
    ↓
Model evaluation + threshold tuning
    ↓
Model registry (MLflow)
    ↓
FastAPI inference service (versioned)
    ↓
Evidently monitoring dashboard
    ↓
Drift alert → trigger retraining pipeline
```

## Tech Stack

- `pandas`, `scikit-learn`, `xgboost` — ML
- `MLflow` — experiment tracking + model registry
- `FastAPI` — serving
- `Docker`, `Docker Compose` — containerization
- `Evidently` — drift monitoring
- `GitHub Actions` — CI/CD for retraining
- `pytest` — testing

## Project Structure

```
p3-churn-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01-eda.ipynb
│   └── 02-feature-analysis.ipynb
├── src/
│   ├── pipeline.py          # sklearn Pipeline
│   ├── train.py             # MLflow-tracked training
│   ├── evaluate.py
│   ├── monitor.py           # Evidently drift checks
│   └── retrain_trigger.py
├── api/
│   └── main.py
├── docker-compose.yml       # MLflow + API + monitoring
├── .github/
│   └── workflows/
│       └── retrain.yml
├── requirements.txt
└── README.md
```

## MLflow Tracking Schema

Every run logs:
```python
mlflow.log_params({
    "model_type": "xgboost",
    "n_estimators": 200,
    "max_depth": 6,
    "data_version": "v1.3",
    "data_hash": "abc123",
    "git_sha": os.environ.get("GITHUB_SHA")
})
mlflow.log_metrics({
    "train_auc": 0.91,
    "val_auc": 0.87,
    "test_auc": 0.86,
    "precision_churn": 0.72,
    "recall_churn": 0.68
})
```

## Key Metrics

| Metric | Target |
|--------|--------|
| ROC-AUC | > 0.85 |
| Recall (churn class) | > 0.70 |
| Data drift alert latency | < 24h |
| Retraining pipeline time | < 20 min |

## Milestones

- [ ] Week 1: EDA, feature engineering, baseline model
- [ ] Week 2: MLflow integration, model comparison, registry
- [ ] Week 3: API with versioning, Docker Compose stack
- [ ] Week 4: Evidently monitoring, GitHub Actions retraining pipeline

## Extension Ideas

- Combine with CLV model to prioritize retention outreach
- Segment churn by cohort (new vs long-term customers)
- Integrate with CRM webhook to trigger retention campaigns automatically
