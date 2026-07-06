# MLOps Concepts

## What is MLOps?

MLOps = DevOps principles applied to machine learning systems.

The goal: make ML systems reliable, reproducible, and continuously improving.

```
Data → Training → Evaluation → Deployment → Monitoring → Retraining (loop)
```

---

## 1. Experiment Tracking

Track every training run so you can reproduce and compare.

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", 0.92)
    mlflow.log_metric("f1_score", 0.87)
    mlflow.sklearn.log_model(model, "model")
```

### What to always log
- Hyperparameters
- Training data version/hash
- Evaluation metrics (train + val + test)
- Model artifact
- Git commit SHA

---

## 2. Data Versioning

A model is only reproducible if you can reproduce the exact training data.

Options:
- `DVC` (Data Version Control) — tracks datasets in git-like fashion
- Store data in S3/GCS with versioned paths: `s3://bucket/data/v1.2/train.csv`
- Record row counts, feature distributions, and hash per dataset version

---

## 3. Model Serving Patterns

### Batch Inference
- Run predictions on a large dataset on a schedule
- Use case: overnight churn scoring, daily recommendations
- Stack: cron/Airflow → Python/Spark → write predictions to DB

### Online Inference
- REST API serving real-time predictions
- Use case: credit scoring at checkout, recommendation on page load
- Stack: FastAPI + Docker + Kubernetes or managed inference endpoints

```python
# FastAPI serving example
from fastapi import FastAPI
import joblib, numpy as np

app = FastAPI()
model = joblib.load("model.pkl")

@app.post("/predict")
def predict(features: dict):
    X = np.array([list(features.values())])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0].max()
    return {"prediction": int(prediction), "confidence": float(probability)}
```

### Async/Queue-Based Inference
- Decouple request from response for high-latency models
- Stack: API → message queue (Kafka/SQS) → worker → callback

---

## 4. Model Monitoring

Models degrade silently. You need to detect it.

### Data Drift
Input feature distribution changes from what the model was trained on.
```python
# Using Evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_df, current_data=production_df)
report.save_html("drift_report.html")
```

### Concept Drift
The relationship between inputs and output changes (e.g., customer behavior shifts).

### What to monitor
| Signal | Description |
|--------|-------------|
| Input feature distributions | Are features within expected ranges? |
| Prediction distribution | Is the model predicting mostly one class? |
| Confidence scores | Dropping confidence = model uncertainty |
| Business metrics | Revenue, conversion — ultimately what matters |
| Latency and error rates | Operational health |

---

## 5. CI/CD for ML

```yaml
# GitHub Actions example
name: ML Pipeline

on:
  push:
    paths:
      - 'src/**'
      - 'data/**'

jobs:
  train-and-evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run training
        run: python train.py
      - name: Run evaluation
        run: python evaluate.py --threshold 0.85
      - name: Build and push Docker image
        if: success()
        run: docker build -t my-model:$GITHUB_SHA .
```

---

## 6. ML System Design Checklist

Before deploying any ML model to production:

- [ ] Training pipeline is reproducible with pinned data version
- [ ] Evaluation metrics meet defined thresholds
- [ ] Model card written (what it does, limitations, fairness notes)
- [ ] API latency tested under load
- [ ] Input validation and schema enforcement
- [ ] Monitoring alerts configured for drift and latency
- [ ] Rollback plan documented
- [ ] A/B test or shadow deployment strategy planned

---

## Personal Notes
- MLOps complexity should match team size — start with MLflow + FastAPI + Docker, not Kubernetes on day one
- Shadow deployment is the safest way to test a new model in production
- TODO: Add MLflow tracking to P3 (Churn Prediction Engine)
