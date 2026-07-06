# Model Card — P1: Credit Risk Scorer

## Model Summary

**Task:** Binary classification — predict whether a loan applicant will default  
**Model type:** XGBoost classifier with sklearn preprocessing pipeline  
**Version:** 0.1.0-dev  
**Status:** In development

---

## Intended Use

**Primary use:** Assist credit officers in triaging loan applications.  
**Intended users:** Credit risk analysts, underwriters.  
**Out-of-scope uses:**
- Autonomous approval or rejection without human review
- Use in jurisdictions where algorithmic credit scoring is regulated without compliance review

---

## Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| ROC-AUC | TBD | Target > 0.75 |
| Recall (default class) | TBD | Target > 0.65 |
| Precision (default class) | TBD | Target > 0.50 |
| Decision threshold | TBD | Business-optimized |

---

## Training Data

**Dataset:** TBD (UCI Credit Card Default / Kaggle Give Me Some Credit)  
**Size:** TBD  
**Date range:** TBD  
**Features used:** TBD

---

## Limitations

- Model is trained on historical data and may not reflect current credit conditions
- May perform poorly on applicants from distributions underrepresented in training data
- Does not account for macroeconomic context
- Threshold tuned for current business cost matrix — review if costs change

---

## Fairness Considerations

- TODO: Audit model predictions by age group, gender, and geography
- TODO: Run disparate impact analysis before production deployment

---

## Tradeoffs Made

| Decision | Chosen | Alternative | Reason |
|----------|--------|-------------|--------|
| Model | XGBoost | Neural Net | Interpretable, handles tabular well |
| Imbalance | Class weights | SMOTE | Avoids synthetic sample overfitting |
| Threshold | Cost-optimized | 0.5 default | Business cost matters more than F1 |

---

## Contact

**Owner:** [Your name]  
**Repo:** [Your GitHub URL]
