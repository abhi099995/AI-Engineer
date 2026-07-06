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
| ROC-AUC | 0.82–0.87 (synthetic) | Baseline; improves with real data |
| Recall (default class) | ~0.70 | At threshold 0.4 |
| Precision (default class) | ~0.55 | At threshold 0.4 |
| Decision threshold | 0.40 | F1-optimized; > 0.4 = rejected |

---

## Training Data

**Dataset:** Synthetic (development phase) / UCI Credit Card Default (production target)  
**Size:** 2,000 samples (dev); 30,000+ target (prod)  
**Date range:** Ongoing (dev); 2004–2005 (UCI)  
**Features used:** age, balance, credit_limit, payment_amount, pay_status + engineered (utilization_ratio, payment_behavior, log_balance)

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
