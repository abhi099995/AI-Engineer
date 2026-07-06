# P2: Support Ticket Classifier

> **Phase:** NLP + Classical ML | **Status:** ⏳ Upcoming (Month 2)

## Problem Statement

Large support teams receive thousands of tickets daily. Manually routing them is slow and inconsistent.
This project builds a multi-class text classifier that routes tickets to the correct team automatically.

## Business Context

- Faster routing → lower time-to-resolution
- Consistent classification → better SLA compliance
- Confidence thresholds → uncertain tickets go to human review

## Learning Goals

- [ ] Text feature extraction: TF-IDF, word embeddings
- [ ] Multi-class classification
- [ ] Handling imbalanced class distributions in NLP
- [ ] Compare TF-IDF + LR vs fine-tuned transformer
- [ ] Evaluate with precision/recall per class
- [ ] Build a confidence threshold for human escalation

## Dataset

Options:
- [Kaggle: Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- [Zendesk Public Ticket Dataset](https://github.com/zendesk/benchmarks)
- Synthesize labeled tickets (GPT-generated for prototyping)

## Classes (Example)

- `BILLING` — invoice issues, charge disputes
- `TECHNICAL` — product not working, login issues
- `SHIPPING` — delivery, tracking, returns
- `GENERAL` — general inquiries, feedback
- `ESCALATE` — angry customers, legal threats

## Architecture

```
Raw ticket text
    ↓
Preprocessing (lowercase, remove PII, stopwords)
    ↓
Feature extraction (TF-IDF or embedding)
    ↓
Classifier (Logistic Regression or BERT)
    ↓
Confidence check
    ├─ High confidence → auto-route
    └─ Low confidence → human review queue
    ↓
FastAPI endpoint: POST /classify
```

## Tech Stack

- `pandas`, `nltk`, `spacy` — preprocessing
- `scikit-learn` — TF-IDF, Logistic Regression, evaluation
- `transformers` (HuggingFace) — optional: BERT fine-tuning
- `FastAPI` — serving
- `pytest` — testing

## Project Structure

```
p2-support-ticket-classifier/
├── data/
│   └── .gitkeep
├── notebooks/
│   ├── 01-eda-text-analysis.ipynb
│   ├── 02-tfidf-baseline.ipynb
│   └── 03-transformer-comparison.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── classify.py
├── api/
│   └── main.py
├── requirements.txt
└── README.md
```

## Key Metrics

| Class | Target Precision | Target Recall |
|-------|-----------------|---------------|
| BILLING | > 0.85 | > 0.80 |
| TECHNICAL | > 0.85 | > 0.80 |
| ESCALATE | > 0.90 | > 0.90 |
| Macro F1 | > 0.82 | — |

## Milestones

- [ ] Week 1: EDA, preprocessing, TF-IDF + LogReg baseline
- [ ] Week 2: Improve with n-grams, test on transformer embeddings
- [ ] Week 3: Confidence threshold, API, Dockerize
- [ ] Week 4: Evaluation report, tradeoff doc

## Extension Ideas

- Active learning: route low-confidence predictions to labelers
- Multilingual support with multilingual BERT
- Streaming inference for real-time Slack/Zendesk integration
