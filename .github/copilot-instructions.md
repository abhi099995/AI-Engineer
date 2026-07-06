# Copilot Instructions — AI-Engineer Repository

## Project Purpose

This is a 6-month hands-on learning repository documenting a journey from senior software engineer (8 years experience) to AI Engineer. Every module has notes, experiments, and production-style projects committed daily.

---

## Repository Structure

```
AI-Engineer/
├── notes/                        # Study notes per phase (Markdown)
│   ├── phase-1-foundations/      # NumPy, Pandas, Math
│   ├── phase-2-classical-ml/     # sklearn, metrics, evaluation
│   ├── phase-3-deep-learning/    # PyTorch, transformers
│   ├── phase-4-llm-genai/        # LLMs, RAG, embeddings, agents
│   └── phase-5-mlops/            # MLflow, serving, monitoring
├── experiments/                  # Quick runnable scripts and notebooks
│   ├── numpy-pandas/
│   ├── sklearn/
│   ├── pytorch/
│   └── llm-experiments/
└── projects/                     # Full production-style projects
    ├── p1-credit-risk-scorer/
    ├── p2-support-ticket-classifier/
    ├── p3-churn-prediction/
    ├── p4-resume-screener/
    ├── p5-devops-copilot/
    ├── p6-incident-rca/
    ├── p7-code-review-bot/
    └── p8-ai-observability/
```

---

## Tech Stack

| Layer | Libraries / Tools |
|-------|------------------|
| Data / ML | `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm` |
| Deep Learning | `torch`, `transformers` (HuggingFace) |
| LLM / GenAI | `openai`, `chromadb`, `pgvector`, `tiktoken` |
| Serving | `fastapi`, `uvicorn`, `pydantic` |
| MLOps | `mlflow`, `evidently`, `docker` |
| Testing | `pytest`, `pytest-asyncio`, `httpx` |
| Visualization | `matplotlib`, `seaborn` |

**Python version:** 3.9+  
**Virtual environment:** `.venv/` (not committed)

---

## Coding Conventions

### General
- All source code is Python 3.9+
- Use type hints on all function signatures
- Use `pydantic` models for API request/response schemas
- Prefer `sklearn` Pipeline for all ML preprocessing — never transform data outside a pipeline to avoid data leakage
- Use `structlog` for structured logging in production code, not `print()`

### ML / Data Science
- Always split train/test before any feature computation — fit transformers only on training data
- Use `random_state=42` for all stochastic operations (reproducibility)
- Log all experiments with MLflow: params, metrics, data version, git SHA
- Evaluate classification models with ROC-AUC + precision/recall — never accuracy alone on imbalanced data
- Use `class_weight="balanced"` for imbalanced classification unless explicitly overridden

### LLM / GenAI
- Never hardcode API keys — use `.env` with `python-dotenv`
- Always validate structured LLM outputs with `pydantic`
- Include a confidence or groundedness score in every LLM response schema
- Sanitize tool outputs before injecting into prompts (prompt injection prevention)
- Cap tool response length before including in context

### API (FastAPI)
- All endpoints have request and response `pydantic` models
- Use `async def` for all route handlers
- Return HTTP 422 for validation errors (pydantic handles this automatically)
- Health check endpoint at `GET /health`

### Testing
- Unit tests in `tests/` per project
- Experiment scripts are standalone — runnable with `python3 <file>.py` from repo root
- No external data dependencies in tests — use `numpy.random.seed(42)` to generate synthetic data

---

## Project Conventions

Each project under `projects/` follows this structure:

```
projects/pN-project-name/
├── README.md          # problem, arch, stack, metrics, milestones, tradeoffs
├── model_card.md      # model purpose, limitations, fairness notes (ML projects)
├── data/              # gitignored — instructions in README
├── notebooks/         # exploratory notebooks
├── src/               # production source code
├── api/               # FastAPI app
├── tests/
├── Dockerfile
└── requirements.txt
```

---

## Current Phase

**Active:** Phase 1 — Foundations (NumPy, Pandas, Math)  
**Active project:** P1 — Credit Risk Scorer  
**Next milestone:** EDA on UCI Credit Card Default dataset, baseline logistic regression

---

## Commit Message Format

```
<prefix>: <short description>

Examples:
learn: add numpy broadcasting notes
exp: compare lr vs rf on credit data
feat: add feature engineering pipeline p1
fix: handle missing values in age column
eval: xgboost roc-auc 0.87 on test set
docs: add tradeoff table to p1 readme
```

---

## Do Not

- Commit `.env`, `data/raw/*.csv`, `*.pkl`, `*.joblib`, `mlruns/`, `.venv/`
- Use `print()` for logging in `src/` or `api/` code (use `structlog`)
- Fit transformers on full dataset before train/test split
- Use bare `except:` — always catch specific exceptions
- Hardcode file paths — use `pathlib.Path` relative to project root
