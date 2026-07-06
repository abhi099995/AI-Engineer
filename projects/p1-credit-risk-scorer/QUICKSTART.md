# P1: Credit Risk Scorer Setup

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python3 src/train.py
```

### Evaluate
```bash
python3 src/evaluate.py
```

### Serve with FastAPI
```bash
uvicorn api.main:app --reload --port 8000
```

Then visit: http://localhost:8000/docs (interactive API docs)

### Run tests
```bash
pytest tests/ -v
```

## Project structure
- `src/features.py` — feature engineering pipeline
- `src/train.py` — model training with MLflow
- `src/evaluate.py` — evaluation with threshold optimization
- `api/main.py` — FastAPI serving
- `tests/test_predict.py` — unit tests
- `requirements.txt` — project dependencies
- `model_card.md` — model documentation
