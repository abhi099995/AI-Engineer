# P8: AI Observability Dashboard

> **Phase:** MLOps + Monitoring | **Status:** ⏳ Upcoming (Month 6)

## Problem Statement

AI systems fail silently: model accuracy drops, prompts drift, embeddings degrade, and costs spike.
This project builds an observability platform for AI systems that monitors:
1. LLM app performance (latency, tokens, cost, quality scores)
2. ML model performance (prediction distribution, data drift, concept drift)
3. Vector search health (retrieval quality, embedding coverage)

## Business Context

- One team's LLM costs can spike 10x overnight without visibility
- Model quality degradation is often discovered by end users, not engineers
- This project is the "Datadog for AI systems" experience

## Learning Goals

- [ ] Instrument LLM calls with OpenTelemetry-style tracing
- [ ] Build a metric collection and storage pipeline
- [ ] Create drift detection alerts
- [ ] Build a Grafana-based dashboard
- [ ] Implement cost and quality budgets with alerts

## Architecture

```
AI applications (LLM apps, ML inference services)
    ↓
Instrumentation SDK (decorator/middleware)
    Records:
    - Prompt + response (masked)
    - Token counts (input + output)
    - Latency
    - Model name + version
    - Quality scores (async evaluation)
    ↓
Metrics collector (FastAPI ingest service)
    ↓
PostgreSQL (structured metrics) + InfluxDB/Prometheus (time series)
    ↓
Evidently for drift reports
    ↓
Grafana dashboards:
    - LLM cost dashboard
    - Quality trends
    - Drift alerts
    - Retrieval health
```

## Tech Stack

- `FastAPI` — metrics ingest API
- `PostgreSQL` — structured metric storage
- `Prometheus` — time-series metrics
- `Grafana` — dashboards
- `Evidently` — drift detection
- `Docker Compose` — full stack orchestration
- `opentelemetry-sdk` — tracing instrumentation
- `pytest` — testing

## Instrumentation SDK Design

```python
from ai_observability import track_llm_call

@track_llm_call(model="gpt-4o-mini", application="resume-screener")
def generate_screening_report(prompt: str) -> str:
    response = client.chat.completions.create(...)
    return response.choices[0].message.content

# The decorator automatically records:
# - input_tokens, output_tokens, total_cost
# - latency_ms
# - timestamp, model, application
# - anonymized prompt hash (for deduplication)
```

## Dashboards to Build

### 1. LLM Cost Dashboard
- Daily/weekly cost per application
- Cost per request (P50, P95, P99)
- Token usage breakdown (input vs output)
- Cost budget utilization and alerts

### 2. Quality Trends
- Quality score trend per application (LLM-as-judge, running daily sample)
- Groundedness score over time
- Hallucination detection rate
- User satisfaction proxy (thumbs up/down if available)

### 3. Drift Dashboard
- Feature distribution shift (ML models)
- Prediction distribution over time
- Embedding cluster visualization (UMAP)
- Retrieval quality score trend

### 4. Operational Health
- Request volume and error rates
- Latency percentiles per service
- Model version distribution (what % of traffic on each version)

## Project Structure

```
p8-ai-observability/
├── sdk/
│   ├── decorators.py        # @track_llm_call, @track_ml_predict
│   └── client.py            # sends metrics to ingest API
├── ingest/
│   └── main.py              # FastAPI metrics ingest
├── storage/
│   ├── models.py            # SQLAlchemy models
│   └── migrations/
├── monitoring/
│   ├── drift_detector.py    # Evidently integration
│   └── alerts.py            # threshold-based alerts
├── dashboards/
│   └── grafana/
│       ├── cost-dashboard.json
│       ├── quality-dashboard.json
│       └── drift-dashboard.json
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Key Metrics

| Metric | Target |
|--------|--------|
| Metric collection latency | < 100ms overhead per call |
| Dashboard data freshness | < 60s |
| Drift detection latency | < 30 min |
| System availability | > 99.5% |

## Milestones

- [ ] Week 1: Instrumentation SDK, ingest API, PostgreSQL storage
- [ ] Week 2: Prometheus integration, first Grafana dashboard
- [ ] Week 3: Evidently drift detection, alerts
- [ ] Week 4: UMAP embedding visualization, full stack demo

## Extension Ideas

- Integrate with all previous projects (P3, P4, P5, P6) as real instrumentation targets
- Cost anomaly detection with ML (detect unusual spending patterns)
- Session-level quality scoring for multi-turn conversations
- Export metrics to DataDog or New Relic via standard APIs
