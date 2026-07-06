# P6: Incident Root Cause Analyzer

> **Phase:** LLM + RAG + Agents | **Status:** ⏳ Upcoming (Month 5)

## Problem Statement

Post-incident reviews (PIRs) take hours. Engineers manually correlate logs, metrics, traces, and past incidents.
This project builds an AI system that automatically:
1. Ingests incident data (alerts, logs, metrics, traces, Slack threads)
2. Retrieves similar past incidents from a knowledge base
3. Generates a structured root cause analysis report
4. Suggests preventive measures based on past learnings

## Business Context

- Typical RCA takes 2-8 hours of senior engineer time
- Similar incidents repeat because learnings are buried in docs
- This system turns every incident into a learning that improves future analysis

## Learning Goals

- [ ] Multi-source data ingestion and normalization
- [ ] Hybrid retrieval: keyword + semantic
- [ ] Multi-step RAG with cross-document reasoning
- [ ] Structured report generation
- [ ] Feedback loop: human corrections improve future retrieval

## System Architecture

```
Incident triggered
    ↓
Data collector:
    - Alert metadata (PagerDuty/OpsGenie API)
    - Logs (ELK/CloudWatch, last 2h)
    - Metrics (Prometheus/Datadog, last 2h)
    - Slack thread (incident channel)
    ↓
Embeddings → vector store
    ↓
Retrieve similar past incidents (hybrid search)
    ↓
Cross-document reasoning:
    - What changed recently?
    - What services are affected?
    - What pattern matches past incidents?
    ↓
Generate structured RCA report:
    {
        "summary": "...",
        "timeline": [...],
        "root_cause": "...",
        "contributing_factors": [...],
        "similar_incidents": [...],
        "remediation_applied": "...",
        "preventive_actions": [...],
        "confidence": 0.82
    }
    ↓
Post to Slack + Confluence + incident tracker
```

## Tech Stack

- `openai` — LLM + embeddings
- `pgvector` — production vector store with metadata filtering
- `FastAPI` — API
- `httpx` — async data collection from multiple sources
- `pydantic` — structured output validation
- `slack_sdk` — Slack integration (mock in dev)
- `pytest` — testing

## Project Structure

```
p6-incident-rca/
├── data/
│   └── sample-incidents/   # anonymized incident examples
├── notebooks/
│   └── 01-retrieval-experiments.ipynb
├── src/
│   ├── collector.py         # multi-source data ingestion
│   ├── embed.py             # chunking + embedding
│   ├── retriever.py         # hybrid search (BM25 + vector)
│   ├── analyzer.py          # cross-document reasoning
│   ├── report.py            # structured report generation
│   └── feedback.py         # human correction feedback loop
├── api/
│   └── main.py
├── eval/
│   ├── test_incidents.json  # 20 labeled test cases
│   └── eval_metrics.py
├── requirements.txt
└── README.md
```

## Hybrid Retrieval Design

```python
def hybrid_search(query: str, k: int = 5) -> list[Incident]:
    # Dense retrieval (semantic)
    semantic_results = vector_store.similarity_search(query, k=k*2)
    
    # Sparse retrieval (keyword/BM25)
    keyword_results = bm25_index.search(query, k=k*2)
    
    # RRF fusion (Reciprocal Rank Fusion)
    return reciprocal_rank_fusion(semantic_results, keyword_results, k=k)
```

## Evaluation

| Test Case | Input | Expected RCA | Pass Criteria |
|-----------|-------|--------------|---------------|
| DB connection pool exhaustion | alerts + logs | Connection pool limit hit during traffic spike | Root cause mentioned in report |
| Memory leak OOMKill | metrics + traces | Memory leak in payment-service v2.4.1 | Correct service and version identified |

Metrics:
| Metric | Target |
|--------|--------|
| Root cause accuracy (human eval) | > 75% |
| Report groundedness (LLM-as-judge) | > 4/5 |
| P95 report generation time | < 60s |
| Retrieval recall@5 | > 0.80 |

## Milestones

- [ ] Week 1: Data collection, normalization, embedding pipeline
- [ ] Week 2: Hybrid retrieval, similarity search experiments
- [ ] Week 3: Multi-step reasoning, structured report generation
- [ ] Week 4: Slack integration, eval harness, write-up

## Extension Ideas

- Auto-create Jira tickets for preventive action items
- Severity predictor: estimate blast radius before human assessment
- Knowledge base health score: how many incidents lack good learnings?
