# P4: Resume Screener AI

> **Phase:** LLM + RAG | **Status:** ⏳ Upcoming (Month 4)

## Problem Statement

Recruiters spend 6-7 seconds per resume. This project builds an AI-powered resume screener that:
1. Ingests job descriptions and candidate resumes
2. Extracts structured candidate profiles
3. Scores and ranks candidates against the job requirements
4. Generates a grounded screening summary with citations

## Business Context

- Reduces time-to-shortlist from days to minutes
- Provides consistent criteria across all applicants
- Human reviewer sees AI reasoning + source — not a black box

## Learning Goals

- [ ] Build a RAG pipeline from scratch (no LangChain abstraction)
- [ ] PDF/text ingestion and chunking strategies
- [ ] Structured output extraction with LLMs
- [ ] Embedding-based semantic matching
- [ ] Evaluation: measure match quality against human labels
- [ ] Grounded generation with citation links

## Architecture

```
Job Description (text)
    ↓
Extract requirements → structured JSON (LLM)

Candidate Resumes (PDF/text)
    ↓
Extract candidate profile → structured JSON (LLM)
    ↓
Embed candidate sections → vector store

Matching:
    Query: job requirements
    Retrieve: relevant candidate sections
    Generate: match score + reasoning + citations

Output:
    {
      "candidate": "Jane Doe",
      "score": 87,
      "match_summary": "...",
      "strengths": [...],
      "gaps": [...],
      "citations": [{"text": "...", "source": "resume_p2"}]
    }
```

## Tech Stack

- `openai` — LLM + embeddings (or open-weight alternative)
- `pypdf2` / `pdfplumber` — PDF parsing
- `chromadb` — local vector store (swap to pgvector for prod)
- `FastAPI` — API
- `pydantic` — structured output validation
- `pytest` — testing
- `ragas` or custom eval — evaluation

## Project Structure

```
p4-resume-screener/
├── data/
│   ├── sample-jd.txt
│   └── sample-resumes/
├── notebooks/
│   ├── 01-ingestion-experiments.ipynb
│   └── 02-rag-pipeline-dev.ipynb
├── src/
│   ├── ingest.py           # PDF → chunks → embeddings
│   ├── extract.py          # structured profile extraction
│   ├── match.py            # scoring + ranking
│   ├── generate.py         # grounded summary generation
│   └── evaluate.py         # eval harness
├── api/
│   └── main.py
├── eval/
│   ├── test_cases.json     # manually labeled test cases
│   └── eval_report.md
├── requirements.txt
└── README.md
```

## Evaluation Framework

Manually label 30 candidate-JD pairs with expected scores.

```python
test_case = {
    "job_description": "Senior Python engineer with FastAPI...",
    "resume": "...",
    "expected_score_range": [80, 95],
    "expected_strengths": ["FastAPI", "5+ years Python"],
    "expected_gaps": ["no Kubernetes experience"]
}
```

Metrics:
| Metric | Target |
|--------|--------|
| Score correlation with human labels | > 0.75 |
| Groundedness (LLM-as-judge) | > 4/5 |
| Hallucinated skill mentions | < 5% |
| P95 API latency | < 5s |

## Milestones

- [ ] Week 1: PDF ingestion, chunking, embedding experiments
- [ ] Week 2: JD and resume structured extraction
- [ ] Week 3: RAG matching pipeline
- [ ] Week 4: Evaluation harness, API, write-up

## Tradeoffs

| Decision | Options | Chosen | Reason |
|----------|---------|--------|--------|
| Chunking | Fixed size vs semantic | Semantic (by section) | Resume sections = natural semantic units |
| Vector store | Chroma vs pgvector | Chroma (dev) → pgvector (prod) | Start local, migrate for persistence |
| Output format | Free text vs structured JSON | Structured JSON | Easier downstream filtering and ranking |
| Model | GPT-4o vs GPT-4o-mini | GPT-4o-mini + fallback | Cost — 10x cheaper, good enough for extraction |

## Extension Ideas

- Bias audit: test for demographic signals leaking into scores
- Multi-language resume support
- Batch processing pipeline for 1000+ resumes
- Integrate with ATS (Greenhouse, Lever) via webhooks
