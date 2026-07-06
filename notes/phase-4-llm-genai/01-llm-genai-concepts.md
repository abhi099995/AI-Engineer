# LLM + GenAI Engineering Concepts

## The GenAI Engineer Mindset

Traditional ML: train model → evaluate → deploy  
GenAI Engineering: design system → integrate model → evaluate outputs → optimize cost/latency → add guardrails

---

## 1. How LLMs Work (Practical)

- LLMs predict the next token given all previous tokens
- They do not retrieve facts; they generate based on patterns from training
- Context window: the maximum tokens an LLM can see at once
- Temperature: higher = more random, lower = more deterministic
- Top-p (nucleus sampling): restrict to top p% probability mass

---

## 2. Prompting Patterns

### Zero-shot
```
Classify this customer message as BILLING, TECHNICAL, or GENERAL.

Message: "My invoice shows double charge."
```

### Few-shot
```
Classify: "I can't log in" → TECHNICAL
Classify: "Charge me twice" → BILLING
Classify: "What are your hours?" → GENERAL

Classify: "My invoice shows double charge."
```

### Chain of Thought
```
Think step-by-step before answering.
Question: Is this customer eligible for a refund given these policies?
```

### Structured Output
```
Reply only with a JSON object:
{
  "category": "...",
  "urgency": "low|medium|high",
  "sentiment": "positive|negative|neutral"
}
```

---

## 3. RAG — Retrieval Augmented Generation

The core pattern for grounding LLM outputs in real data.

```
User query
    ↓
Embed query → vector
    ↓
Search vector DB → top-k relevant chunks
    ↓
Inject chunks into prompt as context
    ↓
LLM generates grounded answer
    ↓
Return answer + source citations
```

### Why RAG vs Fine-tuning?
| Approach | When to use |
|----------|-------------|
| RAG | Knowledge changes often, need citations, reduce hallucinations |
| Fine-tuning | Change model tone/style, domain-specific vocabulary, task formatting |
| Both | High accuracy domains like legal, medical |

---

## 4. Embeddings + Vector Search

```python
from openai import OpenAI
client = OpenAI()

# Create embedding
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="customer complaint about billing"
)
vector = response.data[0].embedding  # list of 1536 floats

# Cosine similarity
import numpy as np
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### Vector DB options
| DB | Best for |
|----|----------|
| pgvector | Already on Postgres, simple setup |
| Chroma | Local dev, prototyping |
| Pinecone | Managed, scalable, easy |
| Weaviate | Rich filtering + vectors |
| Milvus | Self-hosted, high scale |

---

## 5. Agents and Tool Use

An agent is an LLM that decides when and how to call tools.

```
User: "What's the status of order #12345?"
    ↓
LLM: I need to call get_order_status(order_id="12345")
    ↓
Tool executes, returns JSON
    ↓
LLM: "Your order #12345 is in transit, expected July 10."
```

### Key design decisions
1. When does the agent hand off to a human?
2. How do you handle tool call failures and retries?
3. How do you prevent prompt injection in tool responses?
4. How do you log agent decisions for debugging?

---

## 6. LLM Evaluation

No single metric — use a rubric or LLM-as-judge pattern.

| Dimension | Description |
|-----------|-------------|
| Groundedness | Answer supported by retrieved context |
| Faithfulness | No hallucinated facts |
| Relevance | Answer addresses the question |
| Completeness | Covers all parts of the question |
| Latency | Time to first token / total response time |
| Cost | Tokens used × price per token |

```python
# LLM-as-judge pattern
eval_prompt = """
Given the question, context, and answer below, rate the answer from 1-5 on:
- Groundedness (is the answer supported by the context?)
- Completeness (does it fully answer the question?)

Question: {question}
Context: {context}
Answer: {answer}

Reply with JSON: {"groundedness": X, "completeness": X, "reasoning": "..."}
"""
```

---

## Personal Notes
- RAG quality depends more on chunking and retrieval than on the LLM itself
- Always evaluate before optimizing — measure, don't guess
- TODO: Build an evaluation harness before starting P4 (Resume Screener)
