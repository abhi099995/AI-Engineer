# P5: DevOps Copilot

> **Phase:** LLM Agent | **Status:** ⏳ Upcoming (Month 4-5)

## Problem Statement

On-call engineers spend significant time on repetitive diagnostics: checking pod status, reading logs,
querying dashboards, and looking up runbooks. This project builds a DevOps Copilot that:
1. Accepts natural language queries from engineers
2. Calls real tools (kubectl, logs, metrics, runbook search)
3. Reasons across tool outputs
4. Returns actionable answers with evidence

## Business Context

- Reduces mean-time-to-detect (MTTD) and mean-time-to-resolve (MTTR)
- Frees senior engineers from routine on-call queries
- Builds institutional knowledge into a queryable system

## Learning Goals

- [ ] Build a tool-using agent from scratch (not a framework wrapper)
- [ ] Handle multi-step tool calling sequences
- [ ] Implement retry logic and graceful tool failure handling
- [ ] Prevent prompt injection from tool outputs
- [ ] Log every agent decision for auditability
- [ ] Implement human handoff when confidence is low

## Agent Architecture

```
User query: "Why is the payment-service pod restarting?"
    ↓
Agent: plan execution
    Step 1: get_pod_status(service="payment-service")
    Step 2: get_recent_logs(service="payment-service", lines=100)
    Step 3: search_runbooks(query="OOMKilled payment service")
    ↓
Synthesize: root cause + recommended action
    ↓
"The payment-service pod is OOMKilled (3 restarts in 1h).
 Logs show heap size exceeded 512Mi at 14:32.
 Runbook suggests increasing memory limit to 1Gi.
 Recommended action: kubectl set resources deployment/payment-service
 --limits=memory=1Gi"
```

## Tool Definitions

```python
tools = [
    {
        "name": "get_pod_status",
        "description": "Get current status of a Kubernetes pod or deployment",
        "parameters": {
            "service": "string — name of the service",
            "namespace": "string — k8s namespace (default: production)"
        }
    },
    {
        "name": "get_recent_logs",
        "description": "Retrieve recent logs for a service",
        "parameters": {
            "service": "string",
            "lines": "integer — number of log lines (default: 50)"
        }
    },
    {
        "name": "get_metrics",
        "description": "Get CPU/memory/latency metrics for a service from the last N minutes",
        "parameters": {
            "service": "string",
            "minutes": "integer"
        }
    },
    {
        "name": "search_runbooks",
        "description": "Search internal runbooks for relevant procedures",
        "parameters": {
            "query": "string — natural language query"
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Escalate the issue to a human engineer with a summary",
        "parameters": {
            "summary": "string",
            "severity": "low|medium|high|critical"
        }
    }
]
```

## Tech Stack

- `openai` — LLM with function calling
- `FastAPI` — API + WebSocket for streaming
- `httpx` — async HTTP calls to mock/real tool backends
- `pydantic` — input/output validation
- `structlog` — structured audit logging
- `pytest` — testing agent behavior

## Project Structure

```
p5-devops-copilot/
├── mock_backends/
│   ├── k8s_mock.py          # simulated kubectl responses
│   ├── logs_mock.py         # simulated log data
│   └── runbook_mock.py      # simulated runbook search
├── src/
│   ├── agent.py             # core agent loop
│   ├── tools.py             # tool registry + execution
│   ├── safety.py            # prompt injection detection
│   └── audit_log.py         # structured decision logging
├── api/
│   └── main.py
├── tests/
│   ├── test_agent.py
│   └── test_tool_failures.py
├── requirements.txt
└── README.md
```

## Security Considerations

Tool outputs can contain adversarial content. Mitigations:
- Sanitize tool responses before injecting into the prompt
- Limit tool output length with hard caps
- Detect prompt injection patterns: "Ignore previous instructions..."
- Never execute code from tool responses without explicit user confirmation
- Audit log every tool call with input, output, and agent reasoning

## Key Metrics

| Metric | Target |
|--------|--------|
| Task completion rate | > 80% |
| Incorrect escalations | < 10% |
| Prompt injection detection rate | > 95% |
| Mean time to answer | < 30s |
| Tool call audit coverage | 100% |

## Milestones

- [ ] Week 1: Tool definitions, mock backends, simple single-step agent
- [ ] Week 2: Multi-step reasoning, retry logic, tool failure handling
- [ ] Week 3: Safety layer, audit logging, streaming API
- [ ] Week 4: Test suite, evaluation scenarios, write-up

## Extension Ideas

- Slack slash command integration (`/ask-devops why is payment-service slow?`)
- Knowledge base auto-population: agent writes successful diagnoses back to runbooks
- Multi-agent: specialist agents for k8s, databases, networking
