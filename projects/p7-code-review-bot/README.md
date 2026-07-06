# P7: Code Review Bot

> **Phase:** GenAI + GitHub API | **Status:** ⏳ Upcoming (Month 5)

## Problem Statement

Code review is a bottleneck in most engineering teams. Senior engineers spend hours reviewing PRs.
This project builds an AI code review bot that:
1. Triggers on every PR via GitHub Actions
2. Analyzes the diff for bugs, security issues, and code quality
3. Posts specific, grounded inline comments
4. Learns from reviewer feedback to improve over time

## Business Context

- Catches common bugs before human review (saves senior engineer time)
- Enforces coding standards without manual checklists
- Especially valuable for security — catches OWASP top 10 patterns

## Learning Goals

- [ ] GitHub Actions integration
- [ ] Diff parsing and code context extraction
- [ ] Security-aware prompt design
- [ ] Grounded inline comment generation
- [ ] Feedback loop: thumbs up/down on AI comments

## Architecture

```
PR opened/updated
    ↓
GitHub Actions trigger
    ↓
Fetch diff + file context (GitHub API)
    ↓
For each changed file:
    Parse diff into hunks
    Build context window: changed lines + surrounding context
    ↓
    Analysis prompt:
    - Bug detection
    - Security issues (injection, auth, secrets)
    - Code style and complexity
    - Missing error handling
    ↓
    Generate inline comments (file + line + comment)
    ↓
Post review via GitHub API:
    - Inline comments on specific lines
    - Summary review comment
    - Approve / Request Changes / Comment
```

## Tech Stack

- `openai` — LLM analysis
- `PyGithub` or `httpx` + GitHub REST API
- GitHub Actions (`.github/workflows/`)
- `pydantic` — structured comment output
- `pytest` — test against synthetic diffs

## Project Structure

```
p7-code-review-bot/
├── .github/
│   └── workflows/
│       └── code-review.yml
├── src/
│   ├── diff_parser.py       # parse GitHub diff format
│   ├── context_builder.py   # extract surrounding code context
│   ├── reviewer.py          # LLM analysis logic
│   ├── comment_poster.py    # GitHub API integration
│   └── security_checks.py  # OWASP-focused prompt patterns
├── tests/
│   ├── sample_diffs/
│   └── test_reviewer.py
├── requirements.txt
└── README.md
```

## Review Categories

| Category | Examples |
|----------|---------|
| Security | SQL injection, hardcoded secrets, unvalidated input, broken auth |
| Bugs | Off-by-one errors, null dereference, wrong comparison operator |
| Performance | N+1 queries, missing indexes, synchronous I/O in async context |
| Code quality | Magic numbers, duplicate code, overly complex functions |
| Testing | Missing edge case tests, test with no assertions |

## Security-Aware Prompt Design

```python
system_prompt = """
You are a security-focused code reviewer. For each code hunk:
1. Check for OWASP Top 10 vulnerabilities
2. Check for logic bugs and null safety
3. Check for hardcoded credentials or secrets
4. Suggest specific fixes, not vague recommendations

IMPORTANT: Only comment on what you can see in the diff.
Do not invent issues that are not visible in the provided code.
Format each issue as JSON with file, line, severity, and suggestion.
"""
```

## Key Metrics

| Metric | Target |
|--------|--------|
| True positive rate on known issues | > 0.70 |
| False positive rate | < 0.20 |
| Developer thumbs-up rate | > 0.65 |
| P95 review time per PR | < 90s |

## Milestones

- [ ] Week 1: GitHub Actions setup, diff parsing, first prompts
- [ ] Week 2: Inline comment posting, security checks
- [ ] Week 3: Feedback collection, prompt improvement loop
- [ ] Week 4: Test against real repos, write-up

## Extension Ideas

- Language-specific rules (Python, TypeScript, Go)
- Complexity scoring per function/file
- Historical analysis: "this pattern caused 3 incidents in the last year"
- Integration with SonarQube or Semgrep for combined static + AI analysis
