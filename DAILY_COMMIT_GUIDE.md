# Daily Commit Guide

> Goal: maintain a green GitHub activity graph while building real AI skills.

---

## Why Daily Commits Matter

1. Builds public proof of consistent learning
2. Forces you to break work into small, shippable units
3. Creates a searchable log of your learning journey
4. Signals discipline to recruiters

---

## Commit Categories

Use these prefixes to keep commits meaningful and searchable:

| Prefix | Use case | Example |
|--------|----------|---------|
| `learn:` | Study notes, concept summaries | `learn: add numpy broadcasting notes` |
| `exp:` | Experiments, notebooks, scratch code | `exp: compare lr vs rf on credit data` |
| `feat:` | Project feature implementation | `feat: add feature engineering pipeline p1` |
| `fix:` | Bug fixes in project code | `fix: handle missing values in age column` |
| `eval:` | Evaluation results and analysis | `eval: xgboost roc-auc 0.87 on test set` |
| `docs:` | README updates, diagrams, decisions | `docs: add tradeoff table to p1 readme` |
| `refactor:` | Code cleanup without behavior change | `refactor: extract feature transforms to pipeline.py` |
| `test:` | Test additions | `test: add prediction api endpoint tests` |

---

## Daily 15-Minute Commit Routine

Even on non-coding days, you can always commit something:

### Option A — Study notes (10 min)
Write 5-10 bullet points from what you read or watched today.
Save to `notes/phase-X-topic/XX-topic.md`.

### Option B — Code experiment (15 min)
Run a quick experiment: try a new sklearn parameter, test a prompt variant, explore a dataset column.
Save to `experiments/`.

### Option C — Project progress (any time)
Push whatever you worked on. Even a partial function.

### Option D — Decision log (5 min)
Document a tradeoff or decision made for a project.
Add a row to the Tradeoffs table in the project README.

---

## Weekly Themes

| Week | Focus | Commit goal |
|------|-------|-------------|
| 1 | NumPy + Pandas | 5+ commits: notes + experiments |
| 2 | Data EDA on real dataset | 5+ commits: EDA notebook progress |
| 3 | Baseline ML model | 5+ commits: training script + eval |
| 4 | P1 FastAPI + Docker | 5+ commits: API + tests |
| Ongoing | Notes + project progress | 1 commit/day minimum |

---

## GitHub Profile Tips

1. Make this repository public so commits appear on your profile
2. Pin this repository on your GitHub profile
3. Add a profile README (`github.com/username`) that links to this repo
4. Star and watch repos from teams you want to work at (ML signal)

---

## What NOT to do

- Do not make empty commits (`git commit --allow-empty`)
- Do not commit 30 files at once with "WIP" — split into meaningful units
- Do not commit `.env` or data files — check `.gitignore`
- Do not squash all your commits — the history is the point

---

## Quick Git Commands

```bash
# Daily workflow
git add notes/phase-1-foundations/01-numpy-basics.md
git commit -m "learn: add numpy array indexing and broadcasting notes"
git push

# Check what's changed
git status
git diff

# See your commit history
git log --oneline -20
```
