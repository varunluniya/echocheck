# EchoCheck

_Continuous LLM evaluation: accuracy and consistency kept separate, a grader that is itself evaluated, regression gating against a baseline, and human rulings that make the grader smarter._

[![ci](https://github.com/varunluniya/echocheck/actions/workflows/ci.yml/badge.svg)](https://github.com/varunluniya/echocheck/actions/workflows/ci.yml)

## What's new in v2: an intelligent, deployable service

| Layer | In EchoCheck |
|---|---|
| **Retrieval** | Grading rubric sections behind each verdict method are cited in every run |
| **Context** | Suite, model label, provider (live, offline or recorded), promoted baseline |
| **Memory** | Every run with raw answers, baselines, and per-question answer keys (aliases and rejections) |
| **Feedback** | `POST /adjudicate` teaches the answer key, `regrade` shows the effect, run history surfaces flaky questions |

**Routed grader** (`grader.py`): numeric with units → alias with negation guard → LLM judge (live) → strict fuzzy.

| | v1 matcher | v2 grader |
|---|---|---|
| Agreement with 25 human-labelled grading cases | 14/25 | **25/25** |
| "10,000" graded against "1,000" | ✓ pass (false) | ✗ fail, value mismatch |
| "The answer is 96." / "100°C" / "About 11 m/s" | ✗ fail (false) | ✓ pass (numeric) |
| "It is not Paris, it's Lyon." | ✗ | ✗ (negated mention) |

Full framework write-up: [FRAMEWORK.md](FRAMEWORK.md).

## Run it

```bash
pip install -r requirements-dev.txt
uvicorn app:app --reload        # http://127.0.0.1:8000/docs
python -m pytest -q
python run_evals.py             # evaluates the evaluator: 27 cases x 3 runs
python compare_v1_v2.py         # before/after proof
python cli.py --input sample_data/questions.json   # v1 CLI still works
```

Docker: `docker build -t echocheck . && docker run -p 8000:8000 -v echocheck-data:/data echocheck` · Render: `render.yaml`.
Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` to test live models and enable the LLM judge.

```bash
curl -X POST localhost:8000/runs -H 'content-type: application/json' -d '{"suite":"sample","model_label":"baseline","runs":3}'
curl -X POST localhost:8000/suites/sample/baseline/<run_id>
curl -X POST localhost:8000/runs -H 'content-type: application/json' -d '{"suite":"sample","model_label":"candidate","provider":"anthropic"}'
```

| Endpoint | Purpose |
|---|---|
| `PUT /suites/{name}` · `GET /suites` | Manage question suites |
| `POST /runs` | Run a suite against a live model, the offline stand-in, or recorded `responses`. Returns accuracy, consistency, per-question verdicts with method, and a regression verdict |
| `POST /suites/{name}/baseline/{run_id}` | Promote a baseline |
| `POST /adjudicate` · `POST /runs/{id}/regrade` | Human ruling → answer key → re-score |
| `GET /suites/{name}/flaky` | Questions with unstable answers across runs |

## Demo data

`python seed.py` fills `data/echocheck.db` with synthetic history so every endpoint returns something meaningful on first run: A 20-question lending-ops suite plus the sample suite, seven runs of a fictional support bot across five versions (one promoted baseline, one blocked regression), two human adjudications, and flaky-question history.

```bash
python seed.py            # create data/echocheck.db
python seed.py --reset    # rebuild it from scratch
```

The Docker image seeds `/data` on first boot (set `GEN4_SEED=0` to start empty). All of it is synthetic: no real customers, patients, tickets or model outputs. `GET /health` shows the dataset's counts.

---

## The original engine (v1)

The v1 demo still runs unchanged; the service wraps it.

_A provider-agnostic LLM eval harness that measures accuracy and run-to-run consistency separately._


A small, provider-agnostic eval harness that measures both **accuracy** and
**run-to-run consistency** for any LLM API — not just binary right/wrong,
which is the common weak spot in quick evals.

```
pip install -r requirements.txt
python3 cli.py --input sample_data/questions.json
```

### Why consistency, not just accuracy

A model can be accurate on average while still being unreliable — right two
times out of three, or right every time but phrased differently each run.
Reporting accuracy alone hides that. This harness calls the model 3 times per
question (configurable) and reports both numbers separately:

- **Accuracy** — does the answer match the expected answer (fuzzy-matched, so
  trivial formatting differences don't count against it)?
- **Consistency** — across the 3 runs, do the answers agree with each other
  (majority agreement), independent of whether they're correct?

On the sample question set, several questions score 100% accuracy but only
66.7% consistency — the model got the right answer every time but phrased one
run differently. That gap is exactly what a consistency score is for.

### Usage

```
python3 cli.py --input sample_data/questions.json                     # offline mode, no key needed
python3 cli.py --input sample_data/questions.json --provider openai   # needs OPENAI_API_KEY
python3 cli.py --input sample_data/questions.json --provider anthropic --runs 5
python3 cli.py --input sample_data/questions.json --output report.json
```

`--input` takes a JSON file: a list of `{"question": ..., "expected_answer": ...}`
objects (see `sample_data/questions.json`).

### Sample run (offline mode)

| Metric | Value |
|---|---|
| Total accuracy | 85.7% (18/21 runs) |
| Average consistency | 81.0% |
| Errors | 0 |

Full captured output: `run_output.txt`. Full API usage example: `run_demo.py`.

**Note on offline mode:** this sandbox has no live API key, so the sample
numbers above come from a deterministic offline stand-in with realistic
run-to-run variance built in (see `ModelClient._call_offline` in
`eval_harness.py`) — not a real model. Swap `--provider openai` or
`--provider anthropic` with the matching API key set and the exact same code
runs against a live model; nothing else changes.

### Robustness

Every model call is wrapped in retry with exponential backoff. If a question
fails on every retry, it's recorded in the report's `errors` list — the
harness completes with a partial, honest report rather than crashing.
Verified directly with a forced-failure test (`tests/test_eval_harness.py::
test_run_eval_handles_repeated_api_failure_gracefully`): mocks the model call
to always raise, confirms all 3 runs are logged as errors and the harness
still returns cleanly.

```
pytest tests/
```

### Files

- `eval_harness.py` — `ModelClient` (OpenAI / Anthropic / offline, retry +
  backoff) and `run_eval()` (the core accuracy + consistency logic).
- `cli.py` — command-line entry point.
- `run_demo.py` — a plain Python usage example (no CLI).
- `sample_data/questions.json` — 7 sample Q&A pairs.
- `tests/test_eval_harness.py` — pytest suite (8 tests, including the
  forced-failure robustness check).
- `run_output.txt` — captured output from an actual run.

### Design notes

- **Consistency = majority-agreement across runs**, not average pairwise
  similarity — simpler to reason about and matches how a person would judge
  "did it basically say the same thing three times."
- **Accuracy is computed at the run level**, not the question level, so a
  question right 2-out-of-3 times contributes partial credit rather than
  being all-or-nothing.
- Correctness and consistency are kept as fully separate signals in the code
  on purpose — collapsing them into one score is the most common weak-eval
  pattern this project is built to avoid.

### Why this exists

Most "eval" scripts report a single right/wrong number and call it a day. That hides exactly the failure mode that matters most in practice: a model that's accurate on average but inconsistent run to run. This harness keeps the two signals separate on purpose, with a CLI and test suite built around that idea.
