# LLM Eval Harness

A small, provider-agnostic eval harness that measures both **accuracy** and
**run-to-run consistency** for any LLM API — not just binary right/wrong,
which is the common weak spot in quick evals.

```
pip install -r requirements.txt
python3 cli.py --input sample_data/questions.json
```

## Why consistency, not just accuracy

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

## Usage

```
python3 cli.py --input sample_data/questions.json                     # offline mode, no key needed
python3 cli.py --input sample_data/questions.json --provider openai   # needs OPENAI_API_KEY
python3 cli.py --input sample_data/questions.json --provider anthropic --runs 5
python3 cli.py --input sample_data/questions.json --output report.json
```

`--input` takes a JSON file: a list of `{"question": ..., "expected_answer": ...}`
objects (see `sample_data/questions.json`).

## Sample run (offline mode)

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

## Robustness

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

## Files

- `eval_harness.py` — `ModelClient` (OpenAI / Anthropic / offline, retry +
  backoff) and `run_eval()` (the core accuracy + consistency logic).
- `cli.py` — command-line entry point.
- `run_demo.py` — a plain Python usage example (no CLI).
- `sample_data/questions.json` — 7 sample Q&A pairs.
- `tests/test_eval_harness.py` — pytest suite (8 tests, including the
  forced-failure robustness check).
- `run_output.txt` — captured output from an actual run.

## Design notes

- **Consistency = majority-agreement across runs**, not average pairwise
  similarity — simpler to reason about and matches how a person would judge
  "did it basically say the same thing three times."
- **Accuracy is computed at the run level**, not the question level, so a
  question right 2-out-of-3 times contributes partial credit rather than
  being all-or-nothing.
- Correctness and consistency are kept as fully separate signals in the code
  on purpose — collapsing them into one score is the most common weak-eval
  pattern this project is built to avoid.

## Background

Started as Guide 1, Assignment 3 in a self-directed FDE (Forward Deployed
Engineer) learning program — "write working code that measures model output
consistency." Extended into a CLI + test suite for this portfolio version.
