# EchoCheck — FDE Framework Build Notes

## Part 1 — Problem Reframing

| Step | Output |
|---|---|
| Ordinary problem | "Score an LLM's answers against expected answers." |
| Lever 1: Data liquidity | Every eval run is thrown away after it prints. Human judgement about borderline answers ("is CO2 an acceptable answer?") lives in Slack threads, not in the grader. |
| Lever 2: Network effect | Each adjudication improves grading for every future run and every model tested on that suite. Run history turns single numbers into trends, flaky-question detection, and regression baselines. |
| Lever 3: Algorithmic leverage | Routed grading: numeric comparison with units, alias containment with negation guard, then LLM judge, then strict fuzzy. The grader says which method decided and why. |
| Lever 4: First principles × JTBD | The team's job is not "get an accuracy number". It is "know whether this model or prompt change is safe to ship". An eval nobody trusts, or that silently passes a 10× error, fails that job. |
| Extraordinary problem | **Build a continuous evaluation system whose grader is itself evaluated**, that remembers every run, blocks regressions against a promoted baseline, and gets more accurate every time a human rules on an edge case. |
| Rating | 4 / 5 |

## Part 2 — Design the Eval (of the evaluator)

**Outcome of intelligence**
- The grader agrees with human labels on 25/25 labelled grading cases, with zero false passes on numeric errors.
- A degraded model run is blocked against the baseline.
- An adjudication changes the verdict of the same answer on regrade.

**EQ(PRE)**
| Angle | Hypothesis |
|---|---|
| Causal | String similarity is a poor stand-in for correctness: "10,000" vs "1,000" scores 0.91 and passes. |
| Context | The grader never sees what humans already decided about acceptable variants. |
| Consistency | Accuracy with no history can't tell a flaky question from a regressed model. |

**EQ(POST), cheapest first**
1. Prompt: the LLM judge must return `{correct: bool, reason}`. Anything else falls back.
2. Context/retrieval: numeric and unit routing, adjudicated aliases and rejections, rubric citations.
3. Feedback: adjudication updates the answer key, regrade shows the effect, and run history drives flaky and regression detection.

**Executable:** `python run_evals.py` runs 27 cases × 3 runs (grader, regression, adjudication). CI fails below 100%.

## Part 3 — Gen-4 Architecture

| Layer | What EchoCheck does |
|---|---|
| Retrieval | `knowledge/grading_rubric.md`. Each run cites the rubric sections for the grading methods it used, plus the regression gate. |
| Context | Suite, model label, provider (live, offline or recorded responses), and the promoted baseline run. |
| Memory | Every run with raw answers, so it can be regraded; per-suite baselines; per-question answer keys (aliases and rejections); adjudication count. |
| Feedback | `POST /adjudicate` teaches the answer key. `POST /runs/{id}/regrade` re-scores stored answers. `GET /suites/{name}/flaky` mines history. Promotion sets the regression baseline. |

The model: live providers generate answers (`provider=anthropic|openai`) and act as the judge for open-ended answers. Offline, the deterministic stand-in and recorded `responses` keep the system usable without keys.

## Part 4 — Implementation Plan

| Component | Priority | Estimate | Status |
|---|---|---|---|
| Routed grader (numeric/unit, alias + negation, judge, fuzzy) | MVP | 1 day | done |
| Labelled grading set (25 cases) + meta-eval | MVP | 0.25 day | done |
| Suites, runs, recorded responses, run memory | MVP | 0.5 day | done |
| Baseline promotion + regression gate | MVP | 0.25 day | done |
| Adjudication → answer key → regrade | MVP | 0.25 day | done |
| Flaky-question detection | MVP | 0.1 day | done |
| API, Docker, CI | MVP | 0.5 day | done |
| Semantic similarity for long answers (embeddings) | nice-to-have | 1 day | future |
| Dashboard of accuracy/consistency trends per model | nice-to-have | 1 day | future |
| GitHub Action that runs a suite on every prompt change | future | 0.5 day | future |

## Part 5 — Prove It

**Visible change** (`python compare_v1_v2.py`, output in `run_output_v1_vs_v2.txt`):

| | v1 matcher | v2 grader |
|---|---|---|
| Agreement with human labels (25 cases) | 14 / 25 | **25 / 25** |
| False passes (e.g. "10,000" for "1,000") | 1 | **0** |
| False fails ("The answer is 96.", "100°C", "About 11 m/s", "Mumbai (formerly Bombay)") | 10 | **0** |
| Remembers runs, blocks regressions, learns from rulings | no | yes |

An outside observer would see an eval that stops failing correct answers written as sentences or with units, stops passing an order-of-magnitude error, and refuses to promote a model that broke questions the baseline got right.
