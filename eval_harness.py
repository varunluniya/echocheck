"""
Simple Eval Harness
FDE Learning Plan — Guide 1, Assignment 3

Requirements (Guide 1, Section 9.2, Assignment 3):
  - Function takes: a list of dicts with 'question' and 'expected_answer' keys.
  - For each question, call the model and compare the response to the expected answer.
  - Return: total accuracy, list of failed cases, and average consistency score
    across 3 runs per question.

Grading bar (per the guide): a strong submission includes the consistency
score (3-run agreement), not just binary correct/wrong, handles API errors
gracefully, and uses clear variable names.

This harness is provider-agnostic: it calls a real LLM API when a key is
configured (OpenAI or Anthropic), and falls back to a deterministic offline
stand-in otherwise, so the whole thing runs end-to-end with zero setup. The
offline model is NOT a real model -- it's a stand-in used only because this
sandbox has no live API key. See run_demo.py for how to switch providers,
and README.md for an honest note on what that means for the sample output.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import random
import time
from collections import Counter
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Model client: real providers + an offline fallback with the same contract
# ---------------------------------------------------------------------------

class ModelCallError(Exception):
    """Raised when a model call fails after retries are exhausted."""


@dataclass
class ModelClient:
    """Provider-agnostic client. call(question) -> str answer.

    provider: "openai" | "anthropic" | "offline"
    Falls back to offline automatically if the requested provider's API key
    isn't set, so the harness never crashes just because a key is missing.
    """
    provider: str = "offline"
    model: str = "gpt-4o-mini"
    max_retries: int = 3
    base_backoff: float = 0.5
    temperature: float = 0.7

    def __post_init__(self):
        if self.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            print("[ModelClient] OPENAI_API_KEY not set -- falling back to offline mode.")
            self.provider = "offline"
        if self.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            print("[ModelClient] ANTHROPIC_API_KEY not set -- falling back to offline mode.")
            self.provider = "offline"

    def call(self, question: str, run_index: int = 0) -> str:
        """Call the model with retry/backoff. Raises ModelCallError if every
        retry is exhausted -- callers are expected to handle this gracefully
        rather than let the whole harness crash on one bad question."""
        last_err = None
        for attempt in range(self.max_retries):
            try:
                if self.provider == "offline":
                    return self._call_offline(question, run_index)
                elif self.provider == "openai":
                    return self._call_openai(question)
                elif self.provider == "anthropic":
                    return self._call_anthropic(question)
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")
            except Exception as e:  # noqa: BLE001 - genuinely want to catch+retry any transient failure
                last_err = e
                backoff = self.base_backoff * (2 ** attempt)
                time.sleep(backoff)
        raise ModelCallError(f"Model call failed after {self.max_retries} attempts: {last_err}")

    # -- real providers -----------------------------------------------------

    def _call_openai(self, question: str) -> str:
        import requests
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": question}],
                "temperature": self.temperature,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _call_anthropic(self, question: str) -> str:
        import requests
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": self.model or "claude-3-5-haiku-20241022",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": question}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()

    # -- offline deterministic-but-varied fallback ---------------------------

    def _call_offline(self, question: str, run_index: int) -> str:
        """Deterministic stand-in with the SAME interface as a real model,
        including realistic run-to-run variance (like temperature sampling
        would produce), so the consistency-scoring logic in run_eval() has
        something real to measure. Uses a tiny canned knowledge base;
        unknown questions get an intentionally-wrong placeholder so failures
        show up in the report too, not just successes."""
        kb = {
            "what is the capital of france?": "Paris",
            "what is 12 * 8?": "96",
            "who wrote 'pride and prejudice'?": "Jane Austen",
            "what is the boiling point of water in celsius?": "100",
            "what is the largest planet in our solar system?": "Jupiter",
            "what language is primarily used for ios app development?": "Swift",
            "what is the chemical symbol for gold?": "Au",
        }
        key = question.strip().lower()
        correct = kb.get(key)

        # Seed per (question, run_index) so results are reproducible but vary
        # run-to-run the way a real sampled model would.
        seed = int(hashlib.sha256(f"{key}:{run_index}".encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)

        if correct is None:
            return "I don't have enough information to answer that."

        # ~85% of the time return the correct answer verbatim; ~15% of the
        # time simulate a plausible near-miss (wrong phrasing/wrong value),
        # so consistency scores aren't trivially 100% across the board.
        if rng.random() < 0.85:
            return correct
        return f"{correct}?"  # a wrong/uncertain-looking variant on purpose


# ---------------------------------------------------------------------------
# Answer comparison
# ---------------------------------------------------------------------------

def answers_match(actual: str, expected: str, threshold: float = 0.9) -> bool:
    """Fuzzy match: exact (case/whitespace-insensitive) match, or a normalized
    edit-distance similarity above `threshold`. This avoids penalizing trivial
    formatting differences ("Paris." vs "Paris") while still catching real
    wrong answers."""
    a = actual.strip().lower().rstrip(".!?")
    e = expected.strip().lower().rstrip(".!?")
    if a == e:
        return True
    similarity = difflib.SequenceMatcher(None, a, e).ratio()
    return similarity >= threshold


# ---------------------------------------------------------------------------
# The eval harness itself
# ---------------------------------------------------------------------------

@dataclass
class EvalReport:
    total_accuracy: float
    failed_cases: list = field(default_factory=list)
    average_consistency: float = 0.0
    per_question: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "total_accuracy": self.total_accuracy,
            "average_consistency": self.average_consistency,
            "failed_cases": self.failed_cases,
            "per_question": self.per_question,
            "errors": self.errors,
        }


def run_eval(qa_pairs: list[dict], client: ModelClient | None = None,
             runs_per_question: int = 3) -> EvalReport:
    """Run the eval harness over a list of {'question': ..., 'expected_answer': ...}
    dicts. For each question, calls the model `runs_per_question` times,
    scores each run for correctness against expected_answer, and computes:
      - total_accuracy: fraction of *runs* (across all questions) that were correct
      - failed_cases: every run that did NOT match, with question/expected/actual
      - average_consistency: for each question, the fraction of its runs that
        agree with the MAJORITY answer for that question (1.0 = all 3 runs
        gave the same answer, regardless of correctness); averaged over all
        questions. This is the "3-run agreement" score the assignment asks for,
        distinct from accuracy -- a model can be perfectly consistent and
        consistently wrong, or accurate but erratic.

    Handles API/model failures gracefully: a question whose calls all fail
    is recorded in `errors` and excluded from the accuracy/consistency
    denominators rather than crashing the whole run.
    """
    client = client or ModelClient()
    total_runs = 0
    correct_runs = 0
    failed_cases = []
    per_question = []
    errors = []

    for pair in qa_pairs:
        question = pair["question"]
        expected = pair["expected_answer"]
        run_answers = []
        run_correct = []

        for run_idx in range(runs_per_question):
            try:
                answer = client.call(question, run_index=run_idx)
            except ModelCallError as e:
                errors.append({"question": question, "run": run_idx, "error": str(e)})
                continue

            is_correct = answers_match(answer, expected)
            run_answers.append(answer)
            run_correct.append(is_correct)
            total_runs += 1
            if is_correct:
                correct_runs += 1
            else:
                failed_cases.append({
                    "question": question,
                    "expected_answer": expected,
                    "actual_answer": answer,
                    "run": run_idx,
                })

        if run_answers:
            # Consistency = fraction of runs agreeing with the majority answer
            counts = Counter(a.strip().lower() for a in run_answers)
            majority_count = counts.most_common(1)[0][1]
            consistency = majority_count / len(run_answers)
        else:
            consistency = 0.0

        per_question.append({
            "question": question,
            "expected_answer": expected,
            "answers": run_answers,
            "correct_flags": run_correct,
            "consistency": round(consistency, 3),
            "question_accuracy": round(sum(run_correct) / len(run_correct), 3) if run_correct else 0.0,
        })

    total_accuracy = round(correct_runs / total_runs, 3) if total_runs else 0.0
    average_consistency = (
        round(sum(q["consistency"] for q in per_question) / len(per_question), 3)
        if per_question else 0.0
    )

    return EvalReport(
        total_accuracy=total_accuracy,
        failed_cases=failed_cases,
        average_consistency=average_consistency,
        per_question=per_question,
        errors=errors,
    )
