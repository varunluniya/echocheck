"""Unit tests for the eval harness. Run with: pytest"""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval_harness import ModelClient, ModelCallError, answers_match, run_eval


def test_answers_match_exact():
    assert answers_match("Paris", "Paris")


def test_answers_match_case_and_punctuation_insensitive():
    assert answers_match("paris.", "Paris")
    assert answers_match("PARIS!", "paris")


def test_answers_match_rejects_wrong_answer():
    assert not answers_match("London", "Paris")


def test_answers_match_fuzzy_near_miss():
    # small formatting difference should still count as a match
    assert answers_match("Paris?", "Paris")


def test_run_eval_basic_accuracy():
    client = ModelClient(provider="offline")
    report = run_eval(
        [{"question": "What is the capital of France?", "expected_answer": "Paris"}],
        client=client, runs_per_question=3,
    )
    assert 0.0 <= report.total_accuracy <= 1.0
    assert 0.0 <= report.average_consistency <= 1.0
    assert len(report.per_question) == 1


def test_run_eval_reports_failed_cases_for_unknown_question():
    client = ModelClient(provider="offline")
    report = run_eval(
        [{"question": "What is the meaning of life?", "expected_answer": "42"}],
        client=client, runs_per_question=3,
    )
    assert report.total_accuracy == 0.0
    assert len(report.failed_cases) == 3


def test_run_eval_handles_repeated_api_failure_gracefully():
    """The core robustness requirement: a dead model must not crash the harness."""
    client = ModelClient(provider="offline", max_retries=2, base_backoff=0.01)
    with patch.object(ModelClient, "_call_offline", side_effect=RuntimeError("simulated outage")):
        report = run_eval(
            [{"question": "What is the capital of France?", "expected_answer": "Paris"}],
            client=client, runs_per_question=3,
        )
    assert len(report.errors) == 3
    assert report.total_accuracy == 0.0
    assert report.per_question[0]["consistency"] == 0.0


def test_model_client_falls_back_to_offline_without_api_key():
    os.environ.pop("OPENAI_API_KEY", None)
    client = ModelClient(provider="openai")
    assert client.provider == "offline"
