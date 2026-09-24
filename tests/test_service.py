import pytest

from gen4 import KnowledgeBase, LLMClient, Memory
from grader import grade
from service import EchoCheckService


@pytest.fixture
def svc():
    return EchoCheckService(Memory(), KnowledgeBase.from_dir("knowledge"), LLMClient(provider="offline"))


def test_numeric_grading_catches_order_of_magnitude_error():
    assert grade("10,000", "1,000").correct is False
    assert grade("1,000", "1000").correct is True


def test_units_must_be_compatible():
    assert grade("About 11 m/s", "11 meters per second").correct
    assert not grade("11 km/h", "11 meters per second").correct


def test_negated_mention_is_not_a_match():
    assert not grade("It is not Paris, it's Lyon.", "Paris").correct


def test_sample_suite_seeded_and_run_is_remembered(svc):
    r = svc.run("sample", "offline-model", runs=3)
    assert 0 < r["accuracy"] < 1 and r["regression"]["verdict"] == "no_baseline"
    assert svc.memory.get_decision(r["run_id"])["output"]["model_label"] == "offline-model"
    assert r["trace"]["retrieval"], "rubric sections must be cited"


def test_regression_gate(svc):
    good = svc.run("sample", "v1", runs=3)
    svc.promote("sample", good["run_id"])
    same = svc.run("sample", "v1-again", runs=3)
    assert same["regression"]["verdict"] == "promote_candidate"
    bad = {q["id"]: ["no idea"] * 3 for q in svc.suite("sample")}
    assert svc.run("sample", "v2", runs=3, responses=bad)["regression"]["verdict"] == "block"


def test_adjudication_is_learned_and_regrade_reflects_it(svc):
    svc.save_suite("chem", [{"id": "c1", "question": "q", "expected_answer": "carbon dioxide"}])
    r = svc.run("chem", "m", runs=2, responses={"c1": ["CO2", "Carbon dioxide"]})
    assert r["accuracy"] == 0.5
    svc.adjudicate("chem", "c1", "CO2", True)
    assert svc.regrade(r["run_id"])["after"] == 1.0
    svc.adjudicate("chem", "c1", "CO2", False)       # a later ruling can reverse it
    assert svc.regrade(r["run_id"])["after"] == 0.5


def test_flaky_questions_surface_from_history(svc):
    svc.save_suite("f", [{"id": "a", "question": "q", "expected_answer": "x"}])
    for _ in range(2):
        svc.run("f", "m", runs=2, responses={"a": ["x", "y"]})
    assert svc.flaky("f")[0]["id"] == "a"


def test_failed_model_calls_are_scored_not_crashed(svc):
    svc.save_suite("e", [{"id": "a", "question": "q", "expected_answer": "x"}])
    r = svc.run("e", "m", runs=2, responses={"a": []})
    assert r["accuracy"] == 0.0
