#!/usr/bin/env python3
"""
EchoCheck eval gate -- evaluating the evaluator. Segments:
  grader       25 labelled (expected, answer, correct?) pairs; the grader must agree
  regression   a degraded model run is blocked against a promoted baseline
  adjudication a human ruling (CO2 == carbon dioxide) changes future verdicts
"""

import json

from gen4 import KnowledgeBase, LLMClient, Memory
from gen4.evals import EvalCase, gate, print_and_exit, run_eval
from grader import grade
from service import EchoCheckService


def fresh():
    return EchoCheckService(Memory(), KnowledgeBase.from_dir("knowledge"), LLMClient(provider="offline"))


def cases():
    out = []
    for i, c in enumerate(json.load(open("sample_data/grading_cases.json"))):
        out.append(EvalCase(f"g{i+1:02d}-{c['note']}", ("grade", c["answer"], c["expected"]), c["correct"], "grader"))
    out.append(EvalCase("regression-blocked", ("regression",), "block", "regression"))
    out.append(EvalCase("adjudication-learns", ("adjudicate",), (False, True), "adjudication"))
    return out


def system(inp):
    if inp[0] == "grade":
        return grade(inp[1], inp[2]).correct
    svc = fresh()
    if inp[0] == "regression":
        good = svc.run("sample", "model-v1", runs=3)
        svc.promote("sample", good["run_id"])
        broken = {q["id"]: ["I don't know"] * 3 for q in svc.suite("sample")[:3]}
        broken.update({q["id"]: q["answers"] for q in good["per_question"][3:]})
        return svc.run("sample", "model-v2", runs=3, responses=broken)["regression"]["verdict"]
    if inp[0] == "adjudicate":
        svc.save_suite("chem", [{"id": "c1", "question": "Gas plants absorb?", "expected_answer": "carbon dioxide"}])
        r = svc.run("chem", "m", runs=1, responses={"c1": ["CO2"]})
        svc.adjudicate("chem", "c1", "CO2", True)
        return (r["accuracy"] == 1.0, svc.regrade(r["run_id"])["after"] == 1.0)
    raise ValueError(inp)


if __name__ == "__main__":
    report = run_eval(cases(), system, runs=3)
    ok, why = gate(report, min_accuracy=1.0, min_consistency=1.0)
    print_and_exit("EchoCheck", report, ok, why)
