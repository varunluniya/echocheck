"""
EchoCheck intelligent service — a continuous LLM evaluation system.

    Retrieval  grading rubric sections behind each verdict method used
    Context    suite, model label, provider, runs, baseline in force
    Memory     every run, per-question results, promoted baselines,
               human-adjudicated aliases and rejections
    Feedback   adjudications change the answer key for all future grading;
               regrading a stored run shows the effect immediately; flaky
               questions surface from run history

v1 was a one-shot script with a single fuzzy matcher. v2 grades with
numeric/alias/judge/fuzzy routing (grader.py), remembers every run, blocks
regressions against a promoted baseline, and learns from human rulings.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from eval_harness import ModelCallError, ModelClient
from gen4 import Context, KnowledgeBase, LLMClient, Memory, Trace
from grader import grade, normalise

SYSTEM = "echocheck"
HERE = Path(__file__).parent
REGRESSION_DROP = 0.02
RUBRIC = {"numeric": "Numeric answers", "alias": "Named entities and short answers",
          "judge": "Open-ended answers", "fuzzy": "Open-ended answers",
          "adjudicated": "Accepted aliases", "empty": "Consistency", "error": "Consistency"}


class EchoCheckService:
    def __init__(self, memory: Memory | None = None, kb: KnowledgeBase | None = None,
                 llm: LLMClient | None = None):
        self.memory = memory or Memory(os.environ.get("GEN4_DB", HERE / "data" / f"{SYSTEM}.db"))
        self.kb = kb or KnowledgeBase.from_dir(HERE / "knowledge")
        self.llm = llm or LLMClient()
        if not self.suite("sample"):
            qs = json.loads((HERE / "sample_data" / "questions.json").read_text())
            self.save_suite("sample", [{"id": f"q{i+1}", **q} for i, q in enumerate(qs)])

    # -- suites & answer key -----------------------------------------------------
    def save_suite(self, name: str, cases: list[dict]) -> list[dict]:
        clean = []
        for i, c in enumerate(cases):
            clean.append({"id": str(c.get("id") or f"q{i+1}"), "question": c["question"],
                          "expected_answer": str(c["expected_answer"])})
        self.memory.add_fact(f"suite:{name}", "cases", clean, source="user")
        return clean

    def suite(self, name: str) -> list[dict] | None:
        return self.memory.fact(f"suite:{name}", "cases")

    def suites(self) -> list[str]:
        return [f["subject"].split(":", 1)[1] for f in self.memory.facts(predicate="cases")]

    def _key(self, suite: str, qid: str) -> dict:
        return self.memory.fact(f"suite:{suite}/q:{qid}", "answer_key",
                                {"aliases": [], "rejected": []})

    def adjudicate(self, suite: str, question_id: str, answer: str, correct: bool,
                   reviewer: str = "human") -> dict:
        if not any(c["id"] == question_id for c in (self.suite(suite) or [])):
            raise KeyError(question_id)
        key = self._key(suite, question_id)
        bucket = "aliases" if correct else "rejected"
        if normalise(answer) not in [normalise(a) for a in key[bucket]]:
            key[bucket].append(answer)
        other = "rejected" if correct else "aliases"
        key[other] = [a for a in key[other] if normalise(a) != normalise(answer)]
        self.memory.add_fact(f"suite:{suite}/q:{question_id}", "answer_key", key, source=reviewer)
        n = int(self.memory.fact("grader", "adjudications", 0) or 0) + 1
        self.memory.add_fact("grader", "adjudications", n)
        return key

    # -- grading -------------------------------------------------------------------
    def grade_one(self, suite: str, case: dict, answer: str):
        key = self._key(suite, case["id"])
        if any(normalise(answer) == normalise(r) for r in key["rejected"]):
            from grader import Verdict
            return Verdict(False, "adjudicated", "a human marked this answer wrong")
        if any(normalise(answer) == normalise(a) for a in key["aliases"]):
            from grader import Verdict
            return Verdict(True, "adjudicated", "a human accepted this answer")
        return grade(answer, case["expected_answer"], key["aliases"], llm=self.llm, question=case["question"])

    # -- runs ----------------------------------------------------------------------
    def run(self, suite: str, model_label: str, provider: str = "offline", runs: int = 3,
            responses: dict[str, list[str]] | None = None) -> dict:
        cases = self.suite(suite)
        if not cases:
            raise KeyError(suite)
        client = None if responses else ModelClient(provider=provider)
        answers: dict[str, list[str | None]] = {}
        for c in cases:
            if responses is not None:
                answers[c["id"]] = list(responses.get(c["id"], []))[:runs]
            else:
                got = []
                for r in range(runs):
                    try:
                        got.append(client.call(c["question"], run_index=r))
                    except ModelCallError:
                        got.append(None)
                answers[c["id"]] = got
        report = self._score(suite, cases, answers, runs)
        report.update({"suite": suite, "model_label": model_label,
                       "provider": "recorded" if responses is not None else client.provider})
        return self._store(report, answers)

    def _score(self, suite, cases, answers, runs) -> dict:
        per_q, hits, total, methods = [], 0, 0, Counter()
        for c in cases:
            got = answers.get(c["id"], [])
            verdicts = []
            for a in got:
                if a is None:
                    from grader import Verdict
                    v = Verdict(False, "error", "model call failed")
                else:
                    v = self.grade_one(suite, c, a)
                verdicts.append(v)
                methods[v.method] += 1
            ok = [v.correct for v in verdicts]
            hits += sum(ok)
            total += max(len(ok), runs)
            norm = [normalise(a) for a in got if a]
            cons = (Counter(norm).most_common(1)[0][1] / len(norm)) if norm else 0.0
            per_q.append({"id": c["id"], "question": c["question"], "expected": c["expected_answer"],
                          "answers": got, "verdicts": [v.as_dict() for v in verdicts],
                          "accuracy": round(sum(ok) / max(len(ok), runs), 4), "consistency": round(cons, 4)})
        return {"accuracy": round(hits / total, 4) if total else 0.0,
                "consistency": round(sum(q["consistency"] for q in per_q) / len(per_q), 4) if per_q else 0.0,
                "runs": runs, "per_question": per_q, "grading_methods": dict(methods)}

    def _store(self, report: dict, answers: dict) -> dict:
        suite = report["suite"]
        trace = Trace()
        passages = []
        for m in report["grading_methods"]:
            passages += self.kb.search(RUBRIC.get(m, m), k=1)
        passages += self.kb.search("regression gate baseline", k=1)
        trace.cite({(p.source, p.heading): p for p in passages}.values())
        base_id = self.memory.get_param(f"baseline:{suite}")
        trace.context = (Context().add("suite", suite).add("model_label", report["model_label"])
                         .add("provider", report["provider"]).add("baseline_run", base_id)).as_dict()
        trace.model = {"judge": self.llm.provider}
        comparison = self.compare(report, base_id)
        report["regression"] = comparison
        rid = self.memory.record_decision(SYSTEM, f"suite:{suite}",
                                          {"model_label": report["model_label"], "provider": report["provider"],
                                           "answers": answers}, {k: v for k, v in report.items()})
        history = self.memory.history(f"suite:{suite}")
        trace.memory = {"runs_on_suite": len(history), "adjudications": self.memory.fact("grader", "adjudications", 0)}
        return {"run_id": rid, **report, "trace": trace.as_dict()}

    def compare(self, report: dict, base_id: str | None) -> dict:
        if not base_id:
            return {"verdict": "no_baseline", "reasons": ["promote a run to set a baseline"]}
        base = self.memory.get_decision(base_id)["output"]
        reasons = []
        drop = base["accuracy"] - report["accuracy"]
        if drop > REGRESSION_DROP:
            reasons.append(f"accuracy dropped {drop:.1%} ({base['accuracy']:.1%} -> {report['accuracy']:.1%})")
        base_q = {q["id"]: q for q in base["per_question"]}
        for q in report["per_question"]:
            b = base_q.get(q["id"])
            if b and b["accuracy"] == 1.0 and q["accuracy"] < 0.5:
                reasons.append(f"{q['id']} was always right in baseline, now {q['accuracy']:.0%}")
        return {"verdict": "block" if reasons else "promote_candidate", "baseline_run": base_id,
                "baseline_accuracy": base["accuracy"], "reasons": reasons}

    def promote(self, suite: str, run_id: str) -> dict:
        d = self.memory.get_decision(run_id)
        if not d or d["subject"] != f"suite:{suite}":
            raise KeyError(run_id)
        self.memory.set_param(f"baseline:{suite}", run_id, f"promoted {d['features']['model_label']}")
        return {"suite": suite, "baseline_run": run_id, "accuracy": d["output"]["accuracy"]}

    def regrade(self, run_id: str) -> dict:
        """Re-score a stored run with the *current* answer key (after adjudications)."""
        d = self.memory.get_decision(run_id)
        if not d:
            raise KeyError(run_id)
        suite = d["subject"].split(":", 1)[1]
        report = self._score(suite, self.suite(suite), d["features"]["answers"], d["output"]["runs"])
        return {"run_id": run_id, "before": d["output"]["accuracy"], "after": report["accuracy"],
                "per_question": report["per_question"]}

    def flaky(self, suite: str, min_runs: int = 2) -> list[dict]:
        hist = [h for h in self.memory.history(f"suite:{suite}", limit=200)]
        seen: dict[str, list[float]] = {}
        for h in hist:
            for q in h["output"]["per_question"]:
                seen.setdefault(q["id"], []).append(q["consistency"])
        return [{"id": qid, "runs_seen": len(c), "avg_consistency": round(sum(c) / len(c), 3)}
                for qid, c in seen.items() if len(c) >= min_runs and sum(c) / len(c) < 1.0]
