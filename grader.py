"""
Grader v2 — decides whether a model answer matches the expected answer, and
says *how* it decided (so every verdict is auditable).

v1 used one fuzzy string ratio (>= 0.9) for everything. That fails both ways:
    "100°C"  vs "100"       -> 0.86, marked WRONG (false fail)
    "Austen" vs "Jane Austen" -> 0.71, marked WRONG (false fail)
    "1,000"  vs "10,000"    -> 0.91, marked RIGHT (false pass: 10x error)

v2 routes each answer to a method, in order:
    numeric   expected is a number (+ optional unit): compare values exactly
              (0.5% relative tolerance), units must be compatible if present
    alias     normalised answer equals, or briefly contains, the expected
              answer or any human-adjudicated alias (learned via feedback)
    judge     live LLM-as-judge when an API key is configured
    fuzzy     offline fallback, only for non-numeric strings (v1 behaviour)
Negated mentions ("not Paris") never count as containing the answer.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

UNIT_ALIASES = {
    "m/s": {"m/s", "mps", "meters per second", "metres per second", "meter per second"},
    "km/h": {"km/h", "kmph", "kilometers per hour", "kilometres per hour"},
    "c": {"c", "°c", "celsius", "degrees celsius", "degree celsius", "degrees c"},
    "f": {"f", "°f", "fahrenheit", "degrees fahrenheit"},
    "kg": {"kg", "kilogram", "kilograms", "kilo", "kilos"},
    "%": {"%", "percent", "per cent"},
    "$": {"$", "usd", "dollars", "dollar"},
    "₹": {"₹", "inr", "rupees", "rupee", "rs"},
}
_NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")
_NEG = re.compile(r"\b(not|isn't|isnt|never|no)\b")
_ARTICLES = re.compile(r"\b(the|a|an)\b")


@dataclass
class Verdict:
    correct: bool
    method: str
    detail: str = ""

    def as_dict(self) -> dict:
        return {"correct": self.correct, "method": self.method, "detail": self.detail}


def normalise(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\"'`“”‘’]", "", s)
    s = re.sub(r"[.!?;:]+$", "", s)
    s = _ARTICLES.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def _numbers(s: str) -> list[float]:
    out = []
    for m in _NUM.findall(s):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _unit(s: str) -> str | None:
    t = " " + s.lower() + " "
    for canon, names in UNIT_ALIASES.items():
        for n in sorted(names, key=len, reverse=True):
            if re.search(r"(?<![a-z])" + re.escape(n) + r"(?![a-z])", t):
                return canon
    return None


def is_numeric_expected(expected: str) -> bool:
    nums = _numbers(expected)
    stripped = _NUM.sub("", expected.lower())
    for names in UNIT_ALIASES.values():
        for n in names:
            stripped = stripped.replace(n, "")
    return len(nums) == 1 and not re.search(r"[a-z]{3,}", re.sub(r"\b(about|approx|approximately)\b", "", stripped))


def grade_numeric(answer: str, expected: str, rel_tol: float = 0.005) -> Verdict:
    target = _numbers(expected)[0]
    got = _numbers(answer)
    if not got:
        return Verdict(False, "numeric", "no number in answer")
    distinct = {round(g, 9) for g in got}
    hit = any(abs(g - target) <= max(1e-9, abs(target) * rel_tol) for g in distinct)
    if not hit:
        return Verdict(False, "numeric", f"expected {target:g}, got {sorted(distinct)}")
    if len(distinct) > 1:
        return Verdict(False, "numeric", f"hedged between several values {sorted(distinct)}")
    eu, au = _unit(expected), _unit(answer)
    if eu and au and eu != au:
        return Verdict(False, "numeric", f"unit mismatch: expected {eu}, got {au}")
    return Verdict(True, "numeric", f"value {target:g} matched" + (f" ({au})" if au else ""))


def grade_alias(answer: str, accepted: list[str]) -> Verdict | None:
    a = normalise(answer)
    for alias in accepted:
        n = normalise(alias)
        if not n:
            continue
        if a == n:
            return Verdict(True, "alias", f"exact match '{alias}'")
        if len(a.split()) <= max(8, len(n.split()) + 5) and re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", a):
            window = a[:a.find(n)]
            if _NEG.search(window.split(",")[-1][-20:]):
                return Verdict(False, "alias", f"'{alias}' mentioned but negated")
            return Verdict(True, "alias", f"contains '{alias}'")
    return None


def grade(answer: str, expected: str, aliases: list[str] | None = None, llm=None,
          question: str = "") -> Verdict:
    if answer is None or not str(answer).strip():
        return Verdict(False, "empty", "no answer")
    if is_numeric_expected(expected):
        return grade_numeric(answer, expected)
    accepted = [expected] + list(aliases or [])
    v = grade_alias(answer, accepted)
    if v is not None:
        return v
    if llm is not None and llm.is_live:
        res = llm.complete_json(
            f"Question: {question}\nReference answer: {expected}\nAccepted aliases: {aliases or []}\n"
            f"Candidate answer: {answer}\nIs the candidate answer correct? "
            'Return {"correct": true|false, "reason": "..."}',
            offline=lambda: {"correct": None},
            validate=lambda d: isinstance(d.get("correct"), bool))
        if isinstance(res.get("correct"), bool):
            return Verdict(res["correct"], "judge", res.get("reason", "")[:200])
    ratio = difflib.SequenceMatcher(None, normalise(answer), normalise(expected)).ratio()
    return Verdict(ratio >= 0.9, "fuzzy", f"similarity {ratio:.2f}")


def grade_v1(answer: str, expected: str) -> bool:
    """The original EchoCheck matcher, kept for before/after comparison."""
    from eval_harness import answers_match
    return answers_match(answer, expected)
