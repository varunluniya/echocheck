#!/usr/bin/env python3
"""Prove-It: v1 matcher vs v2 grader on 25 labelled grading cases."""
import json

from grader import grade, grade_v1

cases = json.load(open("sample_data/grading_cases.json"))
v1_ok = v2_ok = fp1 = fn1 = 0
print(f"{'case':<28}{'truth':<7}{'v1':<7}{'v2':<7}method / detail")
for c in cases:
    a, b = grade_v1(c["answer"], c["expected"]), grade(c["answer"], c["expected"])
    v1_ok += a == c["correct"]
    v2_ok += b.correct == c["correct"]
    fp1 += a and not c["correct"]
    fn1 += (not a) and c["correct"]
    mark = lambda x: ("✓" if x else "✗")
    print(f"{c['note'][:27]:<28}{mark(c['correct']):<7}{mark(a) + ('' if a == c['correct'] else '!'):<7}"
          f"{mark(b.correct) + ('' if b.correct == c['correct'] else '!'):<7}{b.method}: {b.detail}")
print(f"\nv1 agreement with human labels: {v1_ok}/{len(cases)} (false passes {fp1}, false fails {fn1})")
print(f"v2 agreement with human labels: {v2_ok}/{len(cases)}")
