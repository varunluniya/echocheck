#!/usr/bin/env python3
"""
Seed EchoCheck's memory with 90 days of synthetic evaluation history.

    python seed.py [--reset] [--if-empty]

Adds a 20-question "lending-ops" suite (loan arithmetic with computed
answers, plus acronym/definition questions), then records five versions of
a fictional support bot answering it over three months:

    support-bot-v1.0   baseline quality
    support-bot-v1.1   better -- promoted to baseline
    support-bot-v1.2   a regression the gate blocks
    support-bot-v1.3   fixed, best so far
    support-bot-v1.3   re-run a week later (consistency check)

Answers are generated from the correct values with realistic variation
(sentences, units, thousands separators) and realistic errors (10x slips,
digit swaps, "not sure"). Two answers are adjudicated by a human, so the
answer key has learned aliases. All data is synthetic.
"""

from pathlib import Path

from gen4 import Memory
from gen4.seedkit import already_seeded, args, iso, mark
from service import EchoCheckService


def rupees(x):
    s = f"{int(round(x)):d}"
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail]) if parts else tail


def emi(p, annual, n):
    r = annual / 12 / 100
    return p * r * (1 + r) ** n / ((1 + r) ** n - 1)


NUMERIC = [  # (question, answer); rupee amounts are answered with rupee phrasing
    ("A loan of Rs 1,20,000 at 0% interest is repaid over 12 months. What is the monthly EMI in rupees?", 120000 / 12),
    ("What is a 2% processing fee on a loan of Rs 2,50,000, in rupees?", 5000),
    ("Simple interest for 1 year at 12% on Rs 50,000, in rupees?", 6000),
    ("EMI in rupees for Rs 1,00,000 at 12% annual interest over 12 months (rounded to the nearest rupee)?", round(emi(100000, 12, 12))),
    ("EMI in rupees for Rs 5,00,000 at 10.5% annual interest over 36 months (nearest rupee)?", round(emi(500000, 10.5, 36))),
    ("A borrower earns Rs 80,000 a month and pays Rs 24,000 in EMIs. What is their debt-to-income ratio in percent?", 30),
    ("A two-wheeler costs Rs 1,10,000 and the down payment is 20%. What is the loan amount in rupees?", 88000),
    ("GST of 18% on a processing fee of Rs 2,000 is how many rupees?", 360),
    ("A late fee of Rs 500 is charged on each of 3 missed EMIs. Total late fees in rupees?", 1500),
    ("Loan-to-value ratio in percent for a Rs 40,00,000 loan on a Rs 50,00,000 property?", 80),
    ("How many EMIs are in a 5-year loan with monthly payments?", 60),
    ("What is the lowest possible CIBIL score?", 300),
]
TEXT = [
    ("What does EMI stand for?", "Equated Monthly Instalment"),
    ("What does KYC stand for?", "Know Your Customer"),
    ("Which regulator supervises NBFCs in India?", "RBI"),
    ("What does NBFC stand for?", "Non-Banking Financial Company"),
    ("What does DTI stand for in lending?", "debt-to-income"),
    ("What does LTV stand for in secured lending?", "loan-to-value"),
    ("What is the Indian unique identity number issued by UIDAI called?", "Aadhaar"),
    ("What is the 10-character Indian tax identifier called?", "PAN"),
]
VERSIONS = [  # label, days ago, accuracy
    ("support-bot-v1.0", 88, 0.74), ("support-bot-v1.1", 66, 0.9), ("support-bot-v1.2", 41, 0.66),
    ("support-bot-v1.3", 19, 0.93), ("support-bot-v1.3", 12, 0.93),
]


def right(rng, expected, numeric, money=False):
    if numeric:
        v = rupees(expected) if expected >= 1000 else str(int(expected))
        forms = [v, f"The answer is {v}.", str(int(expected))]
        if money:
            forms += [f"Rs {v}", f"{v} rupees"]
        return rng.choice(forms)
    return rng.choice([expected, f"It stands for {expected}." if " " in expected else f"{expected}.",
                       expected.lower()])


def wrong(rng, expected, numeric):
    if numeric:
        x = int(expected)
        return rng.choice([rupees(x * 10) if x >= 100 else str(x * 10), str(int(str(x)[::-1]) if len(str(x)) > 1 else x + 1),
                           str(round(x * 1.18)), "I'm not sure, please check with the branch."])
    return rng.choice(["I'm not sure.", "It depends on the lender.", "SEBI" if expected == "RBI" else "Not applicable"])


def main():
    a, rng = args("data/echocheck.db", "Seed EchoCheck with synthetic eval history")
    Path(a.db).parent.mkdir(parents=True, exist_ok=True)
    mem = Memory(a.db)
    if a.if_empty and already_seeded(mem):
        print(f"{a.db} already has data -- skipping seed")
        return
    svc = EchoCheckService(memory=mem)
    cases = [{"id": f"n{i+1:02d}", "question": q, "expected_answer": str(int(v))} for i, (q, v) in enumerate(NUMERIC)]
    cases += [{"id": f"t{i+1:02d}", "question": q, "expected_answer": v} for i, (q, v) in enumerate(TEXT)]
    svc.save_suite("lending-ops", cases)
    numeric_ids = {c["id"] for c in cases if c["id"].startswith("n")}

    runs = []
    for label, days_ago, acc in VERSIONS:
        responses = {}
        for c in cases:
            num = c["id"] in numeric_ids
            exp = float(c["expected_answer"]) if num else c["expected_answer"]
            money = num and "rupees" in c["question"]
            responses[c["id"]] = [right(rng, exp, num, money) if rng.random() < acc else wrong(rng, exp, num)
                                  for _ in range(3)]
        if label == "support-bot-v1.0":   # spelling variants a human later accepts
            responses["t01"][0] = "Equated Monthly Installment"
            responses["t04"][1] = "Non Banking Finance Company"
        r = svc.run("lending-ops", label, runs=3, responses=responses)
        mem.backdate(r["run_id"], iso(days_ago))
        runs.append((label, r))
        if label == "support-bot-v1.1":
            svc.promote("lending-ops", r["run_id"])
        if label == "support-bot-v1.0":
            svc.adjudicate("lending-ops", "t01", "Equated Monthly Installment", True, reviewer="qa-lead")
            svc.adjudicate("lending-ops", "t04", "Non Banking Finance Company", True, reviewer="qa-lead")

    # the bundled sample suite gets two runs of the offline stand-in, too
    for d in (30, 5):
        r = svc.run("sample", "offline-standin", runs=3)
        mem.backdate(r["run_id"], iso(d))

    counts = {"suites": len(svc.suites()), "runs": len(runs) + 2, "adjudications": 2}
    mark(mem, "echocheck", a.seed, counts)
    print(f"seeded {a.db}: {counts}")
    for label, r in runs:
        print(f"  {label:<18} acc={r['accuracy']:.1%} cons={r['consistency']:.1%} gate={r['regression']['verdict']}")


if __name__ == "__main__":
    main()
