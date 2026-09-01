"""
Run the eval harness on a set of test cases and print the report.

This file runs 7 test cases, covering a mix of clean
factual questions and a couple engineered to be borderline (to make sure
failed_cases and consistency actually show something other than a perfect
1.0 / 1.0 report).

Switching providers (once outside this sandbox):
    export OPENAI_API_KEY=your_key_here
    client = ModelClient(provider="openai", model="gpt-4o-mini")
  or
    export ANTHROPIC_API_KEY=your_key_here
    client = ModelClient(provider="anthropic", model="claude-3-5-haiku-20241022")
No other code changes needed -- run_eval() takes the client as a parameter.
"""

import json

from eval_harness import ModelClient, run_eval

TEST_CASES = [
    {"question": "What is the capital of France?", "expected_answer": "Paris"},
    {"question": "What is 12 * 8?", "expected_answer": "96"},
    {"question": "Who wrote 'Pride and Prejudice'?", "expected_answer": "Jane Austen"},
    {"question": "What is the boiling point of water in Celsius?", "expected_answer": "100"},
    {"question": "What is the largest planet in our solar system?", "expected_answer": "Jupiter"},
    {"question": "What language is primarily used for iOS app development?", "expected_answer": "Swift"},
    # deliberately NOT in the offline knowledge base -- exercises the
    # "model doesn't know" path and should show up as a failed case
    {"question": "What is the airspeed velocity of an unladen swallow?", "expected_answer": "11 meters per second"},
]


def main():
    client = ModelClient(provider="offline")  # swap to "openai"/"anthropic" with a key set
    report = run_eval(TEST_CASES, client=client, runs_per_question=3)

    print("=" * 70)
    print("EVAL HARNESS REPORT")
    print("=" * 70)
    print(f"Provider:            {client.provider}")
    print(f"Questions evaluated: {len(TEST_CASES)}")
    print(f"Total accuracy:      {report.total_accuracy:.1%}  (across all runs)")
    print(f"Average consistency: {report.average_consistency:.1%}  (3-run agreement, per question)")
    print(f"Errors:              {len(report.errors)}")
    print()

    print("-" * 70)
    print("PER-QUESTION BREAKDOWN")
    print("-" * 70)
    for q in report.per_question:
        print(f"Q: {q['question']}")
        print(f"   expected:    {q['expected_answer']}")
        print(f"   answers:     {q['answers']}")
        print(f"   correct?:    {q['correct_flags']}")
        print(f"   accuracy:    {q['question_accuracy']:.1%}")
        print(f"   consistency: {q['consistency']:.1%}")
        print()

    print("-" * 70)
    print(f"FAILED CASES ({len(report.failed_cases)})")
    print("-" * 70)
    if not report.failed_cases:
        print("  (none)")
    for f in report.failed_cases:
        print(f"  run {f['run']}: \"{f['question']}\" -> got \"{f['actual_answer']}\", "
              f"expected \"{f['expected_answer']}\"")

    print()
    print("-" * 70)
    print("FULL REPORT (JSON)")
    print("-" * 70)
    print(json.dumps(report.as_dict(), indent=2))


if __name__ == "__main__":
    main()
