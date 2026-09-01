#!/usr/bin/env python3
"""
CLI entry point for the eval harness.

Usage:
    python3 cli.py --input sample_data/questions.json
    python3 cli.py --input my_questions.json --provider openai --runs 5
    python3 cli.py --input my_questions.json --output report.json
"""

from __future__ import annotations

import argparse
import json
import sys

from eval_harness import ModelClient, run_eval


def main():
    parser = argparse.ArgumentParser(description="Run the LLM eval harness on a set of Q&A pairs.")
    parser.add_argument("--input", required=True, help="Path to a JSON file: a list of "
                         "{'question': ..., 'expected_answer': ...} objects.")
    parser.add_argument("--provider", default="offline", choices=["offline", "openai", "anthropic"],
                         help="Model provider. Defaults to 'offline' (no API key needed). "
                              "'openai'/'anthropic' require the matching *_API_KEY env var.")
    parser.add_argument("--model", default=None, help="Model name override (provider-specific default if omitted).")
    parser.add_argument("--runs", type=int, default=3, help="Runs per question for consistency scoring (default 3).")
    parser.add_argument("--output", default=None, help="Optional path to write the full JSON report to.")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    if not isinstance(qa_pairs, list) or not qa_pairs:
        print(f"error: {args.input} must contain a non-empty JSON list of question/expected_answer objects", file=sys.stderr)
        sys.exit(1)

    client_kwargs = {"provider": args.provider}
    if args.model:
        client_kwargs["model"] = args.model
    client = ModelClient(**client_kwargs)

    report = run_eval(qa_pairs, client=client, runs_per_question=args.runs)

    print(f"Provider: {client.provider}   Questions: {len(qa_pairs)}   Runs/question: {args.runs}")
    print(f"Total accuracy:      {report.total_accuracy:.1%}")
    print(f"Average consistency: {report.average_consistency:.1%}")
    print(f"Failed cases:        {len(report.failed_cases)}")
    print(f"Errors:              {len(report.errors)}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report.as_dict(), f, indent=2)
        print(f"Full report written to {args.output}")


if __name__ == "__main__":
    main()
