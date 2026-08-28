#!/usr/bin/env python3

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent import ReasoningAgent
from tests.test_cases import ALL_CASES


def run() -> int:
    agent = ReasoningAgent()
    failures = []

    for i, (question, expected_substr, expect_success) in enumerate(ALL_CASES, 1):
        result = agent.solve(question)
        verifier_passed = len(result["metadata"]["checks"]) > 0 and all(
            c["passed"] for c in result["metadata"]["checks"]
        )
        retried = result["metadata"]["retries"] > 0

        print(f"\n=== Test {i} ===")
        print(f"Question: {question}")
        print(json.dumps(result, indent=2))
        print(f"Verifier passed: {verifier_passed} | Retried: {retried}")

        ok = True
        if expect_success and result["status"] != "success":
            ok = False
        if not expect_success and result["status"] != "failed":
            ok = False
        if expected_substr is not None and expected_substr.lower() not in result["answer"].lower():
            ok = False

        if not ok:
            failures.append((i, question, result))
            print(">>> FAILED ASSERTION for this case <<<")
        else:
            print(">>> OK <<<")

    print(f"\n{len(ALL_CASES) - len(failures)}/{len(ALL_CASES)} cases passed.")
    if failures:
        print("\nFailed cases:")
        for i, q, r in failures:
            print(f"  #{i}: {q}\n     -> {r['answer']} (status={r['status']})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
