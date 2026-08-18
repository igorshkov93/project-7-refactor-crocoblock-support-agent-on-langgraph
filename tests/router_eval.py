"""Evaluate router accuracy against the labelled test set."""
import json
import time
from collections import defaultdict
from pathlib import Path

from src.agents.router import RoutingDecision, classify
from src.config import MODELS
from src.settings import settings

TESTSET = Path("tests/router_testset.json")
DELAY_SECONDS = 13  # free tier allows 5 requests per minute


def classify_with_retry(query: str, attempts: int = 3) -> RoutingDecision:
    """Classify a query, backing off when the rate limit is hit.

    Raises:
        RuntimeError: If the retry loop exits without a decision.
    """
    for attempt in range(attempts):
        try:
            return classify(query)
        except Exception as error:
            if "RESOURCE_EXHAUSTED" not in str(error) or attempt == attempts - 1:
                raise
            wait = 30 * (attempt + 1)
            print(f"    rate limited, waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"classify_with_retry exhausted {attempts} attempts")


def main():
    cases = json.loads(TESTSET.read_text(encoding="utf-8"))
    print(f"Provider: {settings.llm_provider} / {MODELS[settings.llm_provider]['fast']}")
    correct = 0
    per_class = defaultdict(lambda: {"total": 0, "correct": 0})
    failures = []
    confidences = []

    for case in cases:
        decision = classify_with_retry(case["query"])
        #time.sleep(DELAY_SECONDS)
        expected = case["expected"]
        actual = decision.query_type
        ok = actual == expected

        correct += ok
        per_class[expected]["total"] += 1
        per_class[expected]["correct"] += ok
        confidences.append(decision.confidence)

        if not ok:
            failures.append((case["query"], expected, actual, decision.reason))

    accuracy = correct / len(cases)
    avg_conf = sum(confidences) / len(confidences)

    print(f"Accuracy: {correct}/{len(cases)} = {accuracy:.1%}")
    print(f"Average confidence: {avg_conf:.2f}")
    print(f"Confidence range: {min(confidences):.2f} - {max(confidences):.2f}\n")

    print("Per category:")
    for label, stats in sorted(per_class.items()):
        rate = stats["correct"] / stats["total"]
        print(f"  {label:10s} {stats['correct']}/{stats['total']}  {rate:.0%}")

    if failures:
        print(f"\nMisclassified ({len(failures)}):")
        for query, expected, actual, reason in failures:
            print(f"\n  \"{query}\"")
            print(f"    expected {expected}, got {actual}")
            print(f"    reason: {reason}")


if __name__ == "__main__":
    main()
