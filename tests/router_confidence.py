"""Check that the router lowers its confidence on ambiguous queries."""
import json
from pathlib import Path

from src.agents.router import classify
from src.graph import CONFIDENCE_THRESHOLD

TESTSET = Path("tests/router_ambiguous.json")


def main():
    cases = json.loads(TESTSET.read_text(encoding="utf-8"))
    print(f"Threshold: {CONFIDENCE_THRESHOLD}\n")

    escalated = 0
    for case in cases:
        decision = classify(case["query"])
        below = decision.confidence < CONFIDENCE_THRESHOLD
        escalated += below
        mark = "ESCALATE" if below else "route    "
        print(f"  [{mark}] {decision.confidence:.2f}  \"{case['query']}\"")
        print(f"             -> {decision.query_type}: {decision.reason}")

    print(f"\n{escalated}/{len(cases)} escalated to human")


if __name__ == "__main__":
    main()
