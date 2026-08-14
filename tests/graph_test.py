"""Verify every routing branch of the graph skeleton."""
from src.graph import graph

CASES = [
    ("how_to", 0.9, "docs_qa"),
    ("bug", 0.9, "bug_investigator"),
    ("code", 0.9, "code_generator"),
    ("rest", 0.9, "escalation"),
    ("how_to", 0.3, "escalation"),  # low confidence overrides the type
]


def main():
    passed = 0
    for query_type, confidence, expected in CASES:
        result = graph.invoke(
            {
                "messages": [{"role": "user", "content": "test"}],
                "query_type": query_type,
                "confidence": confidence,
            }
        )
        actual = result.get("handled_by")
        ok = actual == expected
        passed += ok
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {query_type} @ {confidence} -> {actual}")

    print(f"\n{passed}/{len(CASES)} routes correct")


if __name__ == "__main__":
    main()
