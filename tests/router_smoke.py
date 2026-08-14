"""Classify a few queries to check the router works end to end."""
from src.agents.router import classify

QUERIES = [
    "How do I add a repeater field to my form?",
    "my form doesnt send emails anymore after update",
    "how much is the lifetime license",
]


def main():
    for query in QUERIES:
        decision = classify(query)
        print(f"\n{query}")
        print(f"  -> {decision.query_type} ({decision.confidence:.2f})")
        print(f"     {decision.reason}")


if __name__ == "__main__":
    main()
