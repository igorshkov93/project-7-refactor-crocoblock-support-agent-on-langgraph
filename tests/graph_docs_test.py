"""Run a how-to question through the full graph."""
from src.graph import graph


def main():
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "How do I make a field required in my form?",
                }
            ]
        }
    )

    print(f"query_type:  {result.get('query_type')} "
          f"({result.get('confidence'):.2f})")
    print(f"handled_by:  {result.get('handled_by')}")
    print(f"needs_human: {result.get('needs_human')}")
    print(f"\n{result.get('final_answer')}")


if __name__ == "__main__":
    main()
