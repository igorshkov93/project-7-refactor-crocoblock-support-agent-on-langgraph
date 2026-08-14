"""Run a code request through the full graph."""
from src.graph import graph

CONFIG = {"configurable": {"thread_id": "test-code-1"}}


def main():
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "snippet to prefill a field with the current "
                               "user email",
                }
            ]
        },
        config=CONFIG,
    )

    print(f"query_type: {result.get('query_type')} "
          f"({result.get('confidence'):.2f})")
    print(f"handled_by: {result.get('handled_by')}\n")
    print(result.get("final_answer"))


if __name__ == "__main__":
    main()
