"""Run a bug report through the full graph."""
from src.graph import graph


def main():
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "my form doesnt send emails anymore after update",
                }
            ]
        }
    )

    print(f"query_type:  {result.get('query_type')} "
          f"({result.get('confidence'):.2f})")
    print(f"handled_by:  {result.get('handled_by')}")
    print(f"\n{result.get('final_answer')}")


if __name__ == "__main__":
    main()
