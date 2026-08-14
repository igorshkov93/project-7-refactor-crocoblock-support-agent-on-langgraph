"""Check the clarifying-question loop of the Bug Investigator."""
from langgraph.types import Command

from src.graph import graph

CONFIG = {"configurable": {"thread_id": "test-bug-1"}}


def main():
    state = graph.invoke(
        {
            "messages": [
                {"role": "user", "content": "my form is not working"}
            ],
            "query_type": "bug",
            "confidence": 0.9,
        },
        config=CONFIG,
    )

    if "__interrupt__" in state:
        question = state["__interrupt__"][0].value["question"]
        print(f"AGENT ASKS:\n{question}\n")
        print("=" * 70)

        reply = (
            "The form shows a success message, but no record appears. "
            "I only added the Send Email action, nothing else."
        )
        print(f"CUSTOMER REPLIES:\n{reply}\n")
        print("=" * 70)

        state = graph.invoke(Command(resume=reply), config=CONFIG)

    print(f"FINAL ({state.get('clarifying_rounds', 0)} rounds):\n")
    print(state.get("final_answer"))


if __name__ == "__main__":
    main()
