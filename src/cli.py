"""Command-line interface to the support agent."""

import argparse
import logging
import sys

from src.logging_config import get_logger, setup_logging
from src.observability import flush_traces
from src.runner import new_thread_id, resume, start

logger = get_logger(__name__)


def show(state: dict[str, object]) -> None:
    """Print the agent's answer and how it was handled."""
    print(f"\n[{state.get('query_type')}] -> {state.get('handled_by')}\n")

    answer = state.get("final_answer")
    if answer:
        print(answer)
        return

    logger.error("No answer in final state; keys present: %s", list(state))
    print("No answer was produced. Run with --verbose to see what happened.")


def main() -> None:
    """Run one conversation from the command line."""
    parser = argparse.ArgumentParser(
        description="Ask the Crocoblock support agent a question.",
    )
    parser.add_argument("query", nargs="*", help="the question to ask")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show debug-level logs, including routing decisions",
    )
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.WARNING)

    query = " ".join(args.query).strip()
    if not query:
        query = input("Question: ").strip()
    if not query:
        print("No question given.", file=sys.stderr)
        raise SystemExit(1)

    thread_id = new_thread_id()

    try:
        state, question = start(query, thread_id)

        while question:
            print(f"\n? {question}")
            answer = input("> ").strip()
            state, question = resume(answer, thread_id)

        show(state)
    finally:
        flush_traces()


if __name__ == "__main__":
    main()

