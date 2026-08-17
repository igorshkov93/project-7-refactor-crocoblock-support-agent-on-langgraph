"""Run evaluations against the LangSmith datasets.

Usage:
    python -m tests.langsmith_experiments routing
    python -m tests.langsmith_experiments calibration
    python -m tests.langsmith_experiments retrieval

Each run is tagged with a prefix so that a before/after comparison is visible
in the LangSmith UI without guessing which experiment was which.
"""
import argparse
from collections.abc import Callable
from typing import Any

from src.logging_config import get_logger
from src.observability import configure_tracing, flush_traces
from src.settings import settings
from tests.langsmith_datasets import (
    CALIBRATION_DATASET,
    RETRIEVAL_DATASET,
    ROUTING_DATASET,
)
from tests.langsmith_evaluators import (
    escalation_correct,
    retrieval_hit,
    retrieval_rank,
    routed_type_correct,
    routing_accuracy,
)
from tests.langsmith_targets import (
    calibration_target,
    retrieval_target,
    route_target,
)

logger = get_logger(__name__)

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "routing": {
        "dataset": ROUTING_DATASET,
        "target": route_target,
        "evaluators": [routing_accuracy],
    },
    "calibration": {
        "dataset": CALIBRATION_DATASET,
        "target": calibration_target,
        "evaluators": [escalation_correct, routed_type_correct],
    },
    "retrieval": {
        "dataset": RETRIEVAL_DATASET,
        "target": retrieval_target,
        "evaluators": [retrieval_hit, retrieval_rank],
    },
}


def run(name: str, label: str, concurrency: int) -> Any:
    """Evaluate one dataset and return the LangSmith results object."""
    config = EXPERIMENTS[name]
    from langsmith import Client

    client = Client()
    prefix = f"{name}-{label}"
    logger.info(
        "Running experiment '%s' on dataset '%s' with provider '%s'",
        prefix,
        config["dataset"],
        settings.llm_provider,
    )

    target: Callable[[dict[str, Any]], dict[str, Any]] = config["target"]
    return client.evaluate(
        target,
        data=config["dataset"],
        evaluators=config["evaluators"],
        experiment_prefix=prefix,
        max_concurrency=concurrency,
        metadata={
            "provider": settings.llm_provider,
            "label": label,
            "confidence_threshold": settings.confidence_threshold,
        },
    )


def main() -> None:
    """Parse arguments and run the requested experiment."""
    parser = argparse.ArgumentParser(description="Run a LangSmith experiment.")
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument(
        "--label",
        default="baseline",
        help="Marks the run in the UI, e.g. 'baseline' or 'tuned-prompt'.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Parallel examples. Keep at 1 on a rate-limited free tier.",
    )
    args = parser.parse_args()

    if not configure_tracing():
        raise SystemExit(
            "LangSmith is disabled. Set LANGSMITH_TRACING=true in .env before running."
        )

    try:
        results = run(args.experiment, args.label, args.concurrency)
        print(f"\nExperiment finished: {args.experiment}-{args.label}")
        print("Open the run in LangSmith to see per-example scores.")
        print(results)
    finally:
        flush_traces()


if __name__ == "__main__":
    main()
