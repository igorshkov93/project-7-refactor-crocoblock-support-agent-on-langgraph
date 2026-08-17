"""Upload the local test sets to LangSmith as versioned datasets.

Running this more than once is safe: an existing dataset is reused and its
examples are replaced, so the file on disk stays the source of truth.
"""
import json
from pathlib import Path
from typing import Any

from src.logging_config import get_logger
from src.observability import configure_tracing
from src.settings import settings

logger = get_logger(__name__)

ROUTING_DATASET = "crocoblock-routing"
CALIBRATION_DATASET = "crocoblock-calibration"
RETRIEVAL_DATASET = "crocoblock-retrieval"


def _load(name: str) -> list[dict[str, Any]]:
    """Read a JSON test set from the tests directory."""
    path = Path(__file__).parent / name
    data: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _sync(client: Any, name: str, description: str, cases: list[dict[str, Any]]) -> None:
    """Create or refresh a dataset so it matches the local cases exactly."""
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
        existing = list(client.list_examples(dataset_id=dataset.id))
        for example in existing:
            client.delete_example(example_id=example.id)
        logger.info("Dataset '%s' exists, replaced %d examples", name, len(existing))
    else:
        dataset = client.create_dataset(dataset_name=name, description=description)
        logger.info("Created dataset '%s'", name)

    client.create_examples(
        dataset_id=dataset.id,
        inputs=[case["inputs"] for case in cases],
        outputs=[case["outputs"] for case in cases],
    )
    logger.info("Uploaded %d examples to '%s'", len(cases), name)


def build_routing_cases() -> list[dict[str, Any]]:
    """Labelled queries with the routing category they belong to."""
    return [
        {"inputs": {"query": row["query"]}, "outputs": {"expected_type": row["expected"]}}
        for row in _load("router_testset.json")
    ]


def build_calibration_cases() -> list[dict[str, Any]]:
    """Queries labelled with whether the router should escalate to a human."""
    return [
        {
            "inputs": {"query": row["query"]},
            "outputs": {
                "should_escalate": row["should_escalate"],
                "expected_type": row["expected_type"],
                "note": row["note"],
            },
        }
        for row in _load("router_calibration.json")
    ]


def build_retrieval_cases() -> list[dict[str, Any]]:
    """Questions labelled with the doc slug fragment that should be retrieved."""
    return [
        {
            "inputs": {"question": row["question"]},
            "outputs": {"expected_source": row["expected_source"]},
        }
        for row in _load("rag_testset.json")
    ]


def main() -> None:
    """Push all three test sets to LangSmith."""
    if not configure_tracing():
        raise SystemExit(
            "LangSmith is disabled. Set LANGSMITH_TRACING=true in .env before uploading."
        )

    from langsmith import Client

    client = Client()
    logger.info("Uploading datasets to project '%s'", settings.langsmith_project)

    _sync(
        client,
        ROUTING_DATASET,
        "Labelled support queries for router category accuracy.",
        build_routing_cases(),
    )
    _sync(
        client,
        CALIBRATION_DATASET,
        "Queries labelled with whether the router should escalate to a human. "
        "Covers the defect where short non-English messages score high confidence.",
        build_calibration_cases(),
    )
    _sync(
        client,
        RETRIEVAL_DATASET,
        "Questions labelled with the documentation slug that should be retrieved.",
        build_retrieval_cases(),
    )

    print("Datasets synced:")
    for name in (ROUTING_DATASET, CALIBRATION_DATASET, RETRIEVAL_DATASET):
        dataset = client.read_dataset(dataset_name=name)
        count = len(list(client.list_examples(dataset_id=dataset.id)))
        print(f"  {name:26s} {count} examples")


if __name__ == "__main__":
    main()
