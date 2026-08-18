"""Target functions evaluated against the LangSmith datasets.

Each target runs the narrowest piece of the system the dataset is about: the
router alone, or retrieval alone. Running the whole graph would add agent calls
that the metric does not look at and would exhaust the free-tier quota.

Both providers rate-limit per minute while the retry policy in src/retry.py
backs off in seconds, so a batch run outpaces it and burns quota on retries
that were always going to fail. The pace is set here instead.
"""
import time
from typing import Any

from src.agents.router import classify
from src.rag.retriever import search
from src.settings import settings

# A Cohere trial key allows 10 calls per minute and one retrieval spends two.
RETRIEVAL_PAUSE_SECONDS = 13.0

# Gemini free tier allows 10 requests per minute per model.
ROUTER_PAUSE_SECONDS = 7.0


def route_target(inputs: dict[str, Any]) -> dict[str, Any]:
    """Classify one query and report the category with its confidence."""
    decision = classify(inputs["query"])
    time.sleep(ROUTER_PAUSE_SECONDS)
    return {
        "query_type": decision.query_type,
        "confidence": decision.confidence,
        "reason": decision.reason,
    }


def calibration_target(inputs: dict[str, Any]) -> dict[str, Any]:
    """Classify one query and report whether it falls below the threshold.

    Escalation is derived from the confidence threshold only. The graph also
    sends every ``rest`` query to a human regardless of confidence, but that is
    a routing policy, not a property of the router's calibration, and mixing it
    in would score a correctly classified pre-sales question as a failure.
    """
    decision = classify(inputs["query"])
    time.sleep(ROUTER_PAUSE_SECONDS)
    return {
        "query_type": decision.query_type,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "escalated": decision.confidence < settings.confidence_threshold,
    }


def retrieval_target(inputs: dict[str, Any]) -> dict[str, Any]:
    """Retrieve documentation chunks and report the source URLs in order."""
    results = search(inputs["question"])
    time.sleep(RETRIEVAL_PAUSE_SECONDS)
    return {
        "sources": [result["url"] for result in results],
        "titles": [result["title"] for result in results],
        "top_score": results[0]["rerank_score"] if results else None,
    }
