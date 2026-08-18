"""Evaluators for the LangSmith datasets.

All three are deterministic: they compare labels, never call a model. That
keeps repeated runs reproducible and free, which matters on a free tier.
"""
from typing import Any

CONFIDENCE_KEY = "confidence"
TYPE_KEY = "query_type"
SOURCES_KEY = "sources"


def routing_accuracy(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    """Score 1 when the router picked the labelled category."""
    actual = outputs.get(TYPE_KEY)
    expected = reference_outputs.get("expected_type")
    return {
        "key": "routing_accuracy",
        "score": int(actual == expected),
        "comment": f"expected {expected}, got {actual}",
    }


def escalation_correct(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    """Score 1 when the escalation decision matches the label.

    A vague query must fall below the threshold; a specific one must not,
    regardless of the language it is written in.
    """
    escalated = bool(outputs.get("escalated"))
    should = bool(reference_outputs.get("should_escalate"))
    confidence = outputs.get(CONFIDENCE_KEY)
    return {
        "key": "escalation_correct",
        "score": int(escalated == should),
        "comment": (
            f"should_escalate={should}, escalated={escalated}, confidence={confidence}"
        ),
    }


def routed_type_correct(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    """Score the category only for cases that are meant to be routed.

    Cases labelled for escalation have no correct category, so they are
    reported as None and left out of the average.
    """
    expected = reference_outputs.get("expected_type")
    if expected is None:
        return {"key": "routed_type_correct", "score": None, "comment": "escalation case"}
    actual = outputs.get(TYPE_KEY)
    return {
        "key": "routed_type_correct",
        "score": int(actual == expected),
        "comment": f"expected {expected}, got {actual}",
    }


def retrieval_hit(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    """Score 1 when the labelled slug appears in any retrieved source."""
    expected = str(reference_outputs.get("expected_source", ""))
    sources = outputs.get(SOURCES_KEY) or []
    hit = any(expected in str(source) for source in sources)
    return {
        "key": "retrieval_hit",
        "score": int(hit),
        "comment": f"expected '{expected}' in {list(sources)[:5]}",
    }


def retrieval_rank(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    """Reciprocal rank of the labelled slug among the retrieved sources.

    Distinguishes a source that arrived first from one that scraped in last,
    which a plain hit rate hides.
    """
    expected = str(reference_outputs.get("expected_source", ""))
    sources = outputs.get(SOURCES_KEY) or []
    for position, source in enumerate(sources, start=1):
        if expected in str(source):
            return {
                "key": "retrieval_rank",
                "score": 1.0 / position,
                "comment": f"'{expected}' at position {position}",
            }
    return {"key": "retrieval_rank", "score": 0.0, "comment": f"'{expected}' not retrieved"}
