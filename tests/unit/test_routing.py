"""Routing decisions taken by the graph, tested without invoking any agent.

``route_after_router`` is a pure function of the state: it reads ``confidence``
and ``query_type`` and returns a node name. Testing it directly covers every
branch in microseconds, where a full ``graph.invoke`` would need four live
providers to reach the same assertions.
"""
from __future__ import annotations

import pytest

from src.graph import route_after_router
from src.settings import settings


@pytest.mark.parametrize(
    ("query_type", "expected"),
    [
        ("how_to", "docs_qa"),
        ("bug", "bug_investigator"),
        ("code", "code_generator"),
        ("rest", "escalate"),
    ],
)
def test_confident_query_reaches_its_agent(query_type, expected):
    """Above the threshold, the type alone decides the destination."""
    state = {"query_type": query_type, "confidence": 0.9}
    assert route_after_router(state) == expected


@pytest.mark.parametrize("query_type", ["how_to", "bug", "code", "rest"])
def test_low_confidence_overrides_every_type(query_type):
    """Below the threshold the query type is irrelevant: a human takes over."""
    state = {"query_type": query_type, "confidence": 0.1}
    assert route_after_router(state) == "escalate"


def test_threshold_is_exclusive_at_the_boundary():
    """The graph compares with ``<``, so a value equal to the threshold passes."""
    state = {"query_type": "how_to", "confidence": settings.confidence_threshold}
    assert route_after_router(state) == "docs_qa"


def test_value_just_below_threshold_escalates():
    """One step under the boundary must fall to escalation."""
    state = {"query_type": "how_to", "confidence": settings.confidence_threshold - 0.01}
    assert route_after_router(state) == "escalate"


def test_unknown_query_type_escalates():
    """An unmapped type is a router bug: escalate rather than guess an agent."""
    state = {"query_type": "billing", "confidence": 0.9}
    assert route_after_router(state) == "escalate"


def test_empty_state_escalates():
    """Missing fields default to zero confidence, which escalates."""
    assert route_after_router({}) == "escalate"
