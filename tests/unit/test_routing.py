"""Routing decisions taken by the graph, tested without invoking any agent.

``route_after_router`` is a pure function of the state: it reads ``confidence``
and ``query_type`` and returns a node name. Testing it directly covers every
branch in microseconds, where a full ``graph.invoke`` would need four live
providers to reach the same assertions.
"""
from __future__ import annotations

import pytest
from langgraph.graph import END

from src.graph import escalate_node, route_after_investigation, route_after_router
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

# --- the investigation loop ------------------------------------------------


def test_investigation_loops_back_while_the_answer_is_missing():
    """No final answer means the investigator needs another round."""
    assert route_after_investigation({}) == "bug_investigator"


def test_investigation_ends_once_an_answer_exists():
    """A filled answer terminates the loop instead of asking again."""
    state = {"final_answer": "Enable Save Form Record in the post-submit actions."}
    assert route_after_investigation(state) == END


@pytest.mark.parametrize("answer", ["", None])
def test_falsy_answers_do_not_end_the_investigation(answer):
    """An empty string is not an answer: the check is truthiness, not presence."""
    assert route_after_investigation({"final_answer": answer}) == "bug_investigator"


def test_clarifying_rounds_do_not_by_themselves_stop_the_loop():
    """The round counter is enforced inside the agent, not by the edge."""
    state = {"clarifying_rounds": 99}
    assert route_after_investigation(state) == "bug_investigator"


# --- escalation ------------------------------------------------------------


def test_escalation_marks_the_ticket_for_a_human():
    """The three fields the interface reads after a graph run."""
    update = escalate_node({"query_type": "rest", "confidence": 0.9})

    assert update["needs_human"] is True
    assert update["handled_by"] == "escalation"
    assert update["final_answer"]


def test_escalation_works_on_an_empty_state():
    """Escalation is the fallback path, so it must never depend on prior fields."""
    update = escalate_node({})

    assert update["needs_human"] is True
