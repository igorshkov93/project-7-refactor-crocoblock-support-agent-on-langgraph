"""The router agent, with the provider replaced by a stub.

``classify`` calls ``get_llm("fast")`` inside the function body rather than at
import time, so patching the name in this module is enough — no provider SDK is
ever constructed and no network call is made.

Two things are worth pinning here. First, the tier: routing runs on the cheap
model, and a silent switch to the expensive one would only show up on the bill.
Second, ``router_node`` must copy the decision into state under the exact keys
``route_after_router`` reads back.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents import router as router_module
from src.agents.router import RoutingDecision, classify, router_node
from src.exceptions import LLMResponseError


class SpyLLM:
    """Records the tier it was asked for and returns a prepared decision."""

    def __init__(self, decision=None, raises=None):
        self._decision = decision
        self._raises = raises
        self.received_messages = None
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.received_messages = messages
        if self._raises is not None:
            raise self._raises
        return self._decision


@pytest.fixture
def stub_llm(monkeypatch):
    """Patch ``get_llm`` in the router and hand the test a factory."""
    requested_tiers = []

    def install(decision=None, raises=None):
        llm = SpyLLM(decision=decision, raises=raises)

        def fake_get_llm(tier="smart", temperature=None):  # noqa: ARG001
            requested_tiers.append(tier)
            return llm

        monkeypatch.setattr(router_module, "get_llm", fake_get_llm)
        llm.requested_tiers = requested_tiers
        return llm

    return install


def test_classify_returns_the_models_decision(stub_llm):
    """The happy path: the validated schema instance passes straight through."""
    decision = RoutingDecision(
        query_type="how_to", confidence=0.85, reason="Asks about a documented setting."
    )
    stub_llm(decision=decision)

    result = classify("How do I make a field required?")

    assert result is decision


def test_routing_runs_on_the_fast_tier(stub_llm):
    """Triage must not silently start paying for the reasoning model."""
    llm = stub_llm(
        decision=RoutingDecision(query_type="bug", confidence=0.8, reason="Broken.")
    )

    classify("the form stopped sending emails")

    assert llm.requested_tiers == ["fast"]


def test_the_prompt_carries_the_system_message_and_the_query(stub_llm):
    """System prompt first, raw customer message second — never reordered."""
    llm = stub_llm(
        decision=RoutingDecision(query_type="code", confidence=0.9, reason="Snippet.")
    )

    classify("give me a PHP hook to prefill a field")

    system, user = llm.received_messages
    assert system["role"] == "system"
    assert "JetFormBuilder" in system["content"]
    assert user == {
        "role": "user",
        "content": "give me a PHP hook to prefill a field",
    }


def test_the_schema_sent_to_the_model_is_the_routing_decision(stub_llm):
    """Structured output is bound to the agent's own schema, not a generic dict."""
    llm = stub_llm(
        decision=RoutingDecision(query_type="rest", confidence=0.9, reason="Pricing.")
    )

    classify("how much does a license cost?")

    assert llm.schema is RoutingDecision


def test_malformed_model_output_surfaces_as_a_response_error(stub_llm):
    """A dict instead of the schema is a failure, not something to route on."""
    stub_llm(decision={"query_type": "how_to", "confidence": 0.9, "reason": "x"})

    with pytest.raises(LLMResponseError, match="Expected RoutingDecision"):
        classify("How do I export form records?")


def test_router_node_writes_the_keys_the_graph_reads(stub_llm):
    """``route_after_router`` looks up ``query_type`` and ``confidence`` by name."""
    stub_llm(
        decision=RoutingDecision(
            query_type="bug",
            confidence=0.82,
            reason="Reports a symptom after an update.",
        )
    )
    state = {"messages": [HumanMessage(content="emails stopped after the update")]}

    update = router_node(state)

    assert update == {
        "query_type": "bug",
        "confidence": 0.82,
        "routing_reason": "Reports a symptom after an update.",
    }


def test_router_node_classifies_the_latest_message(stub_llm):
    """In a multi-turn thread the newest turn is the one being triaged."""
    llm = stub_llm(
        decision=RoutingDecision(query_type="how_to", confidence=0.9, reason="Docs.")
    )
    state = {
        "messages": [
            HumanMessage(content="the form is broken"),
            AIMessage(content="Could you describe what happens?"),
            HumanMessage(content="the submit button does nothing"),
        ]
    }

    router_node(state)

    _, user = llm.received_messages
    assert user["content"] == "the submit button does nothing"


@pytest.mark.parametrize("query_type", ["how_to", "bug", "code", "rest"])
def test_every_category_survives_the_round_trip(stub_llm, query_type):
    """All four types the graph knows how to route must validate in the schema."""
    stub_llm(
        decision=RoutingDecision(query_type=query_type, confidence=0.9, reason="ok")
    )

    assert classify("some message").query_type == query_type


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_confidence_outside_the_unit_interval_is_rejected(confidence):
    """The schema bounds the score; the graph compares it against a threshold."""
    with pytest.raises(ValueError, match="confidence"):
        RoutingDecision(query_type="how_to", confidence=confidence, reason="x")


def test_unknown_category_is_rejected_by_the_schema():
    """``QueryType`` is a Literal: an invented category never reaches the graph."""
    with pytest.raises(ValueError, match="query_type"):
        RoutingDecision(query_type="billing", confidence=0.9, reason="x")
