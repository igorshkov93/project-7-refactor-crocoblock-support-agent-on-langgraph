"""The documentation agent, with retrieval and the provider replaced.

``search`` and ``get_llm`` are imported into the agent's own namespace, so
patching them here keeps Pinecone and Cohere clients from ever being built.
``invoke_text`` runs for real: the stub answers on the first call, so the retry
decorator never sleeps.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from src.agents import docs_qa as agent
from src.agents.docs_qa import (
    NO_RESULTS,
    RETRIEVAL_FAILED,
    answer,
    docs_qa_node,
    format_context,
)
from src.exceptions import EmbeddingError, RerankError, VectorStoreError

HIT = {
    "text": "Enable Save Form Record in the post-submit actions.",
    "title": "Save Form Record",
    "url": "https://crocoblock.com/knowledge-base/save-form-record/",
    "vector_score": 0.81,
    "rerank_score": 0.94,
}


class StubResponse:
    def __init__(self, content):
        self.content = content


class StubModel:
    def __init__(self, content="Open JetFormBuilder and enable Save Form Record."):
        self._content = content
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return StubResponse(self._content)


@pytest.fixture
def wired(monkeypatch):
    """Patch retrieval and the provider; return handles to inspect afterwards."""

    def install(hits=None, raises=None):
        model = StubModel()
        tiers = []

        def fake_search(query):  # noqa: ARG001
            if raises is not None:
                raise raises
            return list(hits or [])

        def fake_get_llm(tier="smart", temperature=None):  # noqa: ARG001
            tiers.append(tier)
            return model

        monkeypatch.setattr(agent, "search", fake_search)
        monkeypatch.setattr(agent, "get_llm", fake_get_llm)
        model.tiers = tiers
        return model

    return install


def test_context_is_numbered_with_titles_and_urls():
    """The prompt tells the model to cite URLs, so both must reach it."""
    rendered = format_context([HIT, {**HIT, "title": "Second", "url": "https://x/"}])

    assert "[1] Save Form Record" in rendered
    assert "[2] Second" in rendered
    assert HIT["url"] in rendered
    assert "\n\n---\n\n" in rendered


def test_empty_context_renders_as_an_empty_string():
    assert format_context([]) == ""


def test_answer_returns_text_and_sources(wired):
    wired(hits=[HIT])

    result = answer("How do I save form submissions?")

    assert result["found"] is True
    assert result["sources"] == [HIT["url"]]
    assert result["answer"]


def test_answer_runs_on_the_smart_tier(wired):
    """Generation is the expensive step by design; a downgrade would be silent."""
    model = wired(hits=[HIT])

    answer("How do I save form submissions?")

    assert model.tiers == ["smart"]


def test_excerpts_reach_the_model(wired):
    """The agent may only answer from what retrieval found."""
    model = wired(hits=[HIT])

    answer("How do I save form submissions?")

    system, user = model.messages
    assert HIT["text"] in system["content"]
    assert user["content"] == "How do I save form submissions?"


def test_no_matches_escalates_without_calling_the_model(wired):
    """Nothing retrieved means nothing to ground an answer in."""
    model = wired(hits=[])

    result = answer("How do I integrate with Salesforce?")

    assert result == {"answer": NO_RESULTS, "sources": [], "found": False}
    assert model.messages is None


@pytest.mark.parametrize(
    "failure",
    [
        EmbeddingError("cohere is down"),
        VectorStoreError("pinecone refused the query"),
        RerankError("rerank timed out"),
    ],
)
def test_retrieval_failure_escalates_with_its_own_message(wired, failure):
    """A broken index reads differently to a human than an empty one."""
    wired(raises=failure)

    result = answer("How do I save form submissions?")

    assert result["answer"] == RETRIEVAL_FAILED
    assert result["found"] is False
    assert RETRIEVAL_FAILED != NO_RESULTS


def test_node_maps_the_answer_into_state(wired):
    wired(hits=[HIT])
    state = {"messages": [HumanMessage(content="How do I save submissions?")]}

    update = docs_qa_node(state)

    assert update["handled_by"] == "docs_qa"
    assert update["needs_human"] is False
    assert update["retrieved_docs"] == [HIT["url"]]
    assert update["final_answer"]


def test_node_flags_a_human_when_nothing_was_found(wired):
    """``needs_human`` is the inverse of ``found`` — the graph reads only this."""
    wired(hits=[])
    state = {"messages": [HumanMessage(content="Salesforce integration?")]}

    assert docs_qa_node(state)["needs_human"] is True
