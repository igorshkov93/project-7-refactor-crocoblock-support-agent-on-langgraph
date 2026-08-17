"""The diagnostic agent: rounds, escalation paths and the human-in-the-loop.

Four names are patched: ``load_tools`` and ``create_react_agent`` keep the MCP
subprocess from launching, ``get_llm`` keeps the provider out, and ``interrupt``
would otherwise suspend the graph and hang the test.

The round counter is the subtle part. On the diagnostics-failure path the agent
returns ``clarifying_rounds`` unchanged, and the loop still terminates — the
edge checks ``final_answer``, not the counter. Both halves of that contract are
pinned here and in ``test_routing.py``.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents import bug_investigator as agent
from src.agents.bug_investigator import (
    DIAGNOSTICS_UNAVAILABLE,
    Findings,
    bug_investigator_node,
    extract_text,
)
from src.exceptions import InvestigationError, MCPServerError

REPORT = "my form submits but nothing appears in Form Records"


class Interrupted(Exception):  # noqa: N818 — a suspension, not an error
    """Stands in for LangGraph suspending the graph at ``interrupt``."""

    def __init__(self, payload):
        super().__init__("interrupted")
        self.payload = payload


@pytest.fixture
def wired(monkeypatch):
    """Replace diagnostics, classification and the interrupt mechanism."""

    def install(diagnosis="Checked the site: Save Form Record is disabled.",
                verdict=None, diagnostics_raises=None, resume=None):
        calls = {"conversations": [], "interrupts": []}

        def fake_run_diagnostics(messages):
            calls["conversations"].append(messages)
            if diagnostics_raises is not None:
                raise diagnostics_raises
            return diagnosis

        def fake_classify(text):
            calls.setdefault("classified", []).append(text)
            return verdict or Findings(needs_input=False, message=text)

        def fake_interrupt(payload):
            calls["interrupts"].append(payload)
            if resume is None:
                raise Interrupted(payload)
            return resume

        monkeypatch.setattr(agent, "run_diagnostics", fake_run_diagnostics)
        monkeypatch.setattr(agent, "classify_response", fake_classify)
        monkeypatch.setattr(agent, "interrupt", fake_interrupt)
        return calls

    return install


# --- response parsing ------------------------------------------------------


def test_plain_string_content_passes_through():
    assert extract_text("Save Form Record is disabled.") == "Save Form Record is disabled."


def test_thinking_blocks_are_dropped():
    """Extended thinking must never reach the customer."""
    content = [
        {"type": "thinking", "thinking": "internal reasoning the user must not see"},
        {"type": "text", "text": "Save Form Record is disabled."},
    ]

    assert extract_text(content) == "Save Form Record is disabled."


def test_multiple_text_blocks_are_joined():
    content = [
        {"type": "text", "text": "I checked the plugins."},
        {"type": "text", "text": "Nothing conflicts."},
    ]

    assert extract_text(content) == "I checked the plugins.\nNothing conflicts."


@pytest.mark.parametrize(
    "content",
    [[], [{"type": "thinking", "thinking": "only reasoning"}], [{"type": "text", "text": ""}]],
)
def test_content_without_usable_text_yields_an_empty_string(content):
    """The caller turns this into InvestigationError rather than an empty answer."""
    assert extract_text(content) == ""


# --- the diagnosis path ----------------------------------------------------


def test_a_confident_diagnosis_ends_the_investigation(wired):
    wired(verdict=Findings(needs_input=False, message="Save Form Record is off."))

    update = bug_investigator_node({"messages": [HumanMessage(content=REPORT)]})

    assert update["final_answer"] == "Save Form Record is off."
    assert update["needs_human"] is False
    assert update["handled_by"] == "bug_investigator"


def test_the_report_reaches_the_diagnostic_agent(wired):
    calls = wired()

    bug_investigator_node({"messages": [HumanMessage(content=REPORT)]})

    assert calls["conversations"][0] == [{"role": "user", "content": REPORT}]


def test_the_first_message_is_used_not_the_last(wired):
    """Later turns are replies to the agent; the original report frames the case."""
    calls = wired()
    state = {
        "messages": [
            HumanMessage(content=REPORT),
            AIMessage(content="Which actions are configured?"),
            HumanMessage(content="only Send Email"),
        ]
    }

    bug_investigator_node(state)

    assert calls["conversations"][0][0]["content"] == REPORT

# --- the clarifying loop ---------------------------------------------------


def test_an_open_question_suspends_the_graph(wired):
    """Below the round limit the agent asks instead of guessing."""
    calls = wired(verdict=Findings(needs_input=True, message="What do you see?"))

    with pytest.raises(Interrupted):
        bug_investigator_node({"messages": [HumanMessage(content=REPORT)]})

    assert calls["interrupts"][0] == {"question": "What do you see?", "round": 1}


def test_the_reply_is_recorded_and_the_round_counted(wired):
    """After resuming, the exchange joins the log the next round replays."""
    wired(
        verdict=Findings(needs_input=True, message="Which actions are configured?"),
        resume="only Send Email",
    )

    update = bug_investigator_node({"messages": [HumanMessage(content=REPORT)]})

    assert update["clarifying_rounds"] == 1
    assert update["investigation_log"] == [
        {"question": "Which actions are configured?", "reply": "only Send Email"}
    ]
    assert "final_answer" not in update


def test_history_is_replayed_as_alternating_turns(wired):
    """The ReAct agent is stateless: prior rounds must be resent every time."""
    calls = wired()
    state = {
        "messages": [HumanMessage(content=REPORT)],
        "clarifying_rounds": 1,
        "investigation_log": [{"question": "Which actions?", "reply": "Send Email"}],
    }

    bug_investigator_node(state)

    assert calls["conversations"][0] == [
        {"role": "user", "content": REPORT},
        {"role": "assistant", "content": "Which actions?"},
        {"role": "user", "content": "Send Email"},
    ]


def test_the_round_limit_hands_over_to_a_human(wired):
    """At the limit an open question stops being asked and becomes escalation."""
    calls = wired(verdict=Findings(needs_input=True, message="Still unclear."))
    state = {
        "messages": [HumanMessage(content=REPORT)],
        "clarifying_rounds": agent.settings.max_clarifying_rounds,
    }

    update = bug_investigator_node(state)

    assert update["needs_human"] is True
    assert update["final_answer"] == "Still unclear."
    assert calls["interrupts"] == []


def test_the_last_allowed_round_still_asks(wired):
    """One below the limit is inside it: the comparison is ``>=``."""
    calls = wired(
        verdict=Findings(needs_input=True, message="One more thing?"),
        resume="here it is",
    )
    state = {
        "messages": [HumanMessage(content=REPORT)],
        "clarifying_rounds": agent.settings.max_clarifying_rounds - 1,
    }

    bug_investigator_node(state)

    assert len(calls["interrupts"]) == 1


# --- diagnostics failure ---------------------------------------------------


@pytest.mark.parametrize(
    "failure",
    [
        MCPServerError("the MCP subprocess died on handshake"),
        InvestigationError("the diagnostic agent returned no readable message"),
    ],
)
def test_unreachable_diagnostics_escalate(wired, failure):
    """A site we cannot inspect is a human's job, not a guess."""
    wired(diagnostics_raises=failure)

    update = bug_investigator_node({"messages": [HumanMessage(content=REPORT)]})

    assert update["final_answer"] == DIAGNOSTICS_UNAVAILABLE
    assert update["needs_human"] is True


def test_failure_leaves_the_round_counter_untouched(wired):
    """The counter is not what ends the loop — the answer is."""
    wired(diagnostics_raises=MCPServerError("down"))
    state = {"messages": [HumanMessage(content=REPORT)], "clarifying_rounds": 1}

    update = bug_investigator_node(state)

    assert update["clarifying_rounds"] == 1


def test_failure_never_reaches_the_classifier(wired):
    """No message means nothing to classify; the provider must not be called."""
    calls = wired(diagnostics_raises=MCPServerError("down"))

    bug_investigator_node({"messages": [HumanMessage(content=REPORT)]})

    assert "classified" not in calls
