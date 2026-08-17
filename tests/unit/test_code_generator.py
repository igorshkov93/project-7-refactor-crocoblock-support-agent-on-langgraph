"""The snippet agent: hook reference loading and environment-aware prompting.

The curated hook reference is the guard against invented signatures, so its
three failure modes — missing, unreadable, empty — are configuration faults
rather than runtime hiccups, and each is pinned below. ``KNOWLEDGE`` is a
module-level ``Path``, patched per test with ``monkeypatch.setattr``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from src.agents import code_generator as agent
from src.agents.code_generator import (
    code_generator_node,
    generate,
    load_hook_reference,
)
from src.exceptions import ConfigurationError

ENV = {"wp_version": "6.7.1", "php_version": "8.2", "active_theme": "Kadence"}


class StubResponse:
    def __init__(self, content):
        self.content = content


class StubModel:
    def __init__(self, content="add_action('jet-form-builder/custom-action', ...);"):
        self._content = content
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return StubResponse(self._content)


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Point the agent at a temporary hook reference and a stubbed provider."""

    def install(hooks="## jet-form-builder/custom-action\nFires after submit."):
        reference = tmp_path / "jfb_hooks.md"
        reference.write_text(hooks, encoding="utf-8")
        monkeypatch.setattr(agent, "KNOWLEDGE", reference)

        model = StubModel()
        tiers = []

        def fake_get_llm(tier="smart", temperature=None):  # noqa: ARG001
            tiers.append(tier)
            return model

        monkeypatch.setattr(agent, "get_llm", fake_get_llm)
        model.tiers = tiers
        return model

    return install


# --- the hook reference ----------------------------------------------------


def test_reference_ships_with_the_repository():
    """Not a fixture: the real file must exist, or the agent invents hooks."""
    assert load_hook_reference().strip()


def test_missing_reference_is_a_configuration_fault(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "KNOWLEDGE", tmp_path / "absent.md")

    with pytest.raises(ConfigurationError, match="not found"):
        load_hook_reference()


def test_empty_reference_is_rejected(monkeypatch, tmp_path):
    """An empty file would silently produce a prompt with no verified hooks."""
    reference = tmp_path / "jfb_hooks.md"
    reference.write_text("   \n\n  ", encoding="utf-8")
    monkeypatch.setattr(agent, "KNOWLEDGE", reference)

    with pytest.raises(ConfigurationError, match="is empty"):
        load_hook_reference()


def test_unreadable_reference_is_a_configuration_fault(monkeypatch, tmp_path):
    """A directory in place of the file raises OSError, not FileNotFoundError."""
    monkeypatch.setattr(agent, "KNOWLEDGE", Path(tmp_path))

    with pytest.raises(ConfigurationError):
        load_hook_reference()


# --- generation ------------------------------------------------------------


def test_the_reference_reaches_the_prompt(wired):
    """Only hooks from the reference may be treated as verified."""
    model = wired(hooks="## jet-form-builder/unique-marker\nDetails.")

    generate("validate the email domain before submitting")

    assert "jet-form-builder/unique-marker" in model.messages[0]["content"]


def test_generation_runs_on_the_smart_tier(wired):
    model = wired()

    generate("add a custom option to the Select field")

    assert model.tiers == ["smart"]


def test_the_request_is_the_last_message(wired):
    """Ordering matters: system context first, the customer's words last."""
    model = wired()

    generate("prefill a field with the current user email")

    assert model.messages[-1] == {
        "role": "user",
        "content": "prefill a field with the current user email",
    }


def test_environment_is_added_as_extra_context(wired):
    """What the bug investigator learned about the site shapes the snippet."""
    model = wired()

    generate("make the field required", env_info=ENV)

    combined = " ".join(m["content"] for m in model.messages if m["role"] == "system")
    assert "6.7.1" in combined
    assert "8.2" in combined


def test_without_environment_the_prompt_stays_version_agnostic(wired):
    """Two system messages with env, one without."""
    model = wired()

    generate("make the field required")

    system_messages = [m for m in model.messages if m["role"] == "system"]
    assert len(system_messages) == 1


@pytest.mark.parametrize("env_info", [None, {}])
def test_falsy_environment_is_treated_as_absent(wired, env_info):
    """An empty dict must not produce a message full of ``None`` versions."""
    model = wired()

    generate("make the field required", env_info=env_info)

    assert len([m for m in model.messages if m["role"] == "system"]) == 1


def test_generated_snippet_is_returned(wired):
    model = wired()

    assert generate("add a hook") == model._content


# --- the graph node --------------------------------------------------------


def test_node_writes_the_snippet_into_state(wired):
    wired()
    state = {"messages": [HumanMessage(content="snippet to prefill a field")]}

    update = code_generator_node(state)

    assert update["handled_by"] == "code_generator"
    assert update["final_answer"]


def test_node_forwards_the_environment_from_state(wired):
    """``env_info`` is written by the bug investigator earlier in the graph."""
    model = wired()
    state = {
        "messages": [HumanMessage(content="snippet to prefill a field")],
        "env_info": ENV,
    }

    code_generator_node(state)

    combined = " ".join(m["content"] for m in model.messages if m["role"] == "system")
    assert "6.7.1" in combined
