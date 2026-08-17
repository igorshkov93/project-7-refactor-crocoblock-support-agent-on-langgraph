"""The guarded entry point to a chat model: shape handling and error mapping.

No subclass of ``BaseChatModel`` is needed here. ``invoke_text`` and
``invoke_structured`` touch exactly two methods — ``invoke`` and
``with_structured_output`` — so a small stub covers the contract without
dragging in a provider SDK.

Retryable paths are deliberately absent: both functions carry ``@with_retry()``
with production waits, so a throttling test would really sleep. That behaviour
is pinned in ``test_retry.py`` instead; here we cover the mapping itself and
the failures that surface immediately.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from src.exceptions import LLMError, LLMRateLimitError, LLMResponseError, LLMTimeoutError
from src.llm_call import _translate, invoke_structured, invoke_text


class Answer(BaseModel):
    """Minimal schema standing in for a real agent's structured output."""

    verdict: str
    score: float = Field(ge=0.0, le=1.0)


class StubResponse:
    """What a chat model hands back: only ``content`` is ever read."""

    def __init__(self, content):
        self.content = content


class StubModel:
    """A chat model that returns a prepared answer instead of calling anyone."""

    def __init__(self, content=None, structured_result=None, raises=None):
        self._content = content
        self._structured_result = structured_result
        self._raises = raises
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self._raises is not None:
            raise self._raises
        return StubResponse(self._content)

    def with_structured_output(self, schema):
        self.schema = schema
        return _StructuredView(self)


class _StructuredView:
    """What ``with_structured_output`` returns: an invocable bound to a schema."""

    def __init__(self, parent):
        self._parent = parent

    def invoke(self, messages):
        self._parent.calls.append(messages)
        if self._parent._raises is not None:
            raise self._parent._raises
        return self._parent._structured_result


MESSAGES = [{"role": "user", "content": "does the form save records?"}]


# --- plain text ------------------------------------------------------------


def test_string_content_is_returned_as_is():
    """The common case: one provider, one string."""
    model = StubModel(content="Yes, enable Save Form Record.")
    assert invoke_text(model, MESSAGES) == "Yes, enable Save Form Record."
    assert model.calls == [MESSAGES]


def test_content_blocks_are_joined():
    """Some providers answer with a list of blocks rather than a string."""
    model = StubModel(content=[{"text": "Enable "}, {"text": "Save Form Record."}])
    assert invoke_text(model, MESSAGES) == "Enable Save Form Record."


def test_non_text_blocks_are_skipped():
    """Tool-use and image blocks carry no ``text`` key and must not break parsing."""
    model = StubModel(
        content=[{"type": "tool_use", "id": "x"}, {"text": "Answer."}]
    )
    assert invoke_text(model, MESSAGES) == "Answer."


@pytest.mark.parametrize("content", ["", "   \n  ", [], None, 42])
def test_empty_answers_are_rejected(content):
    """An empty answer is a failure, not a valid response to pass downstream."""
    model = StubModel(content=content)
    with pytest.raises(LLMResponseError, match="empty answer"):
        invoke_text(model, MESSAGES)


# --- structured output -----------------------------------------------------


def test_valid_structured_output_is_returned():
    """A well-formed answer comes back as the schema instance itself."""
    expected = Answer(verdict="how_to", score=0.9)
    model = StubModel(structured_result=expected)
    result = invoke_structured(model, MESSAGES, Answer)
    assert result is expected


def test_wrong_type_from_structured_output_is_rejected():
    """Some providers hand back a raw dict when parsing fails silently."""
    model = StubModel(structured_result={"verdict": "how_to", "score": 0.9})
    with pytest.raises(LLMResponseError, match="Expected Answer"):
        invoke_structured(model, MESSAGES, Answer)


def test_none_from_structured_output_is_rejected():
    """A model that refuses to answer returns nothing at all."""
    model = StubModel(structured_result=None)
    with pytest.raises(LLMResponseError, match="Expected Answer"):
        invoke_structured(model, MESSAGES, Answer)


# --- error mapping ---------------------------------------------------------


@pytest.mark.parametrize(
    ("provider_message", "expected"),
    [
        ("Request timed out after 60s", LLMTimeoutError),
        ("Deadline exceeded", LLMTimeoutError),
        ("429 Too Many Requests", LLMRateLimitError),
        ("Rate limit reached for this model", LLMRateLimitError),
        ("RESOURCE_EXHAUSTED: quota exceeded", LLMRateLimitError),
        ("Overloaded", LLMRateLimitError),
        ("Internal server error", LLMError),
        ("Something nobody predicted", LLMError),
    ],
)
def test_provider_errors_map_onto_the_domain_hierarchy(provider_message, expected):
    """Both SDKs raise their own types; matching on text is the portable option."""
    mapped = _translate(RuntimeError(provider_message))
    assert type(mapped) is expected


def test_timeout_wins_over_rate_limit_when_both_words_appear():
    """Ordering inside ``_translate`` is a decision, not an accident."""
    mapped = _translate(RuntimeError("request timed out while rate limit applied"))
    assert isinstance(mapped, LLMTimeoutError)


def test_mapping_is_case_insensitive():
    """Providers are inconsistent about capitalisation."""
    assert isinstance(_translate(RuntimeError("QUOTA EXCEEDED")), LLMRateLimitError)
