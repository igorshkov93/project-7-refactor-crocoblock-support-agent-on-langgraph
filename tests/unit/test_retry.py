"""Retry policy: what is repeated, what fails immediately, and how often.

The decision is never taken at the call site — it comes from the ``retryable``
class flag on the exception hierarchy. These tests pin that contract down,
including the sub-classes that deliberately flip the inherited flag.

Waits are overridden to milliseconds. The production defaults (1s initial,
10s ceiling) would make the suite take minutes for no added coverage.
"""
from __future__ import annotations

import pytest

from src.exceptions import (
    ConfigurationError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    SupportAgentError,
    WordPressAuthError,
    WordPressError,
)
from src.retry import with_retry

FAST = {"initial_wait": 0.001, "max_wait": 0.002}


def test_transient_failure_is_retried_until_it_succeeds():
    """A call that recovers on the third attempt returns normally."""
    calls = []

    @with_retry(attempts=3, **FAST)
    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise LLMTimeoutError("provider is slow")
        return "answer"

    assert flaky() == "answer"
    assert len(calls) == 3


def test_attempts_are_capped_and_the_domain_error_survives():
    """Once attempts run out the original exception is re-raised, not RetryError."""
    calls = []

    @with_retry(attempts=3, **FAST)
    def always_failing():
        calls.append(1)
        raise LLMRateLimitError("429 from the provider", status_code=429)

    with pytest.raises(LLMRateLimitError) as caught:
        always_failing()

    assert len(calls) == 3
    assert caught.value.status_code == 429


def test_deterministic_failure_is_not_retried():
    """A configuration error will fail identically on every attempt."""
    calls = []

    @with_retry(attempts=3, **FAST)
    def misconfigured():
        calls.append(1)
        raise ConfigurationError("PINECONE_API_KEY is not set")

    with pytest.raises(ConfigurationError):
        misconfigured()

    assert len(calls) == 1


def test_unparseable_model_output_is_not_retried():
    """LLMResponseError inherits from a retryable class but opts out.

    Re-sending the same prompt after a schema violation just burns quota.
    """
    calls = []

    @with_retry(attempts=3, **FAST)
    def bad_schema():
        calls.append(1)
        raise LLMResponseError("router returned an unknown query_type")

    with pytest.raises(LLMResponseError):
        bad_schema()

    assert len(calls) == 1


def test_rejected_credentials_are_not_retried():
    """WordPressAuthError flips the flag inherited from WordPressError."""
    calls = []

    @with_retry(attempts=3, **FAST)
    def unauthorised():
        calls.append(1)
        raise WordPressAuthError("application password rejected", status_code=401)

    with pytest.raises(WordPressAuthError):
        unauthorised()

    assert len(calls) == 1


def test_foreign_exceptions_are_left_alone():
    """Only our own hierarchy is retried; a library bug must surface at once."""
    calls = []

    @with_retry(attempts=3, **FAST)
    def library_bug():
        calls.append(1)
        raise ValueError("this is not a SupportAgentError")

    with pytest.raises(ValueError):
        library_bug()

    assert len(calls) == 1


def test_successful_call_is_not_delayed():
    """The happy path must not pay for the retry machinery."""
    calls = []

    @with_retry(attempts=3, **FAST)
    def healthy():
        calls.append(1)
        return "ok"

    assert healthy() == "ok"
    assert len(calls) == 1


async def test_async_functions_are_retried_too():
    """tenacity detects coroutines and awaits the back-off instead of blocking."""
    calls = []

    @with_retry(attempts=3, **FAST)
    async def flaky_async():
        calls.append(1)
        if len(calls) < 2:
            raise WordPressError("sandbox hiccup")
        return "recovered"

    assert await flaky_async() == "recovered"
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("error_type", "retryable"),
    [
        (LLMTimeoutError, True),
        (LLMRateLimitError, True),
        (LLMResponseError, False),
        (WordPressError, True),
        (WordPressAuthError, False),
        (ConfigurationError, False),
    ],
)
def test_retryable_flags_are_stable(error_type, retryable):
    """The flag is the contract every call site depends on."""
    assert issubclass(error_type, SupportAgentError)
    assert error_type.retryable is retryable
