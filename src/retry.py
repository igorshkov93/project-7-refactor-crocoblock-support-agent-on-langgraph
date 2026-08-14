"""Retry policies for calls to external services.

A single decorator is used everywhere, and the decision to retry is never made
at the call site: it is taken from the ``retryable`` flag on the exception
hierarchy in :mod:`src.exceptions`. Timeouts, rate limits and network blips are
repeated with exponential back-off and jitter; bad credentials, malformed
payloads and configuration errors fail immediately.

The decorator works on both sync and async functions — tenacity detects
coroutine functions and awaits the sleep instead of blocking.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import ParamSpec, TypeVar, cast

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.exceptions import SupportAgentError
from src.settings import settings

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def _is_retryable(exc: BaseException) -> bool:
    """Retry only our own exceptions that are explicitly marked transient."""
    return isinstance(exc, SupportAgentError) and exc.retryable


def _log_retry(state: RetryCallState) -> None:
    """Emit one structured warning per retry, so LangSmith traces have context."""
    exc = state.outcome.exception() if state.outcome is not None else None
    sleep_for = state.next_action.sleep if state.next_action is not None else 0.0
    func_name = getattr(state.fn, "__qualname__", "<unknown>")
    logger.warning(
        "Retrying %s after %s (attempt %d, sleeping %.1fs)",
        func_name,
        type(exc).__name__ if exc is not None else "unknown error",
        state.attempt_number,
        sleep_for,
        extra={
            "retry_func": func_name,
            "retry_attempt": state.attempt_number,
            "retry_sleep": round(sleep_for, 2),
            "error_type": type(exc).__name__ if exc is not None else None,
            "error_detail": str(exc) if exc is not None else None,
        },
    )


def with_retry(
    *,
    attempts: int | None = None,
    initial_wait: float | None = None,
    max_wait: float | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return a decorator retrying transient :class:`SupportAgentError` failures.

    Args:
        attempts: Total number of calls including the first one. Defaults to
            ``RETRY_ATTEMPTS`` from the settings.
        initial_wait: Back-off before the second attempt, in seconds. Defaults
            to ``RETRY_INITIAL_WAIT``.
        max_wait: Upper bound for a single back-off interval, in seconds.
            Defaults to ``RETRY_MAX_WAIT``.
            
    Returns:
        A decorator preserving the signature of the wrapped function.

    The original exception is re-raised once the attempts are exhausted
    (``reraise=True``), so callers keep seeing the domain exception rather than
    a tenacity ``RetryError``.
    """
    attempts = settings.retry_attempts if attempts is None else attempts
    initial_wait = settings.retry_initial_wait if initial_wait is None else initial_wait
    max_wait = settings.retry_max_wait if max_wait is None else max_wait

    decorator = retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(attempts),
        wait=wait_exponential_jitter(initial=initial_wait, max=max_wait),
        before_sleep=_log_retry,
        reraise=True,
    )
    # tenacity types its decorator with a TypeVar bound to Callable[..., Any],
    # which loses the parameter spec; the cast restores the wrapped signature.
    return cast("Callable[[Callable[P, R]], Callable[P, R]]", decorator)
