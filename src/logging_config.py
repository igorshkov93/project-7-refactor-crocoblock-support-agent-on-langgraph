"""Structured logging setup: one configuration point for the whole application."""

import logging
import sys
from typing import Any

_CONFIGURED = False

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once, at application startup.

    Args:
        level: Minimum severity to emit. DEBUG during development,
            INFO in normal operation.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # Third-party libraries are verbose at INFO; keep their noise out.
    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "openai",
        "anthropic",
        "google_genai",
        "google.genai",
        "grpc",
        "asyncio",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger.

    Args:
        name: Pass ``__name__`` from the calling module.
    """
    return logging.getLogger(name)


class ThreadContext(logging.LoggerAdapter[logging.Logger]):
    """Attach a conversation thread id to every message from this logger."""

    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        thread_id = self.extra.get("thread_id", "-") if self.extra else "-"
        return f"[{thread_id}] {msg}", kwargs


def with_thread(logger: logging.Logger, thread_id: str) -> ThreadContext:
    """Bind a thread id to a logger for correlated output.

    Args:
        logger: The module logger to wrap.
        thread_id: LangGraph conversation thread identifier.
    """
    return ThreadContext(logger, {"thread_id": thread_id[:8]})
