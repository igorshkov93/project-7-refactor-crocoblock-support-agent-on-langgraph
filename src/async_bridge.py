"""Run coroutines from synchronous code without tearing down the event loop.

``asyncio.run`` closes its loop on exit. HTTP clients inside the LLM and MCP
SDKs cache connections bound to the loop that created them, so a second call
lands on a closed loop and fails with ``Event loop is closed`` — surfacing as a
connection error from the provider. This bit the bug investigator on the second
human-in-the-loop round, where the same model object is reused.

A single background loop, started once and kept alive for the life of the
process, keeps those connections valid across rounds.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any

from src.logging_config import get_logger

logger = get_logger(__name__)


_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """Start the background loop on first use and return it."""
    global _loop, _thread
    with _lock:
        if _loop is not None and not _loop.is_closed():
            return _loop

        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever,
            name="async-bridge",
            daemon=True,
        )
        thread.start()
        _loop, _thread = loop, thread
        logger.debug("Background event loop started")
        return loop


def run_sync[T](coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
    """Run a coroutine on the shared background loop and wait for its result.

    Args:
        coro: The coroutine to execute.
        timeout: Seconds to wait before giving up. ``None`` waits indefinitely;
            the underlying SDKs apply their own timeouts.

    Returns:
        Whatever the coroutine returns.

    Raises:
        Exception: Whatever the coroutine raises, re-raised in the caller's
            thread with its original traceback.
    """
    loop = _ensure_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)
