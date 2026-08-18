"""LangSmith tracing setup.

The LangSmith SDK reads its configuration from the process environment, while
this project keeps configuration in a validated Settings object. This module is
the one place where the two meet.
"""
import os

from src.logging_config import get_logger
from src.settings import settings

logger = get_logger(__name__)

_configured = False


def configure_tracing() -> bool:
    """Export LangSmith settings to the environment. Returns True when enabled.

    Safe to call more than once: the environment is only written on the first
    call. Entry points may each call it without coordinating.
    """
    global _configured

    if _configured:
        return settings.langsmith_tracing

    if not settings.langsmith_tracing:
        logger.info("LangSmith tracing is disabled")
        _configured = True
        return False

    # Guarded by the check_tracing_credentials validator in settings.
    assert settings.langsmith_api_key is not None

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

    logger.info("LangSmith tracing enabled, project '%s'", settings.langsmith_project)
    _configured = True
    return True

def flush_traces() -> None:
    """Wait for buffered traces to reach LangSmith before the process exits.

    The SDK sends traces from a background thread in batches. A short-lived
    process can exit before the final batch leaves the queue, which silently
    drops the traces for that run.
    """
    if not settings.langsmith_tracing:
        return

    try:
        from langsmith import Client

        Client().flush()
        logger.debug("Flushed pending traces to LangSmith")
    except Exception:
        # Observability must never break the run it observes.
        logger.warning("Could not flush traces to LangSmith", exc_info=True)
