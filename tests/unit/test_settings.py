"""Settings load from the environment and validate their own invariants."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.settings import Settings, settings


def test_settings_import_without_dotenv():
    """The module-level singleton is constructed from the stubbed environment."""
    assert settings.llm_provider == "anthropic"
    assert settings.anthropic_api_key is not None


def test_secret_is_not_exposed_in_repr():
    """API keys must not leak into logs or tracebacks."""
    assert "test-anthropic-key" not in repr(settings)


def test_defaults_match_documented_behaviour():
    """Values other modules depend on: threshold, retries, retrieval sizes."""
    assert settings.confidence_threshold == 0.6
    assert settings.retry_attempts == 3
    assert settings.retrieval_candidates > settings.retrieval_top_n


def test_missing_key_for_active_provider_is_rejected():
    """Selecting a provider without its key fails fast, not at first call."""
    with pytest.raises(ValidationError, match="matching API key"):
        Settings(llm_provider="gemini", google_api_key=None, _env_file=None)


def test_retry_ceiling_below_initial_wait_is_rejected():
    """A ceiling under the initial wait would be silently ignored by tenacity."""
    with pytest.raises(ValidationError, match="RETRY_MAX_WAIT"):
        Settings(retry_initial_wait=5.0, retry_max_wait=1.0)


def test_tracing_without_key_is_rejected():
    """Tracing enabled without a key loses every trace silently."""
    with pytest.raises(ValidationError, match="LANGSMITH_API_KEY"):
        Settings(langsmith_tracing=True, langsmith_api_key=None, _env_file=None)
