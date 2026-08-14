"""Central configuration: provider switching and model tiers."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from src.exceptions import ConfigurationError
from src.settings import Provider, Tier, settings

# "fast"  — classification and simple tasks
# "smart" — reasoning, diagnostics and code generation
MODELS: dict[Provider, dict[Tier, str]] = {
    "anthropic": {
        "fast": "claude-haiku-4-5-20251001",
        "smart": "claude-sonnet-5",
    },
    "gemini": {
        "fast": "gemini-2.5-flash-lite",
        "smart": "gemini-2.5-flash",
    },
}

#: Retries are handled by :func:`src.retry.with_retry` at the call site, so the
#: SDK's own retry loop is disabled — two independent loops multiply the wait
#: before a failure ever surfaces.
SDK_MAX_RETRIES = 0


def _anthropic_model(model_name: str, temperature: float) -> BaseChatModel:
    """Build an Anthropic chat model with a request timeout applied."""
    from langchain_anthropic import ChatAnthropic

    if settings.anthropic_api_key is None:
        raise ConfigurationError("ANTHROPIC_API_KEY is not set in .env")

    kwargs: dict[str, Any] = {
        "model_name": model_name,
        "api_key": settings.anthropic_api_key,
        "max_tokens_to_sample": settings.max_output_tokens,
        "timeout": settings.llm_timeout_seconds,
        "max_retries": SDK_MAX_RETRIES,
        "stop": None,
    }
    # Sonnet 5 and newer models reject the temperature parameter.
    if "sonnet-5" not in model_name:
        kwargs["temperature"] = temperature
    return ChatAnthropic(**kwargs)


def _gemini_model(model_name: str, temperature: float) -> BaseChatModel:
    """Build a Gemini chat model with a request timeout applied."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    if settings.google_api_key is None:
        raise ConfigurationError("GOOGLE_API_KEY is not set in .env")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.google_api_key,
        temperature=temperature,
        max_output_tokens=settings.max_output_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=SDK_MAX_RETRIES,
    )


def get_llm(tier: Tier = "smart", temperature: float | None = None) -> BaseChatModel:
    """Return a LangChain chat model for the active provider.

    Args:
        tier: Model tier — "fast" for routing, "smart" for reasoning.
        temperature: Sampling temperature; falls back to the configured default.

    Returns:
        A configured chat model for the provider selected in settings, with
        ``LLM_TIMEOUT_SECONDS`` applied and SDK-level retries disabled.

    Raises:
        ConfigurationError: If the selected provider has no API key.
    """
    provider = settings.llm_provider
    model_name = MODELS[provider][tier]
    temp = settings.default_temperature if temperature is None else temperature

    if provider == "anthropic":
        return _anthropic_model(model_name, temp)
    return _gemini_model(model_name, temp)
