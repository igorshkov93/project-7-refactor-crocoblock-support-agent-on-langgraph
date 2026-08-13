"""Central configuration: provider switching and model tiers."""

from langchain_core.language_models import BaseChatModel

from src.settings import Provider, Tier, settings

# "fast"  — classification and simple tasks
# "smart" — reasoning, diagnostics and code generation
MODELS: dict[Provider, dict[Tier, str]] = {
    "anthropic": {
        "fast": "claude-haiku-4-5-20251001",
        "smart": "claude-sonnet-5",
    },
    "gemini": {
        "fast": "gemini-2.5-flash",
        "smart": "gemini-2.5-flash",
    },
}


def get_llm(tier: Tier = "smart", temperature: float | None = None) -> BaseChatModel:
    """Return a LangChain chat model for the active provider.

    Args:
        tier: Model tier — "fast" for routing, "smart" for reasoning.
        temperature: Sampling temperature; falls back to the configured default.

    Returns:
        A configured chat model for the provider selected in settings.
    """
    provider = settings.llm_provider
    model_name = MODELS[provider][tier]
    temp = settings.default_temperature if temperature is None else temperature

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")

        # Sonnet 5 and newer models reject the temperature parameter.
        if "sonnet-5" in model_name:
            return ChatAnthropic(
                model_name=model_name,
                api_key=settings.anthropic_api_key,
                max_tokens_to_sample=settings.max_output_tokens,
                timeout=None,
                stop=None,
            )
        return ChatAnthropic(
            model_name=model_name,
            api_key=settings.anthropic_api_key,
            max_tokens_to_sample=settings.max_output_tokens,
            temperature=temp,
            timeout=None,
            stop=None,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY is not set")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.google_api_key,
        temperature=temp,
        max_output_tokens=settings.max_output_tokens,
    )