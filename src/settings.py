"""Application settings: a single validated source of configuration."""

from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["anthropic", "gemini"]
Tier = Literal["fast", "smart"]


class Settings(BaseSettings):
    """Configuration loaded from the environment and validated at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Provider selection
    llm_provider: Provider = "gemini"

    # Credentials
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    pinecone_api_key: SecretStr | None = None
    cohere_api_key: SecretStr | None = None

    # WordPress sandbox
    wp_base_url: str = ""
    wp_user: str = ""
    wp_app_password: SecretStr | None = None

    # Model behaviour
    max_output_tokens: int = Field(default=4000, gt=0)
    default_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # Routing
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    # Retrieval
    pinecone_index: str = "jetformbuilder-docs"
    pinecone_namespace: str = "jfb"
    retrieval_candidates: int = Field(default=20, gt=0)
    retrieval_top_n: int = Field(default=5, gt=0)

    # Human-in-the-loop
    max_clarifying_rounds: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def check_active_provider_key(self) -> "Settings":
        """Fail fast when the selected provider has no credentials."""
        required = {
            "anthropic": self.anthropic_api_key,
            "gemini": self.google_api_key,
        }
        if required[self.llm_provider] is None:
            raise ValueError(
                f"LLM_PROVIDER is '{self.llm_provider}' but the matching API key is not set. "
                f"Add it to your .env file."
            )
        return self

# Timeouts for external dependencies, in seconds
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    pinecone_timeout_seconds: float = Field(default=10.0, gt=0)
    cohere_timeout_seconds: float = Field(default=15.0, gt=0)
    wp_timeout_seconds: float = Field(default=20.0, gt=0)

    # Retry policy for transient failures
    retry_attempts: int = Field(default=3, ge=1, le=10)
    retry_initial_wait: float = Field(default=1.0, gt=0)
    retry_max_wait: float = Field(default=10.0, gt=0)

    @model_validator(mode="after")
    def check_retry_waits(self) -> "Settings":
        """A back-off ceiling below the initial wait would silently be ignored."""
        if self.retry_max_wait < self.retry_initial_wait:
            raise ValueError(
                "RETRY_MAX_WAIT must be greater than or equal to RETRY_INITIAL_WAIT."
            )
        return self

settings = Settings()
