"""Domain exception hierarchy for the Crocoblock support agent.

Every error raised by application code inherits from :class:`SupportAgentError`,
so callers can tell "our" failures apart from bugs in third-party libraries.

Each exception carries a ``retryable`` class flag: transient failures (timeouts,
rate limits, network blips) are retried by the tenacity policies added in the
next sub-step, while deterministic ones (bad credentials, malformed payloads,
invalid configuration) fail fast.
"""

from __future__ import annotations

from typing import ClassVar

__all__ = [
    "AgentError",
    "ConfigurationError",
    "EmbeddingError",
    "ExternalServiceError",
    "InvestigationError",
    "LLMError",
    "LLMRateLimitError",
    "LLMResponseError",
    "LLMTimeoutError",
    "MCPServerError",
    "RerankError",
    "RetrievalError",
    "RoutingError",
    "SupportAgentError",
    "VectorStoreError",
    "WordPressAuthError",
    "WordPressError",
    "WordPressRequestError",
    "WordPressResponseError",
    "WordPressTimeoutError",
]


class SupportAgentError(Exception):
    """Base class for every error raised by application code."""

    retryable: ClassVar[bool] = False


class ConfigurationError(SupportAgentError):
    """A setting is missing, empty or invalid. Never retried."""


# --- External dependencies -------------------------------------------------


class ExternalServiceError(SupportAgentError):
    """Failure while talking to an external dependency."""

    service: ClassVar[str] = "unknown"

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code

    def __str__(self) -> str:
        if self.status_code is None:
            return f"[{self.service}] {self.message}"
        return f"[{self.service}] {self.message} (status={self.status_code})"


class LLMError(ExternalServiceError):
    """The LLM provider (Anthropic or Gemini) failed to answer."""

    service: ClassVar[str] = "llm"
    retryable: ClassVar[bool] = True


class LLMTimeoutError(LLMError):
    """The provider did not respond within the configured timeout."""


class LLMRateLimitError(LLMError):
    """The provider returned a rate-limit or overloaded response."""


class LLMResponseError(LLMError):
    """The model answered, but the output could not be parsed or validated.

    Raised for structured-output failures — e.g. the router returning a
    classification that does not fit the Pydantic schema. Retrying the same
    prompt is unlikely to help, so this one is not retryable.
    """

    retryable: ClassVar[bool] = False


class RetrievalError(ExternalServiceError):
    """Failure anywhere in the two-stage retrieval pipeline."""

    service: ClassVar[str] = "retrieval"
    retryable: ClassVar[bool] = True


class EmbeddingError(RetrievalError):
    """Cohere embed-v4.0 failed to embed the query."""

    service: ClassVar[str] = "cohere-embed"


class VectorStoreError(RetrievalError):
    """Pinecone search failed or returned an unusable payload."""

    service: ClassVar[str] = "pinecone"


class RerankError(RetrievalError):
    """Cohere rerank-v3.5 failed to reorder the candidates."""

    service: ClassVar[str] = "cohere-rerank"


class WordPressError(ExternalServiceError):
    """The WordPress sandbox could not be reached or refused the request."""

    service: ClassVar[str] = "wordpress"
    retryable: ClassVar[bool] = True


class WordPressTimeoutError(WordPressError):
    """The WP REST API did not respond within the configured timeout."""


class WordPressAuthError(WordPressError):
    """Application password rejected (401/403). Retrying will not help."""

    retryable: ClassVar[bool] = False

class WordPressRequestError(WordPressError):
    """The WP REST API rejected the request itself (404, 400 and similar).

    Deterministic: the same request will be rejected again.
    """

    retryable: ClassVar[bool] = False

class WordPressResponseError(WordPressError):
    """The WP REST API returned a payload of an unexpected shape.

    The endpoints exposed by the must-use plugin may answer with either an
    object or a list depending on the error path, so every response is
    narrowed explicitly instead of being trusted as ``dict``.
    """

    retryable: ClassVar[bool] = False


class MCPServerError(ExternalServiceError):
    """The MCP server could not be started or stopped responding.

    Covers the transport itself — a subprocess that fails to launch, hangs on
    handshake, or dies mid-session. Failures *inside* a tool surface as the
    WordPress errors above.
    """

    service: ClassVar[str] = "mcp"
    retryable: ClassVar[bool] = True


# --- Agent-level failures --------------------------------------------------


class AgentError(SupportAgentError):
    """Failure inside an agent node that is not caused by a remote service."""


class RoutingError(AgentError):
    """The router could not produce a usable classification."""


class InvestigationError(AgentError):
    """The bug investigator reached an inconsistent state."""
