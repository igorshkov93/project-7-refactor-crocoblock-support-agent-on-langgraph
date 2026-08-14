"""A single guarded entry point for calling a chat model.

Wrapping ``invoke`` in every agent would mean four copies of the same
try/except. Instead both call shapes live here: plain text generation and
structured output validated against a Pydantic schema. Provider failures become
:class:`~src.exceptions.LLMError` subclasses, and a model that answers with the
wrong shape becomes :class:`~src.exceptions.LLMResponseError` — retried in the
first case, surfaced immediately in the second.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ValidationError

from src.exceptions import LLMError, LLMRateLimitError, LLMResponseError, LLMTimeoutError
from src.logging_config import get_logger
from src.retry import with_retry

logger = get_logger(__name__)


Messages = list[dict[str, str]]

#: Substrings that identify a transient provider failure. Both SDKs raise their
#: own exception types, and matching on the message is the only portable option
#: that survives a provider switch.
_RATE_LIMIT_MARKERS = ("rate limit", "429", "quota", "overloaded", "resource_exhausted")
_TIMEOUT_MARKERS = ("timeout", "timed out", "deadline")


def _translate(error: Exception) -> LLMError:
    """Map a provider exception onto the matching domain exception."""
    text = str(error).lower()
    if any(marker in text for marker in _TIMEOUT_MARKERS):
        return LLMTimeoutError(f"Provider timed out: {error}")
    if any(marker in text for marker in _RATE_LIMIT_MARKERS):
        return LLMRateLimitError(f"Provider is throttling: {error}")
    return LLMError(f"Provider call failed: {error}")


@with_retry()
def invoke_text(llm: BaseChatModel, messages: Messages) -> str:
    """Call the model and return its answer as plain text.

    Raises:
        LLMTimeoutError: If the provider did not answer in time.
        LLMRateLimitError: If the provider is throttling.
        LLMError: On any other provider failure.
        LLMResponseError: If the answer carries no text content.
    """
    try:
        response = llm.invoke(messages)
    except Exception as error:
        raise _translate(error) from error

    content = response.content
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        # Some providers return a list of content blocks.
        parts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and "text" in block
        ]
        text = "".join(parts)
    else:
        text = ""

    if not text.strip():
        raise LLMResponseError("Model returned an empty answer")
    return text


@with_retry()
def invoke_structured[SchemaT: BaseModel](
    llm: BaseChatModel, messages: Messages, schema: type[SchemaT]
) -> SchemaT:
    """Call the model and validate its answer against a Pydantic schema.

    Args:
        llm: The chat model to call.
        messages: Conversation to send, in LangChain's dict form.
        schema: The expected output model.

    Raises:
        LLMTimeoutError: If the provider did not answer in time.
        LLMRateLimitError: If the provider is throttling.
        LLMError: On any other provider failure.
        LLMResponseError: If the answer does not fit the schema. Not retried —
            the same prompt would produce the same malformed output.
    """
    structured: Any = llm.with_structured_output(schema)
    try:
        result = structured.invoke(messages)
    except ValidationError as error:
        raise LLMResponseError(
            f"Model output does not fit {schema.__name__}: {error}"
        ) from error
    except Exception as error:
        raise _translate(error) from error

    if not isinstance(result, schema):
        raise LLMResponseError(
            f"Expected {schema.__name__}, got {type(result).__name__}"
        )
    return result
