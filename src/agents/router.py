
"""Agent #1: classifies incoming support requests."""
from pydantic import BaseModel, Field

from src.config import get_llm
from src.llm_call import invoke_structured
from src.logging_config import get_logger
from src.state import QueryType, SupportState

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the first-line triage agent for Crocoblock plugin \
support (JetFormBuilder, JetEngine). Classify each customer message into \
exactly one category.

Categories:
- how_to: the customer asks how to use an existing feature, how something \
works, or which setting to use. The answer exists in the documentation.
- bug: something is broken, missing, or behaves unexpectedly. Includes white \
screens, errors, and features that stopped working. The cause is unknown and \
must be investigated on the customer's site.
- code: the customer explicitly asks for a PHP snippet, CSS rule, or hook to \
extend functionality beyond what the plugin offers out of the box.
- rest: pre-sales questions, pricing, licensing, refunds, feature requests, \
feedback, greetings, and anything else. These need a human.

Rules:
- If the customer reports something broken AND asks for code, classify as \
bug. The problem must be diagnosed before any fix is written.
- If the customer asks how to achieve something that the plugin supports \
natively, that is how_to, not code.
- Set confidence below 0.6 when the message is too vague to classify \
reliably.

Reply with the category, a confidence score between 0 and 1, and one short \
sentence explaining your choice."""


class RoutingDecision(BaseModel):
    """Structured output of the router agent."""

    query_type: QueryType = Field(description="One of: how_to, bug, code, rest")

    confidence: float = Field(
        description="Confidence between 0.0 and 1.0", ge=0.0, le=1.0
    )
    reason: str = Field(description="One short sentence explaining the choice")


def classify(query: str) -> RoutingDecision:
    """Classify a single customer message.

    Args:
        query: Raw customer message.

    Returns:
        The routing decision with a confidence score.

    Raises:
        LLMResponseError: If the model's answer does not fit RoutingDecision.
        LLMError: If the provider call failed after retries.
    """
    return invoke_structured(
        get_llm("fast"),
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        RoutingDecision,
    )


def router_node(state: SupportState) -> dict[str, object]:
    """Graph node: classify the latest user message."""
    query = str(state["messages"][-1].content)
    logger.info("Routing query: %.60s", query)

    decision = classify(query)

    logger.info(
        "Routed as '%s' (confidence %.2f): %s",
        decision.query_type,
        decision.confidence,
        decision.reason,
    )
    return {
        "query_type": decision.query_type,
        "confidence": decision.confidence,
        "routing_reason": decision.reason,
    }
