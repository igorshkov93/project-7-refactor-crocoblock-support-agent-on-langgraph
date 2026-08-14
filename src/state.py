"""Shared state passed between agents in the support graph."""

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

QueryType = Literal["how_to", "bug", "code", "rest"]


class SupportState(TypedDict, total=False):
    """State object flowing through the multi-agent graph.

    Every agent reads what it needs and writes only its own fields.
    """

    # Conversation
    messages: Annotated[list[AnyMessage], add_messages]

    # Router output
    query_type: QueryType
    confidence: float
    routing_reason: str

    # Bug Investigator
    env_info: dict[str, Any]
    clarifying_rounds: int
    investigation_log: list[dict[str, str]]

    # Docs Q&A
    retrieved_docs: list[dict[str, Any]]

    # Output
    final_answer: str
    needs_human: bool
    handled_by: str
