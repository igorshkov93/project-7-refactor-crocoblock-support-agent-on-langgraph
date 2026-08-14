"""Single entry point into the graph, shared by the CLI and Streamlit."""

import uuid
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.graph import graph as _graph
from src.logging_config import get_logger

logger = get_logger(__name__)


def new_thread_id() -> str:
    """Create an identifier for a new conversation."""
    return str(uuid.uuid4())


def _config(thread_id: str) -> RunnableConfig:
    """Build the LangGraph config that binds a run to a conversation."""
    return {"configurable": {"thread_id": thread_id}}


def _pending_question(config: RunnableConfig) -> str | None:
    """Return the clarifying question if the graph is suspended on interrupt()."""
    snapshot = _graph.get_state(config)
    for task in snapshot.tasks:
        if task.interrupts:
            payload = task.interrupts[0].value
            if isinstance(payload, dict):
                return str(payload.get("question") or payload)
            return str(payload)
    return None


def start(query: str, thread_id: str) -> tuple[dict[str, Any], str | None]:
    """Run the graph on a new question.

    Args:
        query: The customer's message.
        thread_id: Conversation identifier from :func:`new_thread_id`.

    Returns:
        The resulting state, and a clarifying question when the graph paused.
    """
    logger.info("Starting run on thread %s", thread_id[:8])
    config = _config(thread_id)
    _graph.invoke({"messages": [HumanMessage(content=query)]}, config)
    return dict(_graph.get_state(config).values), _pending_question(config)


def resume(answer: str, thread_id: str) -> tuple[dict[str, Any], str | None]:
    """Resume a suspended run with the customer's reply.

    Args:
        answer: The customer's answer to the clarifying question.
        thread_id: The same identifier used to start the conversation.

    Returns:
        The resulting state, and a further question when the graph paused again.
    """
    logger.info("Resuming thread %s", thread_id[:8])
    config = _config(thread_id)
    _graph.invoke(Command(resume=answer), config)
    return dict(_graph.get_state(config).values), _pending_question(config)
