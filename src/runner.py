"""Single entry point into the graph, shared by the CLI and Streamlit."""
import uuid
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.logging_config import get_logger
from src.observability import configure_tracing
from src.settings import settings

# Tracing must be configured before the graph is imported: importing src.graph
# compiles the graph and instantiates LangChain clients, which read their
# tracing configuration from the environment at construction time.
configure_tracing()

from src.graph import graph as _graph  # noqa: E402

logger = get_logger(__name__)


def new_thread_id() -> str:
    """Create an identifier for a new conversation."""
    return str(uuid.uuid4())


def _config(thread_id: str, interface: str, run_name: str) -> RunnableConfig:
    """Build the LangGraph config that binds a run to a conversation.

    Beyond the checkpointer's thread_id, this carries the tags and metadata
    that make traces searchable in LangSmith: which provider answered, which
    interface asked, and the retrieval settings in force at the time. Without
    them every run looks alike and a regression cannot be traced back to the
    configuration that caused it.
    """
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": run_name,
        "tags": [
            "crocoblock-support",
            f"provider:{settings.llm_provider}",
            f"interface:{interface}",
        ],
        "metadata": {
            "thread_id": thread_id,
            "interface": interface,
            "llm_provider": settings.llm_provider,
            "confidence_threshold": settings.confidence_threshold,
            "pinecone_index": settings.pinecone_index,
            "pinecone_namespace": settings.pinecone_namespace,
            "embed_model": settings.embed_model,
            "rerank_model": settings.rerank_model,
            "retrieval_candidates": settings.retrieval_candidates,
            "retrieval_top_n": settings.retrieval_top_n,
        },
    }


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


def start(
    query: str, thread_id: str, interface: str = "cli"
) -> tuple[dict[str, Any], str | None]:
    """Run the graph on a new question.

    Args:
        query: The customer's message.
        thread_id: Conversation identifier from :func:`new_thread_id`.
        interface: Which front end is asking, recorded in the trace.

    Returns:
        The resulting state, and a clarifying question when the graph paused.
    """
    logger.info("Starting run on thread %s", thread_id[:8])
    config = _config(thread_id, interface, "crocoblock-support-agent")
    _graph.invoke({"messages": [HumanMessage(content=query)]}, config)
    return dict(_graph.get_state(config).values), _pending_question(config)


def resume(
    answer: str, thread_id: str, interface: str = "cli"
) -> tuple[dict[str, Any], str | None]:
    """Resume a suspended run with the customer's reply.

    Args:
        answer: The customer's answer to the clarifying question.
        thread_id: The same identifier used to start the conversation.
        interface: Which front end is asking, recorded in the trace.

    Returns:
        The resulting state, and a further question when the graph paused again.
    """
    logger.info("Resuming thread %s", thread_id[:8])
    config = _config(thread_id, interface, "crocoblock-support-resume")
    _graph.invoke(Command(resume=answer), config)
    return dict(_graph.get_state(config).values), _pending_question(config)
