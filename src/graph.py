
"""LangGraph assembly of the multi-agent support system."""
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.bug_investigator import bug_investigator_node
from src.agents.code_generator import code_generator_node
from src.agents.docs_qa import docs_qa_node
from src.agents.router import router_node
from src.logging_config import get_logger
from src.settings import settings
from src.state import SupportState

logger = get_logger(__name__)

def escalate_node(state: SupportState) -> dict[str, object]:
    """Hand the ticket over to a human agent."""
    logger.info(
        "Escalating to a human (type=%s, confidence=%.2f)",
        state.get("query_type", "unknown"),
        state.get("confidence", 0.0),
    )
    return {
        "final_answer": (
            "This request needs a human support agent. "
            "Your ticket has been escalated."
        ),
        "needs_human": True,
        "handled_by": "escalation",
    }



def route_after_router(state: SupportState) -> str:
    """Decide which agent handles the query."""
    confidence = state.get("confidence", 0.0)
    if confidence < settings.confidence_threshold:
        logger.info(
            "Confidence %.2f below threshold %.2f, routing to escalation",
            confidence,
            settings.confidence_threshold,
        )
        return "escalate"

    destinations: dict[str, str] = {
        "how_to": "docs_qa",
        "bug": "bug_investigator",
        "code": "code_generator",
        "rest": "escalate",
    }
    query_type = state.get("query_type")
    destination = destinations.get(query_type or "", "escalate")
    logger.debug("Routing '%s' to node '%s'", query_type, destination)
    return destination

def build_graph() -> CompiledStateGraph[SupportState, Any, Any, Any]:
    """Assemble and compile the support graph."""
    builder = StateGraph(SupportState)

    builder.add_node("router", router_node)
    builder.add_node("docs_qa", docs_qa_node)
    builder.add_node("bug_investigator", bug_investigator_node)
    builder.add_node("code_generator", code_generator_node)
    builder.add_node("escalate", escalate_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_after_router,
        ["docs_qa", "bug_investigator", "code_generator", "escalate"],
    )
    def route_after_investigation(state: SupportState) -> str:
        """Loop back for another round if the investigation is unfinished."""
        if state.get("final_answer"):
            logger.debug("Investigation finished, ending graph")
            return END
        logger.debug("Investigation continues, looping back")
        return "bug_investigator"

    builder.add_conditional_edges(
        "bug_investigator", route_after_investigation, ["bug_investigator", END]
    )
    builder.add_edge("docs_qa", END)
    builder.add_edge("code_generator", END)
    builder.add_edge("escalate", END)


    logger.debug("Graph compiled with in-memory checkpointer")
    return builder.compile(checkpointer=InMemorySaver())


graph = build_graph()
