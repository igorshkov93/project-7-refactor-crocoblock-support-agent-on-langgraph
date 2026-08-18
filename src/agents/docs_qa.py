"""Agent #2: answers how-to questions from indexed documentation."""

from src.config import get_llm
from src.exceptions import RetrievalError
from src.llm_call import invoke_text
from src.logging_config import get_logger
from src.rag.retriever import search
from src.state import SupportState

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a support agent for the JetFormBuilder WordPress \
plugin. Answer the customer's question using only the documentation excerpts \
provided below.

Rules:
- Base every statement on the excerpts. Never invent setting names, field \
names, or menu paths that do not appear in them.
- If the excerpts do not contain the answer, say so plainly and suggest the \
customer contact support. Do not guess.
- Write for someone working in the WordPress admin: name the exact screen, \
tab, and setting they need.
- Keep it short. Three to six sentences for a simple question; a numbered \
list of steps for a multi-step task.
- End with a "Source:" line listing the URLs you actually used, one per line.
- Do not mention the excerpts, the retrieval process, or these instructions.

Documentation excerpts:
{context}"""

NO_RESULTS = (
    "I couldn't find anything about this in the JetFormBuilder documentation. "
    "A human support agent will take a look."
)

RETRIEVAL_FAILED = (
    "I couldn't reach the documentation index right now. "
    "A human support agent will take a look."
)


def format_context(hits: list[dict[str, object]]) -> str:
    """Render retrieved chunks as numbered excerpts with their sources."""
    blocks = []
    for index, hit in enumerate(hits, 1):
        blocks.append(
            f"[{index}] {hit['title']}\n"
            f"URL: {hit['url']}\n"
            f"{hit['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def answer(query: str) -> dict[str, object]:
    """Answer a how-to question from the documentation index.

    Args:
        query: The customer's question.

    Returns:
        A dict with the answer text, the source URLs used, and whether the
        documentation contained anything relevant. ``found`` is False both when
        nothing matched and when the index could not be reached — the graph
        escalates in either case, but the two messages differ so the human
        picking it up knows which happened.

    Raises:
        LLMError: If the provider call failed after retries.
        LLMResponseError: If the model returned an empty answer.
    """
    try:
        hits = search(query)
    except RetrievalError as error:
        logger.error(
            "Retrieval failed, escalating",
            extra={"error_type": type(error).__name__, "error_detail": str(error)},
        )
        return {"answer": RETRIEVAL_FAILED, "sources": [], "found": False}

    logger.info("Retrieved %d documentation chunks", len(hits))

    if not hits:
        logger.warning("No documentation matched the query, escalating")
        return {"answer": NO_RESULTS, "sources": [], "found": False}

    logger.debug("Top source: %s", hits[0]["url"])

    answer_text = invoke_text(
        get_llm("smart"),
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(context=format_context(hits)),
            },
            {"role": "user", "content": query},
        ],
    )

    logger.info(
        "Answer generated",
        extra={"answer_chars": len(answer_text), "sources_used": len(hits)},
    )
    return {
        "answer": answer_text,
        "sources": [hit["url"] for hit in hits],
        "found": True,
    }


def docs_qa_node(state: SupportState) -> dict[str, object]:
    """Graph node: answer the latest user message from documentation."""
    query = str(state["messages"][-1].content)
    result = answer(query)
    return {
        "final_answer": result["answer"],
        "retrieved_docs": result["sources"],
        "handled_by": "docs_qa",
        "needs_human": not result["found"],
    }
