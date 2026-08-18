"""Agent #3: investigates bug reports using live site diagnostics."""

from typing import Any

from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.async_bridge import run_sync
from src.config import get_llm
from src.exceptions import InvestigationError, MCPServerError
from src.llm_call import invoke_structured
from src.logging_config import get_logger
from src.mcp_server.client import load_tools
from src.settings import settings
from src.state import SupportState

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a second-line support engineer for Crocoblock \
WordPress plugins. A customer reports that something is broken.

You have read-only access to their site through these tools:
- get_env_info: WordPress, PHP and MySQL versions, active theme, debug settings
- list_plugins: installed plugins with versions and activation status
- get_error_log: the tail of the WordPress debug log

Process:
1. Call get_env_info first. It is cheap and almost always relevant.
2. Call list_plugins when the report involves a feature that could conflict \
with another plugin, or when a version mismatch is plausible.
3. Call get_error_log when the customer reports a fatal error, a white \
screen, a 500 response, or any failure with no visible message.

Then decide:
- If the diagnostics point to a likely cause, write your findings.
- If you need information only the customer can provide (what they see on \
screen, which settings are configured, what they did before it broke), ask \
for it instead of guessing.

When writing findings:
- State what you checked and what you found.
- Give the most likely cause and how confident you are.
- Never claim a cause the diagnostics do not support.
- Do not include raw JSON or tool names. Write for a customer.

Keep it under 200 words."""

DIAGNOSTICS_UNAVAILABLE = (
    "I couldn't connect to your site to run diagnostics right now. "
    "A human support agent will pick this up and take a look."
)


class Findings(BaseModel):
    """Structured outcome of a diagnostic round."""

    needs_input: bool = Field(
        description="True if customer input is required to proceed"
    )
    message: str = Field(
        description="Either the findings, or the questions for the customer"
    )


def extract_text(content: str | list[Any]) -> str:
    """Get plain text from a response that may contain thinking blocks."""
    if isinstance(content, str):
        return content
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(p for p in parts if p).strip()


def run_diagnostics(messages: list[dict[str, Any]]) -> str:
    """Run the ReAct agent over the conversation so far.

    Args:
        messages: Conversation history in chat-message form.

    Returns:
        The agent's latest message as plain text.

    Raises:
        MCPServerError: If the diagnostic tools could not be loaded.
        InvestigationError: If the agent produced no readable message.
        LLMError: If the provider call failed after retries.
    """
    logger.info("Running diagnostics over %d message(s)", len(messages))
    tools = load_tools()

    agent = create_react_agent(
        model=get_llm("smart"),
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )

    result = run_sync(agent.ainvoke({"messages": messages}))

    text = extract_text(result["messages"][-1].content)
    if not text:
        raise InvestigationError(
            "The diagnostic agent returned no readable message"
        )

    logger.debug("Diagnostics produced %d chars", len(text))
    return text


def classify_response(text: str) -> Findings:
    """Decide whether the agent is asking or concluding.

    Raises:
        LLMResponseError: If the model's answer does not fit Findings.
        LLMError: If the provider call failed after retries.
    """
    return invoke_structured(
        get_llm("fast"),
        [
            {
                "role": "system",
                "content": (
                    "Read the support engineer's message. Decide whether it "
                    "primarily asks the customer for information needed to "
                    "continue (needs_input=true), or presents a diagnosis "
                    "(needs_input=false). Return the message unchanged."
                ),
            },
            {"role": "user", "content": text},
        ],
        Findings,
    )


def bug_investigator_node(state: SupportState) -> dict[str, object]:
    """Graph node: diagnose, asking follow-up questions when needed."""
    rounds = state.get("clarifying_rounds", 0)
    history = state.get("investigation_log", [])
    logger.info("Investigation round %d (max %d)", rounds + 1, settings.max_clarifying_rounds)

    conversation: list[dict[str, Any]] = [
        {"role": "user", "content": state["messages"][0].content}
    ]
    for entry in history:
        conversation.append({"role": "assistant", "content": entry["question"]})
        conversation.append({"role": "user", "content": entry["reply"]})

    try:
        raw = run_diagnostics(conversation)
    except (MCPServerError, InvestigationError) as error:
        logger.error(
            "Diagnostics unavailable, escalating",
            extra={"error_type": type(error).__name__, "error_detail": str(error)},
        )
        return {
            "final_answer": DIAGNOSTICS_UNAVAILABLE,
            "clarifying_rounds": rounds,
            "handled_by": "bug_investigator",
            "needs_human": True,
        }

    verdict = classify_response(raw)

    if not verdict.needs_input or rounds >= settings.max_clarifying_rounds:
        if verdict.needs_input:
            logger.warning("Round limit reached with open questions, handing over to a human")
        else:
            logger.info("Diagnosis complete after %d round(s)", rounds + 1)
        return {
            "final_answer": verdict.message,
            "clarifying_rounds": rounds,
            "handled_by": "bug_investigator",
            "needs_human": verdict.needs_input,
        }

    logger.info("Asking the customer a clarifying question, suspending the graph")
    reply = interrupt({"question": verdict.message, "round": rounds + 1})
    logger.info("Resumed with the customer's reply")

    return {
        "clarifying_rounds": rounds + 1,
        "investigation_log": history + [{"question": verdict.message, "reply": reply}],
        "handled_by": "bug_investigator",
    }
