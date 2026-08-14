"""Agent #4: writes PHP and CSS snippets for JetFormBuilder."""

from pathlib import Path
from typing import Any

from src.config import get_llm
from src.logging_config import get_logger
from src.state import SupportState

logger = get_logger(__name__)

KNOWLEDGE = Path(__file__).parent / "knowledge" / "jfb_hooks.md"

SYSTEM_PROMPT = """You are a WordPress developer on the Crocoblock support \
team. You write small PHP and CSS snippets that extend JetFormBuilder beyond \
its built-in settings.

Hook reference (the only hooks you may treat as verified):

{hooks}

Rules:
- Use only hooks from the reference above. If the task needs a hook that is \
not listed, say so plainly and describe what is known instead of inventing a \
signature.
- If the goal can be achieved through JetFormBuilder's own settings, say that \
first. A snippet that duplicates a built-in feature is a liability.
- Sanitize anything coming from $request or $_SERVER before using it \
(sanitize_text_field, absint, sanitize_email as appropriate).
- Prefer the built-in Action_Exception statuses over arbitrary strings, so \
the customer sees JetFormBuilder's standard error UI.
- Comment any non-obvious block, briefly.

Every answer must state:
1. Where the code goes: a code snippets plugin scoped to "Everywhere", or the \
child theme's functions.php.
2. What must be configured in the form itself, including the position of the \
Call Hook action in the Post-Submit Actions list when order matters.

Write for someone who may not know PHP. Explain what the code does, do not \
just hand it over. Keep the explanation shorter than the code."""


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


def generate(request: str, env_info: dict[str, Any] | None = None) -> str:
    """Write a snippet for the requested customisation.

    Args:
        request: What the customer wants the snippet to do.
        env_info: Site environment from the bug investigator, when available.

    Returns:
        The snippet with an explanation, as plain text.

    Raises:
        FileNotFoundError: If the curated hook reference is missing.
    """
    if not KNOWLEDGE.is_file():
        raise FileNotFoundError(f"Hook reference not found at {KNOWLEDGE}")

    hooks = KNOWLEDGE.read_text(encoding="utf-8")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(hooks=hooks)},
    ]

    if env_info:
        logger.info(
            "Using site environment: WordPress %s, PHP %s",
            env_info.get("wp_version"),
            env_info.get("php_version"),
        )
        messages.append(
            {
                "role": "system",
                "content": (
                    f"The customer's site runs WordPress "
                    f"{env_info.get('wp_version')} on PHP "
                    f"{env_info.get('php_version')}. Keep the code compatible."
                ),
            }
        )
    else:
        logger.debug("No environment info available, writing version-agnostic code")

    messages.append({"role": "user", "content": request})

    logger.info("Generating snippet (%d chars of hook reference)", len(hooks))
    response = get_llm("smart").invoke(messages)
    snippet = extract_text(response.content)
    logger.info("Snippet generated (%d chars)", len(snippet))
    return snippet

    messages.append({"role": "user", "content": request})

    response = get_llm("smart").invoke(messages)
    return extract_text(response.content)


def code_generator_node(state: SupportState) -> dict[str, object]:
    """Graph node: write a snippet for the customer's request."""
    request = str(state["messages"][-1].content)
    snippet = generate(request, state.get("env_info"))
    return {
        "final_answer": snippet,
        "handled_by": "code_generator",
    }
