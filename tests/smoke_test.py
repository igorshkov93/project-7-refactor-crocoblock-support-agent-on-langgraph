"""Verify that the LLM layer is wired up correctly."""
from src.config import LLM_PROVIDER, MODELS, get_llm

PROMPT = (
    "Reply with exactly one word: the name of the Crocoblock plugin "
    "that adds custom post types and meta fields to WordPress."
)


def main():
    print(f"Provider: {LLM_PROVIDER}\n")
    for tier in ("fast", "smart"):
        model_name = MODELS[LLM_PROVIDER][tier]
        llm = get_llm(tier)
        response = llm.invoke(PROMPT)
        print(f"  [{tier}] {model_name} -> {response.content.strip()}")


if __name__ == "__main__":
    main()
