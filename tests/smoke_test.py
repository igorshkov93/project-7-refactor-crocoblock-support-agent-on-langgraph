"""Verify that the LLM layer is wired up correctly."""
from src.config import MODELS, get_llm
from src.settings import settings

PROMPT = (
    "Reply with exactly one word: the name of the Crocoblock plugin "
    "that adds custom post types and meta fields to WordPress."
)


def main():
    print(f"Provider: {settings.llm_provider}\n")
    for tier in ("fast", "smart"):
        model_name = MODELS[settings.llm_provider][tier]
        llm = get_llm(tier)
        response = llm.invoke(PROMPT)
        print(f"  [{tier}] {model_name} -> {response.content.strip()}")


if __name__ == "__main__":
    main()
