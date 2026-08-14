"""Check Code Generator on realistic snippet requests."""
from src.agents.code_generator import generate

REQUESTS = [
    "I need PHP code to validate that the email field uses a company domain, "
    "and block submission if it doesn't",
    "Give me a snippet that adds a custom option to the Select field using "
    "the option-query hook",
]


def main():
    for request in REQUESTS:
        print(f"\n{'=' * 70}\nRequest: {request}\n")
        print(generate(request))


if __name__ == "__main__":
    main()
