"""Check Docs Q&A answers on realistic support questions."""
from src.agents.docs_qa import answer

QUERIES = [
    "How do I save form submissions to the database?",
    "How can I send a confirmation email after the form is submitted?",
    "How do I connect JetFormBuilder to my Salesforce CRM?",
]


def main():
    for query in QUERIES:
        result = answer(query)
        print(f"\n{'=' * 70}\nQ: {query}\n")
        print(result["answer"])


if __name__ == "__main__":
    main()
