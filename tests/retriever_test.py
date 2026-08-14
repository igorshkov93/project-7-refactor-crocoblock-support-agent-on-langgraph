"""Check retrieval quality on realistic support questions."""
from src.rag.retriever import search

QUERIES = [
    "How do I save form submissions to the database?",
    "how to send email after form submit",
    "why is my select field empty",
]


def main():
    for query in QUERIES:
        print(f"\n{'=' * 70}\n{query}\n")
        for rank, hit in enumerate(search(query), 1):
            print(f"  {rank}. [{hit['rerank_score']:.3f}] {hit['title']}")
            print(f"     vector: {hit['vector_score']:.3f}  {hit['url']}")


if __name__ == "__main__":
    main()
