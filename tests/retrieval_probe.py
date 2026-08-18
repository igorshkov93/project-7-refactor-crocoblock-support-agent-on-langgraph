"""Trace one query through both retrieval stages separately.

A miss in the final results does not say which stage lost the page. This prints
the position of an expected page after dense search and after reranking, and
can rerank a second time with the page title prepended to each chunk, to test
whether the reranker is missing the context that the chunk text alone omits.
"""
import argparse
from typing import Any

from src.rag.retriever import _embed_query, _rerank, _vector_search
from src.settings import settings


def _show(
    title: str,
    ranked: list[tuple[int, float]],
    matches: list[dict[str, Any]],
    expected: str,
) -> None:
    """Print one ranked list, marking the expected page."""
    print(f"\n{title}")
    for position, (index, score) in enumerate(ranked, start=1):
        url = str(matches[index]["metadata"]["url"])
        marker = ">>" if expected in url else "  "
        print(f"  {marker} {position:3d}  {score:.4f}  {url}")


def probe(query: str, expected: str, candidates: int) -> None:
    """Report where an expected page ranks at each stage."""
    embedding = _embed_query(query)
    matches = _vector_search(embedding, candidates)

    print(f"\nQuery: {query}")
    print(f"Expected slug: '{expected}'")
    print(f"\nStage 1 — dense search, {len(matches)} candidates:")
    found_at = None
    for position, match in enumerate(matches, start=1):
        url = str(match["metadata"]["url"])
        if expected in url and found_at is None:
            found_at = position
        marker = ">>" if expected in url else "  "
        print(f"  {marker} {position:3d}  {match['score']:.4f}  {url}")

    if found_at is None:
        print(f"\n  '{expected}' is NOT among the {len(matches)} candidates.")
        return

    print(f"\n  '{expected}' found at candidate position {found_at}.")

    plain = [match["metadata"]["text"] for match in matches]
    _show(
        f"Stage 2a — rerank on chunk text, top {settings.retrieval_top_n}:",
        _rerank(query, plain, settings.retrieval_top_n),
        matches,
        expected,
    )

    titled = [
        f"{match['metadata']['title']}\n\n{match['metadata']['text']}" for match in matches
    ]
    _show(
        f"Stage 2b — rerank on title + chunk text, top {settings.retrieval_top_n}:",
        _rerank(query, titled, settings.retrieval_top_n),
        matches,
        expected,
    )


def main() -> None:
    """Probe a single query from the command line."""
    parser = argparse.ArgumentParser(description="Probe both retrieval stages.")
    parser.add_argument("query")
    parser.add_argument("expected", help="Slug fragment that should be retrieved.")
    parser.add_argument("--candidates", type=int, default=None)
    args = parser.parse_args()

    probe(args.query, args.expected, args.candidates or settings.retrieval_candidates)


if __name__ == "__main__":
    main()
