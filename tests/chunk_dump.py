"""Print the chunk text that the reranker actually receives.

The reranker sees isolated chunks, not pages. A page can be the right answer
while every one of its chunks reads as unrelated to the query, and that is
invisible from URLs and scores alone.
"""
import argparse
import textwrap

from src.rag.retriever import _embed_query, _vector_search
from src.settings import settings


def main() -> None:
    """Dump candidate chunks for a query, optionally filtered by URL."""
    parser = argparse.ArgumentParser(description="Inspect candidate chunk text.")
    parser.add_argument("query")
    parser.add_argument("--filter", default="", help="Only show URLs containing this.")
    parser.add_argument("--chars", type=int, default=600, help="Characters per chunk.")
    parser.add_argument("--candidates", type=int, default=None)
    args = parser.parse_args()

    candidates = args.candidates or settings.retrieval_candidates
    matches = _vector_search(_embed_query(args.query), candidates)

    for position, match in enumerate(matches, start=1):
        metadata = match["metadata"]
        url = str(metadata["url"])
        if args.filter and args.filter not in url:
            continue
        text = str(metadata["text"])
        print(f"\n{'=' * 70}")
        print(f"#{position}  score {match['score']:.4f}  {len(text)} chars")
        print(f"title: {metadata['title']}")
        print(f"url:   {url}")
        print(f"{'-' * 70}")
        print(textwrap.fill(text[: args.chars], width=70))
        if len(text) > args.chars:
            print(f"... (+{len(text) - args.chars} chars)")


if __name__ == "__main__":
    main()
