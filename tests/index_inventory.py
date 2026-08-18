"""List the documentation pages present in the Pinecone index.

A retrieval miss has two possible causes: the search ranked the right page
below the cut-off, or the page is not indexed at all. Only an inventory of the
index tells them apart, and the fix differs completely between the two.
"""
import argparse
from collections import Counter
from typing import Any

from src.rag.retriever import _pinecone_index
from src.settings import settings


def collect_urls(limit: int) -> Counter[str]:
    """Return indexed source URLs with the number of chunks each contributed."""
    index = _pinecone_index()
    urls: Counter[str] = Counter()
    for ids in index.list(namespace=settings.pinecone_namespace, limit=limit):
        fetched: Any = index.fetch(ids=list(ids), namespace=settings.pinecone_namespace)
        for vector in fetched.vectors.values():
            metadata = vector.metadata or {}
            url = str(metadata.get("url", "<no url>"))
            urls[url] += 1
    return urls


def main() -> None:
    """Print every indexed URL, optionally filtered by a substring."""
    parser = argparse.ArgumentParser(description="Inspect the Pinecone index.")
    parser.add_argument("--filter", default="", help="Only show URLs containing this.")
    parser.add_argument("--limit", type=int, default=100, help="IDs fetched per page.")
    args = parser.parse_args()

    urls = collect_urls(args.limit)
    matched = {url: n for url, n in urls.items() if args.filter in url}

    print(f"Namespace '{settings.pinecone_namespace}': {len(urls)} unique pages, ")
    print(f"{sum(urls.values())} chunks total")
    if args.filter:
        print(f"Filter '{args.filter}': {len(matched)} pages\n")

    for url, count in sorted(matched.items()):
        print(f"  {count:3d}  {url}")


if __name__ == "__main__":
    main()
