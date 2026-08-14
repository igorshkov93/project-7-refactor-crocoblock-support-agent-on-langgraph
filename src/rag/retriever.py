"""Two-stage retrieval: dense vector search followed by Cohere reranking.

Each stage talks to a different service and fails differently, so each is
wrapped separately: an embedding failure, a vector-store failure and a rerank
failure are distinct exceptions carrying the service name. Clients are built
lazily on first use — constructing them at import time turned a missing API key
into an import error pointing at the wrong place.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cohere
from pinecone import Pinecone

from src.exceptions import (
    ConfigurationError,
    EmbeddingError,
    RerankError,
    VectorStoreError,
)
from src.logging_config import get_logger
from src.retry import with_retry
from src.settings import settings

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _cohere_client() -> cohere.ClientV2:
    """Build the Cohere client once, with a request timeout applied."""
    if settings.cohere_api_key is None:
        raise ConfigurationError("COHERE_API_KEY is not set in .env")
    return cohere.ClientV2(
        api_key=settings.cohere_api_key.get_secret_value(),
        timeout=settings.cohere_timeout_seconds,
    )


@lru_cache(maxsize=1)
def _pinecone_index() -> Any:
    """Build the Pinecone index handle once.

    Returns ``Any``: the concrete class lives in ``pinecone.db_data.index``,
    which the SDK does not export publicly and has already moved between
    versions. The narrow surface we use is typed in :func:`_vector_search`.
    """
    if settings.pinecone_api_key is None:
        raise ConfigurationError("PINECONE_API_KEY is not set in .env")
    client = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
    return client.Index(settings.pinecone_index)


@with_retry()
def _embed_query(query: str) -> list[float]:
    """Embed the query with Cohere.

    Raises:
        EmbeddingError: If the call fails or returns no usable vector.
    """
    try:
        response = _cohere_client().embed(
            texts=[query],
            model=settings.embed_model,
            input_type="search_query",
            embedding_types=["float"],
        )
    except Exception as error:
        raise EmbeddingError(f"Failed to embed the query: {error}") from error

    vectors = response.embeddings.float_
    if not vectors:
        raise EmbeddingError("Cohere returned no embeddings for the query")
    return list(vectors[0])


@with_retry()
def _vector_search(embedding: list[float], candidates: int) -> list[dict[str, Any]]:
    """Run dense search against Pinecone.

    Raises:
        VectorStoreError: If the query fails or the payload lacks matches.
    """
    try:
        response = _pinecone_index().query(
            vector=embedding,
            top_k=candidates,
            namespace=settings.pinecone_namespace,
            include_metadata=True,
            _request_timeout=settings.pinecone_timeout_seconds,
        )
    except Exception as error:
        raise VectorStoreError(f"Pinecone query failed: {error}") from error

    matches = response.get("matches")
    if matches is None:
        raise VectorStoreError("Pinecone response contains no 'matches' field")
    return list(matches)


@with_retry()
def _rerank(query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
    """Reorder candidates with Cohere rerank.

    Returns:
        Pairs of ``(index into documents, relevance score)``, best first.

    Raises:
        RerankError: If the call fails.
    """
    try:
        response = _cohere_client().rerank(
            query=query,
            documents=documents,
            model=settings.rerank_model,
            top_n=top_n,
        )
    except Exception as error:
        raise RerankError(f"Cohere rerank failed: {error}") from error

    return [(item.index, item.relevance_score) for item in response.results]


def search(
    query: str,
    top_k: int | None = None,
    candidates: int | None = None,
) -> list[dict[str, Any]]:
    """Retrieve documentation chunks relevant to a query.

    Stage 1: dense vector search returns a broad candidate set.
    Stage 2: the reranker scores query-document pairs and keeps the best.

    Args:
        query: The user's question.
        top_k: How many chunks to return. Defaults to ``RETRIEVAL_TOP_N``.
        candidates: How many chunks to retrieve before reranking. Defaults to
            ``RETRIEVAL_CANDIDATES``.

    Returns:
        Chunks ordered by relevance, each with its text, title, source URL and
        both scores. Empty if the index has nothing for this query.

    Raises:
        EmbeddingError: If the query could not be embedded.
        VectorStoreError: If the vector search failed.
        RerankError: If reranking failed.
    """
    top_k = settings.retrieval_top_n if top_k is None else top_k
    candidates = settings.retrieval_candidates if candidates is None else candidates

    embedding = _embed_query(query)
    matches = _vector_search(embedding, candidates)

    if not matches:
        logger.info(
            "Vector search returned no candidates",
            extra={"namespace": settings.pinecone_namespace},
        )
        return []

    documents = [match["metadata"]["text"] for match in matches]
    ranked = _rerank(query, documents, top_k)

    results: list[dict[str, Any]] = []
    for position, score in ranked:
        match = matches[position]
        metadata = match["metadata"]
        results.append(
            {
                "text": metadata["text"],
                "title": metadata["title"],
                "url": metadata["url"],
                "vector_score": match["score"],
                "rerank_score": score,
            }
        )

    logger.info(
        "Retrieval complete",
        extra={
            "candidates": len(matches),
            "returned": len(results),
            "top_score": results[0]["rerank_score"] if results else None,
        },
    )
    return results
