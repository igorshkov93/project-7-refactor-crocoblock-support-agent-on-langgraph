"""Embed documentation chunks with Cohere and upload them to Pinecone."""

import json
import time
from pathlib import Path
from typing import Any

import cohere
from pinecone import Pinecone, ServerlessSpec

from src.logging_config import get_logger, setup_logging
from src.settings import settings

logger = get_logger(__name__)

CHUNKS = Path("data/chunks.json")
EMBED_MODEL = "embed-v4.0"
DIMENSION = 1536
BATCH_SIZE = 90
MAX_ATTEMPTS = 4


def main() -> None:
    """Embed every chunk and upsert it into the vector index."""
    setup_logging()

    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    logger.info("Chunks to index: %d", len(chunks))

    pc = Pinecone(
        api_key=settings.pinecone_api_key.get_secret_value()
        if settings.pinecone_api_key
        else None
    )

    existing = [i["name"] for i in pc.list_indexes()]
    if settings.pinecone_index not in existing:
        logger.info("Creating index '%s'", settings.pinecone_index)
        pc.create_index(
            name=settings.pinecone_index,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        time.sleep(10)

    index = pc.Index(settings.pinecone_index)
    co = cohere.ClientV2(
        api_key=settings.cohere_api_key.get_secret_value()
        if settings.cohere_api_key
        else None
    )

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        vectors: list[list[float]] | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = co.embed(
                    texts=texts,
                    model=EMBED_MODEL,
                    input_type="search_document",
                    embedding_types=["float"],
                )
                vectors = response.embeddings.float_
                break
            except (
                cohere.TooManyRequestsError,
                cohere.ServiceUnavailableError,
                cohere.GatewayTimeoutError,
                cohere.InternalServerError,
            ):
                if attempt == MAX_ATTEMPTS - 1:
                    raise
                wait = 20 * (attempt + 1)
                logger.warning("Rate limited, retrying in %ds", wait)
                time.sleep(wait)

        if not vectors:
            raise RuntimeError(f"Cohere returned no embeddings for batch at offset {start}")

        payload: list[dict[str, Any]] = [
            {
                "id": chunk["id"],
                "values": vector,
                "metadata": {
                    "text": chunk["text"],
                    "title": chunk["title"],
                    "url": chunk["url"],
                },
            }
            for chunk, vector in zip(batch, vectors, strict=True)
        ]

        index.upsert(vectors=payload, namespace=settings.pinecone_namespace)
        done = min(start + BATCH_SIZE, len(chunks))
        logger.info("Uploaded %d/%d", done, len(chunks))
        time.sleep(7)

    time.sleep(5)
    logger.info("Index stats: %s", index.describe_index_stats())


if __name__ == "__main__":
    main()
