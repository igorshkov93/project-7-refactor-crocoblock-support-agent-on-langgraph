"""Embed documentation chunks with Cohere and upload them to Pinecone."""
import json
import time
from pathlib import Path

import cohere
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from src.settings import settings

load_dotenv()

CHUNKS = Path("data/chunks.json")

INDEX_NAME = "jetformbuilder-docs"
NAMESPACE = "jfb"
EMBED_MODEL = "embed-v4.0"
DIMENSION = 1536
BATCH_SIZE = 90


def main():
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    print(f"Chunks to index: {len(chunks)}")

    pc = Pinecone(
    api_key=settings.pinecone_api_key.get_secret_value()
    if settings.pinecone_api_key
    else None
)

    existing = [i["name"] for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        time.sleep(10)

    index = pc.Index(INDEX_NAME)
    co = cohere.ClientV2(
    api_key=settings.cohere_api_key.get_secret_value()
    if settings.cohere_api_key
    else None
)

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        for attempt in range(4):
            try:
                response = co.embed(
                    texts=texts,
                    model=EMBED_MODEL,
                    input_type="search_document",
                    embedding_types=["float"],
                )
                vectors = response.embeddings.float
                break
            except Exception as error:
                if attempt == 3:
                    raise
                wait = 20 * (attempt + 1)
                print(f"    retry in {wait}s: {type(error).__name__}")
                time.sleep(wait)

        payload = [
            {
                "id": chunk["id"],
                "values": vector,
                "metadata": {
                    "text": chunk["text"],
                    "title": chunk["title"],
                    "url": chunk["url"],
                },
            }
            for chunk, vector in zip(batch, vectors)
        ]

        index.upsert(vectors=payload, namespace=NAMESPACE)
        done = min(start + BATCH_SIZE, len(chunks))
        print(f"  {done}/{len(chunks)} uploaded")
        time.sleep(7)


    time.sleep(5)
    stats = index.describe_index_stats()
    print(f"\nIndex stats: {stats}")


if __name__ == "__main__":
    main()
