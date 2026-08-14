"""Split documentation pages into chunks for retrieval."""

import json
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

DOCS = Path("data/docs.json")
OUTPUT = Path("data/chunks.json")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def main() -> None:
    """Split every documentation page into overlapping chunks."""
    setup_logging()

    docs = json.loads(DOCS.read_text(encoding="utf-8"))
    logger.info("Source pages: %d", len(docs))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict[str, Any]] = []
    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        for index, piece in enumerate(pieces):
            chunks.append(
                {
                    "id": f"{doc['url'].rstrip('/').split('/')[-1]}--{index}",
                    "text": piece,
                    "title": doc["title"],
                    "url": doc["url"],
                    "chunk_index": index,
                    "total_chunks": len(pieces),
                }
            )

    if not chunks:
        raise ValueError(f"No chunks produced from {DOCS}; is the file empty?")

    OUTPUT.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    sizes = [len(c["text"]) for c in chunks]
    per_page = [c["total_chunks"] for c in chunks]
    logger.info("Chunks: %d", len(chunks))
    logger.info("Average size: %d chars", sum(sizes) // len(sizes))
    logger.info("Size range: %d - %d chars", min(sizes), max(sizes))
    logger.info("Max chunks per page: %d", max(per_page))
    logger.info("Saved to %s", OUTPUT)

    sample = chunks[len(chunks) // 2]
    logger.debug("Sample chunk id: %s", sample["id"])
    logger.debug("Sample chunk title: %s", sample["title"])
    logger.debug("Sample chunk text: %.150s", sample["text"])

if __name__ == "__main__":
    main()
