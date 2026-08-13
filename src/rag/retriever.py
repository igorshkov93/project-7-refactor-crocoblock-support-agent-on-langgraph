"""Two-stage retrieval: vector search followed by Cohere reranking."""
import cohere
from pinecone import Pinecone

from src.rag.index_chunks import EMBED_MODEL
from src.settings import settings

RERANK_MODEL = "rerank-v3.5"

_pc = Pinecone(
    api_key=settings.pinecone_api_key.get_secret_value()
    if settings.pinecone_api_key
    else None
)

_index = _pc.Index(settings.pinecone_index)

_co = cohere.ClientV2(
    api_key=settings.cohere_api_key.get_secret_value()
    if settings.cohere_api_key
    else None
)


def search(
    query: str,
    top_k: int | None = None,
    candidates: int | None = None,
) -> list[dict[str, object]]:

    """Retrieve documentation chunks relevant to a query.

    Stage 1: dense vector search returns a broad candidate set.
    Stage 2: the reranker scores query-document pairs and keeps the best.
    """
    top_k = settings.retrieval_top_n if top_k is None else top_k
    candidates = settings.retrieval_candidates if candidates is None else candidates

    response = _co.embed(
        texts=[query],
        model=EMBED_MODEL,
        input_type="search_query",
        embedding_types=["float"],
    )
    if not response.embeddings.float_:
        raise RuntimeError("Cohere returned no embeddings for the query")
    embedding = response.embeddings.float_[0]

    matches = _index.query(
        vector=embedding,
        top_k=candidates,
        namespace=settings.pinecone_namespace,
        include_metadata=True,
    )["matches"]

    if not matches:
        return []

    documents = [m["metadata"]["text"] for m in matches]
    reranked = _co.rerank(
        query=query,
        documents=documents,
        model=RERANK_MODEL,
        top_n=top_k,
    )

    results = []
    for item in reranked.results:
        match = matches[item.index]
        results.append(
            {
                "text": match["metadata"]["text"],
                "title": match["metadata"]["title"],
                "url": match["metadata"]["url"],
                "vector_score": match["score"],
                "rerank_score": item.relevance_score,
            }
        )

    return results
