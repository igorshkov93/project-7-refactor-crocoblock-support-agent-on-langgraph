"""Two-stage retrieval: stage isolation, error mapping, and result assembly.

The three stages are patched by name rather than mocking the Pinecone and
Cohere SDKs. The clients are built lazily behind ``lru_cache``, so nothing is
constructed as long as the stage functions never run.

The assembly step deserves the attention it gets here: rerank returns indices
into the candidate list, and ``search`` uses them to look metadata back up. An
off-by-one there would pair one chunk's text with another chunk's URL and fail
silently, producing an answer that cites the wrong page.
"""
from __future__ import annotations

import pytest

from src.exceptions import EmbeddingError, RerankError, VectorStoreError
from src.rag import retriever
from src.rag.retriever import search


def make_match(index: int, score: float = 0.5) -> dict:
    """A Pinecone match in the shape ``search`` expects."""
    return {
        "id": f"chunk-{index}",
        "score": score,
        "metadata": {
            "text": f"body of chunk {index}",
            "title": f"Page {index}",
            "url": f"https://crocoblock.com/kb/page-{index}/",
        },
    }


@pytest.fixture
def stages(monkeypatch):
    """Replace all three stages; return a log of what each received."""

    def install(matches=None, ranked=None, embed_raises=None,
                search_raises=None, rerank_raises=None):
        calls = {"embed": [], "search": [], "rerank": []}

        def fake_embed(query):  # noqa: ARG001
            calls["embed"].append(query)
            if embed_raises is not None:
                raise embed_raises
            return [0.1, 0.2, 0.3]

        def fake_search(embedding, candidates):
            calls["search"].append((embedding, candidates))
            if search_raises is not None:
                raise search_raises
            return list(matches or [])

        def fake_rerank(query, documents, top_n):
            calls["rerank"].append((query, documents, top_n))
            if rerank_raises is not None:
                raise rerank_raises
            return list(ranked or [])

        monkeypatch.setattr(retriever, "_embed_query", fake_embed)
        monkeypatch.setattr(retriever, "_vector_search", fake_search)
        monkeypatch.setattr(retriever, "_rerank", fake_rerank)
        return calls

    return install


def test_results_carry_text_title_url_and_both_scores(stages):
    """Every field the docs_qa prompt and the trace depend on."""
    stages(matches=[make_match(0, score=0.77)], ranked=[(0, 0.93)])

    results = search("how do I save records?")

    assert results == [
        {
            "text": "body of chunk 0",
            "title": "Page 0",
            "url": "https://crocoblock.com/kb/page-0/",
            "vector_score": 0.77,
            "rerank_score": 0.93,
        }
    ]


def test_rerank_indices_select_the_right_candidates(stages):
    """Rerank reorders and drops; metadata must follow its indices, not position."""
    matches = [make_match(i) for i in range(5)]
    stages(matches=matches, ranked=[(3, 0.9), (0, 0.7)])

    results = search("query")

    assert [hit["title"] for hit in results] == ["Page 3", "Page 0"]
    assert results[0]["url"].endswith("page-3/")
    assert results[0]["text"] == "body of chunk 3"


def test_rerank_order_wins_over_vector_order(stages):
    """Stage two exists to overrule stage one."""
    matches = [make_match(0, score=0.99), make_match(1, score=0.10)]
    stages(matches=matches, ranked=[(1, 0.95), (0, 0.20)])

    results = search("query")

    assert results[0]["title"] == "Page 1"
    assert results[0]["vector_score"] == 0.10


def test_empty_index_returns_nothing_without_reranking(stages):
    """No candidates means no rerank call — and no wasted Cohere quota."""
    calls = stages(matches=[])

    assert search("nothing matches this") == []
    assert calls["rerank"] == []

def test_defaults_come_from_settings(stages):
    """Candidate width and cut-off are configuration, not literals in the code."""
    calls = stages(matches=[make_match(0)], ranked=[(0, 0.9)])

    search("query")

    assert calls["search"][0][1] == retriever.settings.retrieval_candidates
    assert calls["rerank"][0][2] == retriever.settings.retrieval_top_n


def test_explicit_arguments_override_the_defaults(stages):
    """The evaluation scripts sweep these; they must actually reach the stages."""
    calls = stages(matches=[make_match(0)], ranked=[(0, 0.9)])

    search("query", top_k=2, candidates=7)

    assert calls["search"][0][1] == 7
    assert calls["rerank"][0][2] == 2


def test_rerank_scores_the_candidate_texts(stages):
    """Rerank must see chunk bodies, not ids or titles."""
    calls = stages(matches=[make_match(0), make_match(1)], ranked=[(0, 0.9)])

    search("query")

    _, documents, _ = calls["rerank"][0]
    assert documents == ["body of chunk 0", "body of chunk 1"]


def test_the_embedding_reaches_the_vector_search(stages):
    """The vector handed to Pinecone is the one Cohere produced."""
    calls = stages(matches=[make_match(0)], ranked=[(0, 0.9)])

    search("query")

    assert calls["search"][0][0] == [0.1, 0.2, 0.3]

@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"embed_raises": EmbeddingError("cohere embed failed")}, EmbeddingError),
        ({"search_raises": VectorStoreError("pinecone failed")}, VectorStoreError),
        ({"rerank_raises": RerankError("cohere rerank failed")}, RerankError),
    ],
)
def test_each_stage_fails_with_its_own_exception(stages, kwargs, expected):
    """The service name in the exception is what makes a trace readable."""
    stages(matches=[make_match(0)], ranked=[(0, 0.9)], **kwargs)

    with pytest.raises(expected):
        search("query")


def test_a_failing_stage_stops_the_pipeline(stages):
    """No partial results: a broken embed must not reach the vector store."""
    calls = stages(embed_raises=EmbeddingError("down"))

    with pytest.raises(EmbeddingError):
        search("query")

    assert calls["search"] == []
    assert calls["rerank"] == []
