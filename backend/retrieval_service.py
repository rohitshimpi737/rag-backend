from backend.config import RERANK_TOP_N, RETRIEVAL_K
from backend.embeddings import embed_query
from backend.models import RetrievalResult
from backend.reranker import rerank
from backend.vector_store import get_collection


def vector_search(
    query_vector: list[float],
    k: int = RETRIEVAL_K,
    book_filter: list[str] | None = None,
) -> list[dict]:
    collection = get_collection()

    where = None
    if book_filter and len(book_filter) == 1:
        where = {"book_code": {"$eq": book_filter[0]}}
    elif book_filter and len(book_filter) > 1:
        where = {"book_code": {"$in": book_filter}}

    query_kwargs = {
        "query_embeddings": [query_vector],
        "n_results": min(k, collection.count()),
        "include": ["metadatas", "documents", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    candidates = []
    for meta, doc, dist in zip(
        results["metadatas"][0],
        results["documents"][0],
        results["distances"][0],
    ):
        candidates.append(
            {
                "metadata": meta,
                "chunk_text": doc,
                "vector_score": round(1 - dist, 4),
            }
        )

    return candidates


def retrieve(
    query: str,
    top_n: int = RERANK_TOP_N,
    k: int = RETRIEVAL_K,
    book_filter: list[str] | None = None,
) -> list[RetrievalResult]:
    query_vector = embed_query(query)
    candidates = vector_search(query_vector, k=k, book_filter=book_filter)

    if not candidates:
        return []

    reranked = rerank(query, candidates, top_n=top_n)

    outputs = []
    for item in reranked:
        meta = item["metadata"]
        outputs.append(
            RetrievalResult(
                chunk_id=meta.get("chunk_id", ""),
                reference=meta["reference"],
                book=meta["book"],
                book_code=meta["book_code"],
                translation=meta["translation"],
                purport=meta["purport"],
                verse_sanskrit=meta.get("verse_sanskrit", ""),
                word_for_word=meta.get("word_for_word", ""),
                part=meta.get("part", 1),
                total_parts=meta.get("total_parts", 1),
                vector_score=item["vector_score"],
                rerank_score=item.get("rerank_score", item["vector_score"]),
                chunk_text=item["chunk_text"],
            )
        )

    return outputs
