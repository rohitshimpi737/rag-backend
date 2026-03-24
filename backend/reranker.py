from backend.config import RERANKER_MODEL


_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    reranker = get_reranker()
    pairs = [(query, c["chunk_text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = round(float(score), 4)

    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_n]
