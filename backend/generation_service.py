from backend.config import ACTIVE_LLM_PROVIDER, LLM_SPECS
from backend.llm_service import call_llm
from backend.models import GeneratedResponse, RetrievalResult


def deduplicate_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
    grouped: dict[str, RetrievalResult] = {}
    for result in results:
        key = result.reference
        if key not in grouped or result.rerank_score > grouped[key].rerank_score:
            grouped[key] = result
    return sorted(grouped.values(), key=lambda r: r.rerank_score, reverse=True)


def build_context_block(results: list[RetrievalResult]) -> str:
    blocks = []
    for index, item in enumerate(results, start=1):
        lines = [f"[SOURCE {index}] {item.reference} — {item.book}"]
        if item.verse_sanskrit.strip():
            lines.append(f"Sanskrit: {item.verse_sanskrit}")
        lines.append(f"Translation: {item.translation}")
        if item.purport.strip():
            lines.append(f"Purport: {item.purport}")
        blocks.append("\n".join(lines))
    return "\n\n---\n\n".join(blocks)


def build_prompt(query: str, context: str) -> str:
    return f"""The following passages are from Srila Prabhupada's books. Use ONLY these passages to answer the question.

===== SOURCE PASSAGES =====
{context}
===== END OF PASSAGES =====

Question: {query}

Answer (cite sources using [REFERENCE] format):"""


def generate(query: str, results: list[RetrievalResult]) -> GeneratedResponse:
    if not results:
        return GeneratedResponse(
            query=query,
            answer="No relevant passages were found in the provided texts for this question.",
            sources=[],
            llm_model=LLM_SPECS[ACTIVE_LLM_PROVIDER]["model"],
            context_used=[],
        )

    deduped = deduplicate_results(results)
    context = build_context_block(deduped)
    prompt = build_prompt(query, context)
    answer = call_llm(prompt)

    return GeneratedResponse(
        query=query,
        answer=answer,
        sources=deduped,
        llm_model=LLM_SPECS[ACTIVE_LLM_PROVIDER]["model"],
        context_used=[item.reference for item in deduped],
    )
