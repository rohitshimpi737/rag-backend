from dataclasses import dataclass, field


@dataclass
class RetrievalResult:
    chunk_id: str
    reference: str
    book: str
    book_code: str
    translation: str
    purport: str
    verse_sanskrit: str
    word_for_word: str
    part: int
    total_parts: int
    vector_score: float
    rerank_score: float
    chunk_text: str


@dataclass
class GeneratedResponse:
    query: str
    answer: str
    sources: list[RetrievalResult]
    llm_model: str
    context_used: list[str] = field(default_factory=list)
