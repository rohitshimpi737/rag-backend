from typing import Literal

from pydantic import BaseModel, Field


BookCode = Literal["bg", "iso", "noi", "bs"]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    top_n: int = Field(5, ge=1, le=10, description="Number of final sources")
    k: int = Field(20, ge=1, le=100, description="Vector candidates before rerank")
    book_filter: list[BookCode] | None = Field(
        default=None,
        description="Optional book filter, e.g. ['bg', 'iso']",
    )


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    top_n: int = Field(5, ge=1, le=10, description="Number of final sources")
    k: int = Field(20, ge=1, le=100, description="Vector candidates before rerank")
    book_filter: list[BookCode] | None = Field(
        default=None,
        description="Optional book filter, e.g. ['bg', 'iso']",
    )


class SourceResult(BaseModel):
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


class AskResponse(BaseModel):
    query: str
    answer: str
    llm_model: str
    context_used: list[str]
    sources: list[SourceResult]


class RetrieveResponse(BaseModel):
    query: str
    source_count: int
    sources: list[SourceResult]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    active_embedding_provider: str
    active_llm_provider: str
    llm_model: str
    collection_size: int | None = None
    message: str | None = None
