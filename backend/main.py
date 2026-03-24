from contextlib import asynccontextmanager
from dataclasses import asdict
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    RetrieveRequest,
    RetrieveResponse,
    SourceResult,
)
from backend.config import (
    ACTIVE_EMBEDDING_PROVIDER,
    ACTIVE_LLM_PROVIDER,
    LLM_SPECS,
    PRELOAD_RERANKER_ON_STARTUP,
)
from backend.generation_service import generate
from backend.reranker import get_reranker
from backend.retrieval_service import retrieve
from backend.vector_store import get_collection

logger = logging.getLogger("prabhupada_api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Starting PrabhupadaGPT FastAPI backend...")

    if PRELOAD_RERANKER_ON_STARTUP:
        try:
            get_reranker()
        except Exception as exc:
            logger.warning(f"Reranker preload skipped: {exc}")
    else:
        logger.info("Reranker preload disabled at startup.")

    yield

    logger.info("Shutting down PrabhupadaGPT FastAPI backend...")


app = FastAPI(
    title="PrabhupadaGPT Backend API",
    version="1.0.0",
    description=(
        "FastAPI backend for PrabhupadaGPT RAG system. "
        "Uses the same retrieval and generation pipeline as the current app."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_source_result(result_obj) -> SourceResult:
    return SourceResult(**asdict(result_obj))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        collection = get_collection()
        collection_size = collection.count()
        return HealthResponse(
            status="ok",
            active_embedding_provider=ACTIVE_EMBEDDING_PROVIDER,
            active_llm_provider=ACTIVE_LLM_PROVIDER,
            llm_model=LLM_SPECS[ACTIVE_LLM_PROVIDER]["model"],
            collection_size=collection_size,
        )
    except Exception as exc:
        return HealthResponse(
            status="degraded",
            active_embedding_provider=ACTIVE_EMBEDDING_PROVIDER,
            active_llm_provider=ACTIVE_LLM_PROVIDER,
            llm_model=LLM_SPECS[ACTIVE_LLM_PROVIDER]["model"],
            message=str(exc),
        )


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_sources(payload: RetrieveRequest) -> RetrieveResponse:
    try:
        results = retrieve(
            query=payload.query.strip(),
            top_n=payload.top_n,
            k=payload.k,
            book_filter=payload.book_filter,
        )
        sources = [_to_source_result(r) for r in results]
        return RetrieveResponse(
            query=payload.query,
            source_count=len(sources),
            sources=sources,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    try:
        results = retrieve(
            query=payload.query.strip(),
            top_n=payload.top_n,
            k=payload.k,
            book_filter=payload.book_filter,
        )
        generated = generate(query=payload.query.strip(), results=results)
        return AskResponse(
            query=generated.query,
            answer=generated.answer,
            llm_model=generated.llm_model,
            context_used=generated.context_used,
            sources=[_to_source_result(r) for r in generated.sources],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc
