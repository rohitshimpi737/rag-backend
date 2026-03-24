import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_DIR = str(PROJECT_ROOT / "data" / "chromadb")
CHROMA_COLLECTION = "prabhupada_rag"

ACTIVE_EMBEDDING_PROVIDER = os.getenv("ACTIVE_EMBEDDING_PROVIDER", "baai").strip().lower()
ACTIVE_LLM_PROVIDER = os.getenv("ACTIVE_LLM_PROVIDER", "groq").strip().lower()

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "20"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))


def _get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


PRELOAD_RERANKER_ON_STARTUP = _get_bool_env("PRELOAD_RERANKER_ON_STARTUP", False)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

LLM_SPECS = {
    "gemini": {
        "model": "gemini-2.0-flash",
        "key_env": "GEMINI_API_KEY",
    },
    "openai": {
        "model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
    },
    "anthropic": {
        "model": "claude-haiku-4-5-20251001",
        "key_env": "ANTHROPIC_API_KEY",
    },
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
    },
}
