import os
import time

from backend.config import ACTIVE_EMBEDDING_PROVIDER, BGE_QUERY_PREFIX


RETRY_DELAY = 10
_baai_model = None


def _get_baai_model():
    global _baai_model
    if _baai_model is None:
        from sentence_transformers import SentenceTransformer

        _baai_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    return _baai_model


def _embed_baai(texts: list[str]) -> list[list[float]]:
    model = _get_baai_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def _embed_gemini(texts: list[str]) -> list[list[float]]:
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set.")

    genai.configure(api_key=api_key)

    for attempt in range(3):
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=texts,
                task_type="retrieval_query",
            )
            vectors = result["embedding"] if len(texts) > 1 else [result["embedding"]]
            return vectors
        except Exception as exc:
            if ("quota" in str(exc).lower() or "429" in str(exc)) and attempt < 2:
                time.sleep(RETRY_DELAY)
            else:
                raise

    raise RuntimeError("Gemini embedding failed after retries.")


def _embed_openai(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set.")

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
    )
    return [item.embedding for item in response.data]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if ACTIVE_EMBEDDING_PROVIDER == "baai":
        return _embed_baai(texts)
    if ACTIVE_EMBEDDING_PROVIDER == "gemini":
        return _embed_gemini(texts)
    if ACTIVE_EMBEDDING_PROVIDER == "openai":
        return _embed_openai(texts)
    raise ValueError(f"Unsupported embedding provider: {ACTIVE_EMBEDDING_PROVIDER}")


def embed_query(query: str) -> list[float]:
    if ACTIVE_EMBEDDING_PROVIDER == "baai":
        query = BGE_QUERY_PREFIX + query
    return embed_texts([query])[0]
