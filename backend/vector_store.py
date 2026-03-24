import logging
import chromadb
from chromadb.config import Settings

from backend.config import CHROMA_COLLECTION, CHROMA_DIR

logger = logging.getLogger("vector_store")

_collection = None


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        try:
            client = chromadb.PersistentClient(
                path=CHROMA_DIR,
                settings=Settings(anonymized_telemetry=False),
            )
        except Exception as exc:
            # Suppress telemetry errors during client init; they don't block functionality
            if "capture()" in str(exc) or "telemetry" in str(exc).lower():
                logger.warning(f"Telemetry init error (non-blocking): {exc}")
                # Retry without catching
                client = chromadb.PersistentClient(
                    path=CHROMA_DIR,
                    settings=Settings(anonymized_telemetry=False, allow_reset=True),
                )
            else:
                raise
        
        try:
            _collection = client.get_collection(name=CHROMA_COLLECTION)
        except KeyError as exc:
            if str(exc) == "'_type'":
                raise RuntimeError(
                    "Incompatible ChromaDB collection format detected in data/chromadb. "
                    "Rebuild vectors with the current environment (e.g., run legacy/embed.py --reset)."
                ) from exc
            raise
    return _collection
