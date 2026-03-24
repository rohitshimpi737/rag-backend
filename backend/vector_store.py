import chromadb
from chromadb.config import Settings

from backend.config import CHROMA_COLLECTION, CHROMA_DIR


_collection = None


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
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
