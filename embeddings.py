"""Embedding utilities built on ``BAAI/bge-large-en-v1.5`` (framework-free).

Design notes (these materially affect retrieval quality):

* The model is loaded once per process via a thread-safe module singleton.
* Device auto-selects CUDA when available, else CPU.
* Embeddings are always L2-normalized (``normalize_embeddings=True``) to pair
  correctly with the cosine-distance Chroma collection.
* Queries — and only queries — get the BGE instruction prefix.
"""

from __future__ import annotations

import logging
import threading

from src.config import EMBEDDING_MODEL, QUERY_INSTRUCTION

log = logging.getLogger("talentsphere.embeddings")

_BATCH_SIZE = 32

_lock = threading.Lock()
_model = None


def _auto_device() -> str:
    """Return ``"cuda"`` when a GPU is available, otherwise ``"cpu"``."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - torch missing/broken -> safe CPU fallback
        return "cpu"


def get_model():
    """Load and cache the sentence-transformer model (once per process).

    The first call downloads ~1.3GB; subsequent calls return the cached model.
    Thread-safe: concurrent Flask requests share one instance.
    """
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                log.info("Loading embedding model %s …", EMBEDDING_MODEL)
                _model = SentenceTransformer(EMBEDDING_MODEL, device=_auto_device())
                log.info("Embedding model ready.")
    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed document chunks (no query prefix) as normalized 1024-dim vectors."""
    if not texts:
        return []
    model = get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=_BATCH_SIZE,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


def embed_query(text: str) -> list[float]:
    """Embed a single search query with the required BGE instruction prefix."""
    model = get_model()
    vector = model.encode(
        QUERY_INSTRUCTION + text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector.tolist()
