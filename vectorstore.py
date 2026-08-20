"""ChromaDB persistent vector store with accuracy-focused retrieval.

Beyond plain cosine top-K, :func:`search` implements a retrieval pipeline that
measurably improves answer grounding:

1. **Over-fetch**: pull ``top_k * 4`` candidates from the ANN index so the
   reranker has real choices.
2. **Hybrid rerank**: final score = 0.72 x cosine + 0.28 x lexical overlap
   (query-term hit-rate with position bonus). Catches exact keyword matches
   (menu names, codes, acronyms) that pure dense retrieval under-ranks.
3. **Relevance floor**: candidates below a minimum score are dropped rather
   than padded in, so the LLM never sees noise passages.
4. **Source diversity**: at most 3 chunks per document in the final set,
   preventing one long PDF from crowding out the correct source.
5. **Neighbor stitching**: each winning chunk is expanded with its adjacent
   chunks (same document, chunk_index +/- 1) so the LLM receives complete
   thoughts instead of fragments cut mid-sentence.
"""

from __future__ import annotations

import logging
import re
import threading

from src.config import CHROMA_COLLECTION, CHROMA_DB_PATH

log = logging.getLogger("talentsphere.vectorstore")

_lock = threading.Lock()
_client = None

# Retrieval tuning
_OVERFETCH = 4          # candidates fetched = top_k * _OVERFETCH (capped)
_MAX_CANDIDATES = 48
_MIN_SCORE = 0.28       # relevance floor on the hybrid score
_PER_SOURCE_CAP = 3     # diversity: max chunks from one document
_DENSE_WEIGHT = 0.72
_LEXICAL_WEIGHT = 0.28

_WORD_RE = re.compile(r"[a-z0-9]{3,}")
_STOPWORDS = frozenset(
    "the and for with that this from are was were what when where which whose "
    "how why can could should would will shall may might must been being have "
    "has had does did doing about into over under between your our their its "
    "than then them they there here also only just very much more most some "
    "any all each other such not but nor own same too out off again further "
    "explain describe give list state define".split()
)


def get_client():
    """Return a cached, disk-persistent Chroma client (thread-safe singleton)."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                import chromadb

                _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _client


def get_collection():
    """Return (creating if needed) the cosine-space document collection."""
    return get_client().get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def ingested_hashes() -> set[str]:
    """Return the set of file hashes already present in the index."""
    try:
        result = get_collection().get(include=["metadatas"])
    except Exception:  # noqa: BLE001 - empty/new collection
        return set()
    hashes: set[str] = set()
    for meta in result.get("metadatas") or []:
        file_hash = (meta or {}).get("file_hash")
        if file_hash:
            hashes.add(file_hash)
    return hashes


def add_chunks(chunks: list[dict], embeddings: list[list[float]], file_hash: str) -> int:
    """Upsert chunks and their embeddings; returns the number added."""
    if not chunks:
        return 0
    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = []
    for chunk in chunks:
        meta = dict(chunk["metadata"])
        meta["file_hash"] = file_hash
        metadatas.append(meta)
    get_collection().upsert(
        ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
    )
    return len(ids)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _query_terms(query_text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(query_text.lower()) if w not in _STOPWORDS]


def _lexical_score(terms: list[str], text: str) -> float:
    """Fraction of query terms present in the chunk, with a small phrase bonus."""
    if not terms:
        return 0.0
    low = text.lower()
    hits = sum(1 for t in terms if t in low)
    score = hits / len(terms)
    # Bonus when consecutive query terms appear near each other (phrase-ish).
    if hits >= 2 and len(terms) >= 2:
        joined = " ".join(terms[:4])
        if joined[: max(8, len(joined) // 2)] in low:
            score = min(1.0, score + 0.15)
    return score


def _stitch_neighbors(collection, hit: dict) -> str:
    """Expand a chunk with its adjacent chunks from the same document."""
    source = hit.get("source")
    index = hit.get("chunk_index")
    if source is None or index is None:
        return hit["text"]
    wanted_ids = []
    for neighbor in (index - 1, index + 1):
        if neighbor >= 0:
            # Chunk ids follow "{source}::p{page}::c{index}" but page varies, so
            # look neighbors up by metadata instead of reconstructing the id.
            wanted_ids.append(neighbor)
    try:
        result = collection.get(
            where={"$and": [{"source": {"$eq": source}},
                            {"chunk_index": {"$in": wanted_ids}}]},
            include=["documents", "metadatas"],
        )
    except Exception:  # noqa: BLE001 - stitching is best-effort
        return hit["text"]

    before, after = "", ""
    for text, meta in zip(result.get("documents") or [], result.get("metadatas") or []):
        idx = (meta or {}).get("chunk_index")
        if idx == index - 1:
            before = text
        elif idx == index + 1:
            after = text

    def _tail(text: str, chars: int = 300) -> str:
        return text[-chars:] if text else ""

    def _head(text: str, chars: int = 300) -> str:
        return text[:chars] if text else ""

    parts = []
    if before:
        parts.append("…" + _tail(before))
    parts.append(hit["text"])
    if after:
        parts.append(_head(after) + "…")
    return "\n".join(parts)


def search(
    query_embedding: list[float],
    top_k: int,
    query_text: str = "",
    stitch: bool = True,
) -> list[dict]:
    """Accuracy-focused retrieval (over-fetch → hybrid rerank → diversify → stitch).

    Args:
        query_embedding: The query's embedding vector.
        top_k: Number of results to return.
        query_text: Raw query text; enables the lexical half of the hybrid score.
        stitch: Expand winning chunks with their neighbors for fuller context.

    Returns:
        ``[{text, source, page, chunk_index, score}]`` sorted by descending score.
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    n_candidates = min(max(top_k * _OVERFETCH, top_k), _MAX_CANDIDATES, collection.count())
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_candidates,
        include=["documents", "metadatas", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    terms = _query_terms(query_text)

    candidates: list[dict] = []
    for text, meta, distance in zip(documents, metadatas, distances):
        meta = meta or {}
        dense = 1 - distance
        lexical = _lexical_score(terms, text or "")
        hybrid = _DENSE_WEIGHT * dense + _LEXICAL_WEIGHT * lexical
        candidates.append(
            {
                "text": text,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", "—"),
                "chunk_index": meta.get("chunk_index", 0),
                "score": round(hybrid, 4),
                "dense": round(dense, 4),
            }
        )

    candidates.sort(key=lambda c: c["score"], reverse=True)

    # Relevance floor — but never return nothing when the index has plausible
    # matches: fall back to the best dense hits if the floor filters all.
    strong = [c for c in candidates if c["score"] >= _MIN_SCORE]
    pool = strong or candidates[: top_k]

    # Source diversity cap.
    picked: list[dict] = []
    per_source: dict[str, int] = {}
    for cand in pool:
        if per_source.get(cand["source"], 0) >= _PER_SOURCE_CAP:
            continue
        picked.append(cand)
        per_source[cand["source"]] = per_source.get(cand["source"], 0) + 1
        if len(picked) >= top_k:
            break

    if stitch:
        for hit in picked:
            hit["text"] = _stitch_neighbors(collection, hit)

    for hit in picked:
        hit.pop("dense", None)
    return picked


# ---------------------------------------------------------------------------
# Stats / maintenance
# ---------------------------------------------------------------------------


def stats() -> dict:
    """Return index stats: total chunks and distinct source filenames."""
    collection = get_collection()
    total = collection.count()
    sources: set[str] = set()
    if total:
        try:
            result = collection.get(include=["metadatas"])
            for meta in result.get("metadatas") or []:
                source = (meta or {}).get("source")
                if source:
                    sources.add(source)
        except Exception:  # noqa: BLE001 - best-effort stats
            pass
    return {"total_chunks": total, "sources": len(sources), "source_names": sorted(sources)}


def reset_collection() -> None:
    """Delete and recreate the collection (clears the whole index)."""
    client = get_client()
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:  # noqa: BLE001 - already absent
        pass
    client.get_or_create_collection(
        name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
