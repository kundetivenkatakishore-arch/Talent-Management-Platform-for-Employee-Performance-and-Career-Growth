"""PDF ingestion: text extraction, whitespace cleaning, and chunking.

This module is pure data logic — no Streamlit calls — so it can be reused and
unit-tested independently. It is deliberately robust to messy PDFs: per-page
extraction failures are logged and skipped rather than aborting the whole file.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    """Collapse runs of whitespace into single spaces and strip the ends."""
    return _WHITESPACE_RE.sub(" ", text or "").strip()


def file_hash(data: bytes) -> str:
    """Return the sha256 hex digest of raw file bytes (used for de-dup)."""
    return hashlib.sha256(data).hexdigest()


def extract_pages(file: Any) -> list[dict]:
    """Extract per-page text from a PDF.

    Args:
        file: A path string or a file-like object (e.g. a Streamlit upload).

    Returns:
        A list of ``{"page": int, "text": str}`` dicts, 1-indexed by page,
        with empty/blank pages skipped.
    """
    pages: list[dict] = []
    try:
        reader = PdfReader(file)
    except Exception as exc:  # noqa: BLE001 - surface a friendly message upstream
        logger.error("Failed to open PDF: %s", exc)
        raise

    for index, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - one bad page shouldn't kill the file
            logger.warning("Skipping page %d (extraction error): %s", index, exc)
            continue

        text = _clean(raw)
        if not text:
            logger.info("Skipping page %d (no extractable text)", index)
            continue

        pages.append({"page": index, "text": text})

    return pages


def chunk_pages(pages: list[dict], source_name: str) -> list[dict]:
    """Split page texts into overlapping chunks with source metadata.

    Args:
        pages: Output of :func:`extract_pages`.
        source_name: Original filename, stored as chunk metadata.

    Returns:
        A list of chunk dicts, each shaped as::

            {
                "id": "<source>::p<page>::c<chunk_index>",
                "text": "<chunk text>",
                "metadata": {
                    "source": source_name,
                    "page": page_number,
                    "chunk_index": i,
                },
            }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks: list[dict] = []
    running_index = 0
    for page in pages:
        page_number = page["page"]
        for piece in splitter.split_text(page["text"]):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                {
                    "id": f"{source_name}::p{page_number}::c{running_index}",
                    "text": piece,
                    "metadata": {
                        "source": source_name,
                        "page": page_number,
                        "chunk_index": running_index,
                    },
                }
            )
            running_index += 1

    return chunks
