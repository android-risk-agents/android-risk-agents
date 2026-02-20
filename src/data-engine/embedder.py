# src/embedder.py
from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer

# 384-dim embeddings -> matches your vector(384) column
_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_model = SentenceTransformer(_MODEL_NAME)


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    return _model.encode(texts, normalize_embeddings=True).tolist()


def chunk_text(text: str, chunk_size_chars: int = 1600, overlap_chars: int = 200) -> List[str]:
    """
    Simple char-based chunking (robust + fast).
    Works fine for first version. Later you can switch to token-based chunking.
    """
    t = (text or "").strip()
    if not t:
        return []

    if chunk_size_chars <= 0:
        return [t]

    chunks: List[str] = []
    start = 0
    n = len(t)

    while start < n:
        end = min(n, start + chunk_size_chars)
        chunk = t[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap_chars)

    return chunks
