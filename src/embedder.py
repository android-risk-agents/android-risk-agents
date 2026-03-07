import os
from typing import List
from sentence_transformers import SentenceTransformer

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True
        )
    return _model

def embed_texts(texts: List[str], is_query: bool = False) -> List[List[float]]:
    if not texts:
        return []

    model = _get_model()
    prefix = "search_query: " if is_query else "search_document: "
    processed_texts = [prefix + t for t in texts]

    return model.encode(processed_texts, normalize_embeddings=True).tolist()

def chunk_text(text: str, chunk_size_chars: int = 1600, overlap_chars: int = 200) -> List[str]:
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