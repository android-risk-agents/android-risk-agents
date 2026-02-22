import os
from typing import List
from sentence_transformers import SentenceTransformer

# Read the model choice from your .env file
# Choices: 'baai_small' (384), 'baai_base' (768), 'nomic' (768)
MODEL_CHOICE = os.getenv("EMBEDDING_MODEL", "baai_small").lower()

if MODEL_CHOICE == "nomic":
    # 768 Dimensions - Optimized for long context (8k tokens)
    _model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
elif MODEL_CHOICE == "baai_base":
    # 768 Dimensions - The upgraded BAAI baseline
    _model = SentenceTransformer("BAAI/bge-base-en-v1.5")
else:
    # 384 Dimensions - The original MVP baseline
    _model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def embed_texts(texts: List[str], is_query: bool = False) -> List[List[float]]:
    if not texts:
        return []
    
    # Nomic strictly requires task prefixes
    if MODEL_CHOICE == "nomic":
        prefix = "search_query: " if is_query else "search_document: "
        processed_texts = [prefix + t for t in texts]
    else:
        # BAAI models also benefit from query instructions
        # but technically only need them for the 'query' side
        prefix = "Represent this sentence for searching relevant passages: " if is_query else ""
        processed_texts = [prefix + t for t in texts]
        
    return _model.encode(processed_texts, normalize_embeddings=True).tolist()

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