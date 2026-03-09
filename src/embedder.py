# src/embedder.py
import os
from typing import List
from sentence_transformers import SentenceTransformer

# Default to Nomic since it won our evaluation
MODEL_CHOICE = os.getenv("EMBEDDING_MODEL", "nomic").lower()

# Global variable to hold the model in memory (MLOps optimization)
_model_instance = None

def _get_model():
    """
    Lazy-loads the embedding model into memory exactly once.
    This prevents the pipeline from crashing your RAM or slowing down 
    by reloading the heavy weights on every single chunk.
    """
    global _model_instance
    if _model_instance is None:
        if MODEL_CHOICE == "nomic":
            print("🧠 Loading Nomic v1.5 into memory (trust_remote_code=True)...")
            _model_instance = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        else:
            print("🧠 Loading BAAI Base into memory...")
            _model_instance = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return _model_instance

def embed_texts(texts: List[str], is_query: bool = False) -> List[List[float]]:
    """
    Translates semantic text chunks into 768-dimensional vectors.
    """
    if not texts:
        return []

    model = _get_model()

    if MODEL_CHOICE == "nomic":
        # Nomic uses specific task prefixes to optimize the vector space
        prefix = "search_query: " if is_query else "search_document: "
        prefixed_texts = [prefix + t for t in texts]
        
        # .encode() handles the sub-word tokenization and vector math automatically
        return model.encode(prefixed_texts).tolist()

    else:
        # BAAI-Base logic (Fallback)
        if is_query:
            texts = ["Represent this sentence for searching relevant passages: " + t for t in texts]
        return model.encode(texts).tolist()