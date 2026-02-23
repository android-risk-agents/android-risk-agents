import os
from typing import List

# Import Vertex AI
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

# Existing imports
from sentence_transformers import SentenceTransformer

MODEL_CHOICE = os.getenv("EMBEDDING_MODEL", "baai").lower()

def chunk_text(text: str, chunk_size_chars: int = 1600, overlap_chars: int = 200) -> List[str]:
    """
    Splits a large document into smaller chunks for the embedding model.
    Maintains a 200-character overlap so semantic meaning isn't lost between chunks.
    """
    if not text:
        return []
        
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size_chars, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        
        # If we've reached the end of the text, stop
        if end == text_length:
            break
            
        # Move the start forward, but step back by 'overlap_chars' to create the bridge
        start += (chunk_size_chars - overlap_chars)
        
    return chunks

def embed_texts(texts: List[str], is_query: bool = False) -> List[List[float]]:
    """
    Routes the text to the selected embedding model.
    """
    if not texts:
        return []

    if MODEL_CHOICE == "vertex":
        # Initialize Vertex AI connection
        project_id = os.getenv("GCP_PROJECT_ID")
        location = os.getenv("GCP_LOCATION", "us-central1")
        vertexai.init(project=project_id, location=location)
        
        # Load the 768-dimension model
        model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        
        # Tell Vertex exactly what we are embedding to optimize the vector placement
        task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        
        # Google's API requires a list of TextEmbeddingInput objects
        inputs = [TextEmbeddingInput(text=t, task_type=task_type) for t in texts]
        
        # Fetch embeddings
        embeddings = model.get_embeddings(inputs)
        return [emb.values for emb in embeddings]

    elif MODEL_CHOICE == "nomic":
        # Nomic MoE logic
        model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        prefix = "search_query: " if is_query else "search_document: "
        prefixed_texts = [prefix + t for t in texts]
        return model.encode(prefixed_texts).tolist()

    else:
        # BAAI-Base logic (Default)
        model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        if is_query:
            texts = ["Represent this sentence for searching relevant passages: " + t for t in texts]
        return model.encode(texts).tolist()