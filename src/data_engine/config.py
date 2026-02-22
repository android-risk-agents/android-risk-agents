# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
USER_AGENT = os.getenv("USER_AGENT", "android-risk-agents-bot/0.1")

# ---- Vector / embeddings (pgvector) ----
# Your Supabase table uses vector(384)
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "384"))
VECTOR_TABLE = os.getenv("VECTOR_TABLE", "vector_chunks")
VECTOR_RPC_MATCH = os.getenv("VECTOR_RPC_MATCH", "match_vector_chunks")

# Chunking config
CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "1600"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))

# Control flags
EMBED_BASELINE_ON_FIRST_SNAPSHOT = os.getenv("EMBED_BASELINE_ON_FIRST_SNAPSHOT", "true").lower() in ("1", "true", "yes", "on")
EMBED_DELTAS_ON_CHANGE = os.getenv("EMBED_DELTAS_ON_CHANGE", "true").lower() in ("1", "true", "yes", "on")


def validate_env():
    missing = []
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_SERVICE_ROLE_KEY:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
