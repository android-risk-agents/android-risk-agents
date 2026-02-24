# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
USER_AGENT = os.getenv("USER_AGENT", "android-risk-agents-bot/0.1")

# ---- Embeddings / pgvector ----
# Set via ENV. For your upgraded setup use:
#   VECTOR_DIM=768
#   VECTOR_TABLE=vector_chunks_768   (or keep "vector_chunks" if you migrated in-place)
#   VECTOR_RPC_MATCH=match_vector_chunks_768  (or keep "match_vector_chunks" if migrated in-place)
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "768"))

# If you migrated in-place, leave defaults as below.
# If you keep parallel 384/768 stores, override these via env per workflow/branch.
VECTOR_TABLE = os.getenv("VECTOR_TABLE", "vector_chunks")
VECTOR_RPC_MATCH = os.getenv("VECTOR_RPC_MATCH", "match_vector_chunks")

# ---- Chunking ----
CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "1600"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))

# ---- Control flags ----
EMBED_BASELINE_ON_FIRST_SNAPSHOT = os.getenv("EMBED_BASELINE_ON_FIRST_SNAPSHOT", "true").lower() in (
    "1", "true", "yes", "on"
)
EMBED_DELTAS_ON_CHANGE = os.getenv("EMBED_DELTAS_ON_CHANGE", "true").lower() in (
    "1", "true", "yes", "on"
)

# ---- Model selection (matches your embedder.py) ----
# Choices: baai_small (384), baai_base (768), nomic (768)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic").lower()

# Optional safety guard: ensure VECTOR_DIM matches chosen model
_EXPECTED_DIMS = {
    "baai_small": 384,
    "baai_base": 768,
    "nomic": 768,
}

def validate_env():
    missing = []
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_SERVICE_ROLE_KEY:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

    expected = _EXPECTED_DIMS.get(EMBEDDING_MODEL)
    if expected and VECTOR_DIM != expected:
        raise RuntimeError(
            f"VECTOR_DIM={VECTOR_DIM} does not match EMBEDDING_MODEL='{EMBEDDING_MODEL}' "
            f"(expected {expected}). Set VECTOR_DIM or EMBEDDING_MODEL correctly."
        )