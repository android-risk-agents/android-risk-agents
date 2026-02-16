# demo_streamlit.py
import os
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# ------------------------------------------------------------
# Load .env (local demo only). Keep src/config.py untouched.
# ------------------------------------------------------------
load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"), override=True)

from src.db import get_supabase_client  # noqa: E402
from src.config import VECTOR_RPC_MATCH  # noqa: E402
from src.embedder import embed_texts  # noqa: E402

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(page_title="Android Risk Agents Demo", layout="wide")
st.title("Android Risk Agents - Demo Dashboard")

# ------------------------------------------------------------
# Groq (OpenAI-compatible)
# ------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
GROQ_MODEL_ANALYZE = os.getenv("GROQ_MODEL_ANALYZE", "llama-3.3-70b-versatile")

# ------------------------------------------------------------
# Defaults + RAG context controls (prevents "full pages")
# ------------------------------------------------------------
DEFAULT_LIMIT = 25
DEFAULT_TOPK = 3

MAX_CONTEXT_CHARS_PER_CHUNK = 900   # what Groq sees per chunk
MAX_CONTEXT_CHARS_TOTAL = 2200      # total context sent to Groq
MAX_UI_SNIPPET_CHARS = 280          # what you display in the app


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
def _df(data):
    return pd.DataFrame(data) if data else pd.DataFrame()


def _first_present(d: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def compact_context(blocks: List[str]) -> List[str]:
    """
    Trim each block and cap total context to avoid sending full pages to the LLM.
    """
    trimmed: List[str] = []
    total = 0
    for b in blocks:
        b = (b or "").strip()
        if not b:
            continue

        b = b[:MAX_CONTEXT_CHARS_PER_CHUNK]

        if total + len(b) > MAX_CONTEXT_CHARS_TOTAL:
            remaining = MAX_CONTEXT_CHARS_TOTAL - total
            if remaining <= 0:
                break
            b = b[:remaining]

        trimmed.append(b)
        total += len(b)

        if total >= MAX_CONTEXT_CHARS_TOTAL:
            break

    return trimmed


# ------------------------------------------------------------
# Supabase queries
# ------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_recent_insights(limit: int = DEFAULT_LIMIT) -> List[Dict[str, Any]]:
    sb = get_supabase_client()
    resp = (
        sb.table("insights")
        .select(
            "id, change_id, agent_name, title, summary, confidence, category, risk_score, created_at"
        )
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return resp.data or []


def build_recommendations(insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Creates a Recommendations view robustly.
    If your insight schema later includes specific action fields, this will pick them up.
    Otherwise it falls back to summary.
    """
    recs = []
    for r in insights:
        title = r.get("title") or "(no title)"
        created = r.get("created_at")
        risk = r.get("risk_score")
        conf = r.get("confidence")
        category = r.get("category")

        recommendation = _first_present(
            r,
            [
                "recommendation",
                "recommendations",
                "action_items",
                "actions",
                "next_steps",
                "mitigation",
            ],
        )
        if not recommendation:
            recommendation = r.get("summary") or ""

        recs.append(
            {
                "title": title,
                "recommendation": recommendation,
                "risk_score": risk,
                "confidence": conf,
                "category": category,
                "created_at": created,
                "insight_id": r.get("id"),
            }
        )
    return recs


# ------------------------------------------------------------
# RAG: retrieve relevant chunks via pgvector RPC
# ------------------------------------------------------------
def rag_retrieve(query: str, top_k: int = DEFAULT_TOPK) -> List[Dict[str, Any]]:
    sb = get_supabase_client()

    # Embed query (uses your existing embedder)
    q_emb = embed_texts([query])[0]

    # Try a few common RPC signatures (in case your SQL function differs)
    payload_candidates = [
        {"query_embedding": q_emb, "match_count": top_k},
        {"query_embedding": q_emb, "match_count": top_k, "similarity_threshold": 0.0},
        {"embedding": q_emb, "match_count": top_k},
        {"query_embedding": q_emb, "k": top_k},
    ]

    last_err = None
    for payload in payload_candidates:
        try:
            resp = sb.rpc(VECTOR_RPC_MATCH, payload).execute()
            data = resp.data or []
            if isinstance(data, list):
                return data
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Vector RPC call failed for '{VECTOR_RPC_MATCH}'. "
        f"Your SQL function signature likely differs from payload keys. "
        f"Last error: {type(last_err).__name__}: {last_err}"
    )


def format_match(m: Dict[str, Any]) -> str:
    """
    Convert a match row into readable text.
    Adjust keys if your RPC returns different fields.
    """
    text = _first_present(m, ["content", "chunk_text", "text", "clean_text", "chunk"])
    meta = _first_present(m, ["url", "source_url", "source", "source_name"])
    score = m.get("similarity") or m.get("score")

    header = []
    if meta:
        header.append(f"Source: {meta}")
    if score is not None:
        header.append(f"Score: {score}")

    prefix = ("\n".join(header) + "\n\n") if header else ""
    return (prefix + text).strip()


# ------------------------------------------------------------
# LLM call: Groq (concise, no dumping context)
# ------------------------------------------------------------
def groq_answer(user_query: str, context_blocks: List[str]) -> str:
    if not GROQ_API_KEY:
        return ""

    url = f"{GROQ_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    system = (
        "You are a security and digital risk analyst for Android ecosystem changes.\n"
        "Rules:\n"
        "1) Answer ONLY using the provided context.\n"
        "2) Be concise: 4-6 bullet points max.\n"
        "3) Do NOT quote large passages or reproduce the context.\n"
        "4) If context is insufficient, say what is missing in 1 sentence.\n"
    )

    context = "\n\n".join([f"[Context {i+1}]\n{c}" for i, c in enumerate(context_blocks)])

    payload = {
        "model": GROQ_MODEL_ANALYZE,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question:\n{user_query}\n\nContext:\n{context}"},
        ],
        "temperature": 0.2,
        "max_tokens": 350,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=(10, 120))
    r.raise_for_status()
    out = r.json()
    return out["choices"][0]["message"]["content"]


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.header("Controls")
limit = st.sidebar.slider("Insights to load", 10, 100, DEFAULT_LIMIT, 5)
top_k = st.sidebar.slider("RAG top-k", 1, 8, DEFAULT_TOPK, 1)
show_snippets = st.sidebar.checkbox("Show retrieved snippets", value=True)
refresh = st.sidebar.button("Refresh")

if refresh:
    fetch_recent_insights.clear()

# ------------------------------------------------------------
# Main: only Insights + Recommendations
# ------------------------------------------------------------
insights = fetch_recent_insights(limit=limit)
recs = build_recommendations(insights)

colA, colB = st.columns(2)

with colA:
    st.subheader("Insights")
    if not insights:
        st.info("No insights found yet. Run the pipeline to generate insights.")
    else:
        for r in insights:
            title = r.get("title") or "(no title)"
            meta = []
            if r.get("risk_score") is not None:
                meta.append(f"risk={r.get('risk_score')}")
            if r.get("confidence") is not None:
                meta.append(f"conf={r.get('confidence')}")
            if r.get("category"):
                meta.append(f"cat={r.get('category')}")
            if r.get("created_at"):
                meta.append(f"at={r.get('created_at')}")

            with st.expander(f"{title}  |  " + "  |  ".join(meta)):
                st.write(r.get("summary") or "")
                with st.expander("Raw insight JSON"):
                    st.json(r)

with colB:
    st.subheader("Recommendations")
    if not recs:
        st.info("No recommendations yet.")
    else:
        df = _df(recs)[["title", "recommendation", "risk_score", "confidence", "category", "created_at"]]
        st.dataframe(df, use_container_width=True, height=520)

st.divider()

# ------------------------------------------------------------
# RAG Chat
# ------------------------------------------------------------
st.subheader("RAG Test (chat)")

if "chat" not in st.session_state:
    st.session_state.chat = []

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_msg = st.chat_input("Ask a question, e.g., 'Summarize key changes in the latest bulletin'")

if user_msg:
    st.session_state.chat.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context..."):
            try:
                # Retrieve more than needed, then compact down
                matches = rag_retrieve(user_msg, top_k=max(top_k * 3, 12))
                blocks = [format_match(m) for m in matches]
                blocks = [b for b in blocks if b]
            except Exception as e:
                st.error(str(e))
                blocks = []

        # Compact context for the model (this prevents full pages)
        compact_blocks = compact_context(blocks[:top_k])

        if show_snippets and compact_blocks:
            st.markdown("**Retrieved snippets (short):**")
            for c in compact_blocks:
                st.code(c[:MAX_UI_SNIPPET_CHARS] + ("..." if len(c) > MAX_UI_SNIPPET_CHARS else ""))

        answer = ""
        if compact_blocks and GROQ_API_KEY:
            with st.spinner("Generating answer (Groq)..."):
                try:
                    answer = groq_answer(user_msg, compact_blocks)
                except Exception as e:
                    st.error(f"Groq call failed: {e}")

        if answer:
            st.markdown(answer)
            st.session_state.chat.append({"role": "assistant", "content": answer})
        else:
            fallback = (
                "I retrieved context, but could not generate an answer. "
                "Make sure GROQ_API_KEY is set in your .env, and try a more specific question."
            )
            st.markdown(fallback)
            st.session_state.chat.append({"role": "assistant", "content": fallback})

st.caption(
    "Demo notes: Retrieval uses Supabase pgvector RPC, then context is aggressively trimmed before sending to Groq "
    "to avoid dumping full pages."
)