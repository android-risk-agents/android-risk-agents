# streamlit_app.py
# Android Risk Agents Demo
# Tab 1: Insights + Recommendations
# Tab 2: RAG Chatbot (local embeddings + Supabase pgvector RPC + Groq for answer)

import os
import json
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ----------------------------
# Load .env
# ----------------------------
load_dotenv()  # expects .env in same folder; or set ENV_PATH and use load_dotenv(ENV_PATH)

# ----------------------------
# Env / Config
# ----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL_CHAT", "llama-3.3-70b-versatile")

# ✅ Use your new non-overloaded RPC name
MATCH_FN = os.getenv("MATCH_FN", "match_vector_chunks_v1")

# Tables (based on your schema screenshot)
INSIGHTS_TABLE = os.getenv("INSIGHTS_TABLE", "insights")
RECS_TABLE = os.getenv("RECS_TABLE", "recommendations")

# Local embedder (must match what you used to generate vector_chunks.embedding)
EMBEDDER_NAME = os.getenv("EMBEDDER_NAME", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))

# ----------------------------
# Helpers
# ----------------------------
def jdump(x):
    return json.dumps(x, indent=2, ensure_ascii=False, default=str)

@st.cache_resource
def get_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY / SUPABASE_ANON_KEY")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource
def get_llm():
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY")
    return OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBEDDER_NAME)

def build_context(matches, max_chars=8000):
    # matches contain chunk_text, similarity, etc
    parts = []
    total = 0
    used = []
    for i, m in enumerate(matches, start=1):
        txt = (m.get("chunk_text") or "").strip()
        if not txt:
            continue
        sim = m.get("similarity")
        header = f"[Chunk {i} | sim={sim:.4f} | source_id={m.get('source_id')} | kind={m.get('kind')} | snapshot_sha={m.get('snapshot_sha')}]"
        block = header + "\n" + txt + "\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        used.append(m)
        total += len(block)
    return "\n\n".join(parts).strip(), used

def groq_answer(llm, question, context):
    system = (
        "You are a security risk analyst assistant for Android changes. "
        "Use ONLY the provided context. If context is insufficient, say what you need. "
        "Be practical and actionable."
    )
    user = f"""Question:
{question}

Context:
{context if context else "(no context retrieved)"}"""
    resp = llm.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

# ----------------------------
# Init
# ----------------------------
st.set_page_config(page_title="Android Risk Agents Demo", layout="wide")
st.title("Android Risk Agents Demo")

with st.sidebar:
    st.subheader("Config")
    st.code(
        "\n".join(
            [
                f"SUPABASE_URL={'set' if SUPABASE_URL else 'missing'}",
                f"SUPABASE_KEY={'set' if SUPABASE_KEY else 'missing'}",
                f"GROQ_API_KEY={'set' if GROQ_API_KEY else 'missing'}",
                f"GROQ_BASE_URL={GROQ_BASE_URL}",
                f"GROQ_MODEL={GROQ_MODEL}",
                f"MATCH_FN={MATCH_FN}",
                f"EMBEDDER_NAME={EMBEDDER_NAME}",
                f"TOP_K={TOP_K}",
            ]
        )
    )

sb = get_supabase()
llm = get_llm()
embedder = get_embedder()

tab1, tab2 = st.tabs(["📊 Insights & Recommendations", "🤖 RAG Chatbot"])

# ----------------------------
# TAB 1: Insights + Recommendations
# ----------------------------
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Insights (latest)")
        insights = (
            sb.table(INSIGHTS_TABLE)
            .select("*")
            .order("created_at", desc=True)
            .limit(10)
            .execute()
            .data
        )

        if not insights:
            st.info("No insights found.")
        else:
            for it in insights:
                st.markdown(f"**{it.get('title','(no title)')}**")
                if it.get("summary"):
                    st.write(it["summary"])
                st.caption(f"risk_score={it.get('risk_score')} | confidence={it.get('confidence')} | created_at={it.get('created_at')}")
                if it.get("recommended_actions") is not None:
                    st.json(it["recommended_actions"])
                with st.expander("Raw"):
                    st.code(jdump(it), language="json")
                st.divider()

    with c2:
        st.subheader("Recommendations (latest)")
        recs = (
            sb.table(RECS_TABLE)
            .select("*")
            .order("created_at", desc=True)
            .limit(10)
            .execute()
            .data
        )

        if not recs:
            st.info("No recommendations found.")
        else:
            for r in recs:
                st.markdown(f"**{r.get('title','(no title)')}**")
                st.caption(
                    f"priority={r.get('priority')} | final_risk_score={r.get('final_risk_score')} | "
                    f"confidence={r.get('confidence')} | created_at={r.get('created_at')}"
                )
                if r.get("rationale"):
                    st.write(r["rationale"])
                if r.get("recommended_actions") is not None:
                    st.json(r["recommended_actions"])
                with st.expander("Raw"):
                    st.code(jdump(r), language="json")
                st.divider()

# ----------------------------
# TAB 2: RAG Chatbot
# ----------------------------
with tab2:
    st.subheader("Ask about Android security changes")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Retrieved chunks"):
                    st.code(jdump(msg["sources"]), language="json")

    q = st.chat_input("Ask a question (example: What is the latest high-risk change and what should we do?)")

    if q:
        st.session_state.chat.append({"role": "user", "content": q})

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context..."):
                # 1) Local embedding for query (matches stored vectors)
                q_vec = embedder.encode(q).tolist()

                # 2) Supabase pgvector RPC call (your new function)
                matches = (
                    sb.rpc(
                        MATCH_FN,
                        {
                            "query_embedding": q_vec,
                            "match_count": TOP_K,
                        },
                    )
                    .execute()
                    .data
                )

                context, used = build_context(matches, max_chars=8000)

            with st.spinner("Generating answer (Groq)..."):
                answer = groq_answer(llm, q, context)

            st.markdown(answer)

            if used:
                with st.expander("Retrieved chunks"):
                    compact = [
                        {
                            "id": m.get("id"),
                            "source_id": m.get("source_id"),
                            "snapshot_sha": m.get("snapshot_sha"),
                            "kind": m.get("kind"),
                            "chunk_index": m.get("chunk_index"),
                            "similarity": m.get("similarity"),
                            "chunk_text_preview": (m.get("chunk_text") or "")[:240],
                        }
                        for m in used
                    ]
                    st.code(jdump(compact), language="json")

        st.session_state.chat.append({"role": "assistant", "content": answer, "sources": used})