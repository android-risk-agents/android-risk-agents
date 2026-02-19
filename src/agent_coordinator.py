# src/agent_coordinator.py

import os
from typing import Any, Dict, List, Optional

from .db import (
    create_agent_run,
    finish_agent_run,
    audit_log,
    get_pending_agent_events,
    mark_agent_events_processed,
    get_snapshot_text_by_id,
    insert_recommendation,
    insert_insight,
    mark_change_analyzed,
    get_source_url,
    vector_search,
)

from .embedder import embed_texts
from .nim_client import NimClient


SYSTEM_ANALYZE = (
    "You are the Coordinator AI for Android fraud intelligence.\n\n"
    "Translate Android ecosystem updates into concrete fraud detection opportunities.\n"
    "Explain clearly:\n"
    "- Which fraud signals are affected\n"
    "- How model features may break or improve\n"
    "- Data availability impact\n"
    "- Monitoring adjustments required\n"
    "- Strategic opportunities for fraud prevention\n\n"
    "Make recommendations implementation-oriented.\n"
    "Return ONLY valid JSON."
)


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _as_float(x: Any, default: float = 0.6) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _as_list_str(x: Any, max_items: int, max_len: int) -> Optional[List[str]]:
    if not isinstance(x, list):
        return None
    out: List[str] = []
    for item in x:
        s = str(item).strip()
        if not s:
            continue
        out.append(s[:max_len])
        if len(out) >= max_items:
            break
    return out or None


def build_deep_insight_prompt(url: str, title: str, summary: str, context: str) -> str:
    return f"""
SOURCE: {url}

EVENT TITLE:
{title}

EVENT SUMMARY:
{summary}

RAG CONTEXT:
{context[:6500]}

Return JSON:
{{
"title": string,
"summary": string (2-5 sentences explaining fraud impact),
"category": string,
"affected_signals": list,
"recommended_actions": list (clear implementation steps),
"risk_score": 1-5,
"confidence": 0-1
}}
"""


def rag_context_from_text(text: str, top_k: int) -> str:
    if not text:
        return ""
    emb = embed_texts([text[:1200]])[0]
    resp = vector_search(query_embedding=emb, match_count=top_k)
    rows = resp.data or []
    return "\n\n---\n\n".join(r.get("chunk_text", "")[:900] for r in rows[:top_k])


def main():

    agent_name = "coordinator"
    model = os.getenv("MODEL_ANALYZE", "meta/llama3-8b-instruct")
    top_k = int(os.getenv("RAG_TOP_K", "6"))

    client = NimClient()

    run_id = create_agent_run("coordinator", "workflow_dispatch", "nvidia-nim")

    stats = {"events_processed": 0, "recommendations_written": 0, "insights_written": 0}

    events = get_pending_agent_events(limit=60)

    for ev in events:

        url = get_source_url(ev["source_id"])
        snap_text = get_snapshot_text_by_id(ev["snapshot_id"])
        context = rag_context_from_text(snap_text, top_k)

        prompt = build_deep_insight_prompt(url, ev["title"], ev["summary"], context)

        out = client.chat_json(
            model=model,
            system=SYSTEM_ANALYZE,
            user=prompt,
            temperature=0.2,
            max_tokens=600,
        )

        insert_insight(
            change_id=ev["change_id"],
            source_id=ev["source_id"],
            snapshot_id=ev["snapshot_id"],
            agent_name=agent_name,
            title=out.get("title"),
            summary=out.get("summary"),
            confidence=_as_float(out.get("confidence")),
            category=out.get("category"),
            affected_signals=_as_list_str(out.get("affected_signals"), 5, 120),
            recommended_actions=_as_list_str(out.get("recommended_actions"), 5, 200),
            risk_score=_as_int(out.get("risk_score"), 3),
        )

        insert_recommendation(
            run_id=run_id,
            title=out.get("title"),
            priority="P1",
            final_risk_score=ev.get("local_risk_score", 50),
            confidence=_as_float(out.get("confidence")),
            event_ids=[ev["id"]],
            change_ids=[ev["change_id"]],
            rationale=out.get("summary"),
            recommended_actions=_as_list_str(out.get("recommended_actions"), 5, 200),
            related_tags=ev.get("tags"),
        )

        stats["events_processed"] += 1
        stats["recommendations_written"] += 1
        stats["insights_written"] += 1

    mark_agent_events_processed([e["id"] for e in events])
    finish_agent_run(run_id, "success", stats)