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
    "You are the Coordinator AI agent for a digital fraud risk intelligence system. "
    "You prioritize platform ecosystem updates and translate them into concrete opportunities for digital risk solutions. "
    "Do not invent facts. If something is unknown, say unknown. "
    "Return ONLY valid JSON, no markdown."
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


def _tags(ev: Dict[str, Any], max_items: int = 10) -> List[str]:
    t = ev.get("tags")
    if not isinstance(t, list):
        return []
    out: List[str] = []
    for item in t:
        s = str(item).strip()
        if not s:
            continue
        out.append(s[:60])
        if len(out) >= max_items:
            break
    return out


def build_deep_insight_prompt(url: str, title: str, summary: str, context: str) -> str:
    schema = {
        "title": "string",
        "summary": "string (2-5 sentences, concrete and actionable)",
        "category": "string (optional)",
        "affected_signals": ["string (up to 5)"],
        "recommended_actions": ["string (specific leverage ideas, up to 5)"],
        "risk_score": "integer 1-5",
        "confidence": "number 0-1",
    }
    return (
        f"SOURCE: {url}\n\n"
        f"EVENT TITLE: {title}\n"
        f"EVENT SUMMARY: {summary}\n\n"
        f"CONTEXT (RAG snippets):\n{context[:6500]}\n\n"
        "Return JSON only.\n"
        f"Schema:\n{schema}"
    )


def rag_context_from_text(text: str, top_k: int) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    q = t[:1200]
    q_emb = embed_texts([q])[0]

    resp = vector_search(query_embedding=q_emb, match_count=top_k, filter_source_id=None, filter_kind=None)
    rows = resp.data or []

    out: List[str] = []
    for r in rows[:top_k]:
        chunk = (r.get("chunk_text") or "").strip()
        if chunk:
            out.append(chunk[:900])

    return "\n\n---\n\n".join(out)


def main() -> int:
    agent_name = os.getenv("AGENT_NAME", "coordinator")

    # Keep old behavior: env override allowed, but default is now NIM-friendly
    model = os.getenv("MODEL_ANALYZE", "meta/llama3-8b-instruct")

    top_k = int(os.getenv("RAG_TOP_K", "6"))
    max_recs = int(os.getenv("MAX_RECOMMENDATIONS", "8"))

    client = NimClient()

    run_id = create_agent_run(
        run_name="coordinator",
        trigger="workflow_dispatch",
        llm_backend="nvidia-nim",
    )

    stats = {
        "events_seen": 0,
        "events_processed": 0,
        "recommendations_written": 0,
        "insights_written": 0,
        "errors": 0,
    }

    try:
        events = get_pending_agent_events(limit=60)
        if not events:
            finish_agent_run(run_id, status="success", stats={**stats, "note": "no pending events"})
            return 0

        stats["events_seen"] = len(events)

        events_sorted = sorted(
            events,
            key=lambda e: (_as_int(e.get("local_risk_score", 0)), _as_int(e.get("relevance_score", 0))),
            reverse=True,
        )

        top_events = events_sorted[:max_recs]
        processed_ids: List[int] = []

        for ev in top_events:
            ev_id = int(ev["id"])
            processed_ids.append(ev_id)

            change_id = _as_int(ev.get("change_id"), 0)
            snapshot_id = _as_int(ev.get("snapshot_id"), 0)
            source_id = _as_int(ev.get("source_id"), 0)

            title = str(ev.get("title") or "Update detected")[:160]
            summary = str(ev.get("summary") or "")[:1000]
            tags = _tags(ev)

            try:
                url = get_source_url(source_id) or f"source_id={source_id}"
                snap_text = get_snapshot_text_by_id(snapshot_id) if snapshot_id else ""
                context = rag_context_from_text(snap_text, top_k=top_k)

                prompt = build_deep_insight_prompt(url, title, summary, context)

                out = client.chat_json(
                    model=model,
                    system=SYSTEM_ANALYZE,
                    user=prompt,
                    temperature=0.2,
                    max_tokens=520,
                )

                affected = _as_list_str(out.get("affected_signals"), max_items=5, max_len=120) or []
                actions = _as_list_str(out.get("recommended_actions"), max_items=5, max_len=180) or []
                risk_score = _as_int(out.get("risk_score"), 2)
                confidence = _as_float(out.get("confidence"), 0.6)

                insight_title = str(out.get("title") or title)[:180]
                insight_summary = str(out.get("summary") or summary)[:2000]
                category = out.get("category")
                category = str(category).strip()[:120] if category else None

                final_risk = _as_int(ev.get("local_risk_score"), 50)
                priority = "P0" if final_risk >= 85 else ("P1" if final_risk >= 70 else "P2")

                if change_id > 0 and source_id > 0 and snapshot_id > 0:
                    insert_insight(
                        change_id=change_id,
                        source_id=source_id,
                        snapshot_id=snapshot_id,
                        agent_name=agent_name,
                        title=insight_title,
                        summary=insight_summary,
                        confidence=confidence,
                        category=category,
                        affected_signals=affected,
                        recommended_actions=actions,
                        risk_score=risk_score,
                    )
                    stats["insights_written"] += 1
                    mark_change_analyzed(change_id)

                insert_recommendation(
                    run_id=run_id,
                    title=insight_title,
                    priority=priority,
                    final_risk_score=final_risk,
                    confidence=confidence,
                    event_ids=[ev_id],
                    change_ids=[change_id] if change_id > 0 else [],
                    rationale=insight_summary,
                    recommended_actions=actions
                    or [
                        "Review impact on fraud signals and data availability",
                        "Update monitoring rules and feature pipeline if needed",
                        "Add regression checks for affected signals",
                    ],
                    related_tags=tags,
                )

                stats["recommendations_written"] += 1
                stats["events_processed"] += 1

                audit_log(
                    run_id=run_id,
                    agent_name=agent_name,
                    action="coordinator_written",
                    ref_type="agent_event",
                    ref_id=ev_id,
                    detail={
                        "change_id": change_id,
                        "snapshot_id": snapshot_id,
                        "priority": priority,
                        "final_risk_score": final_risk,
                    },
                )

            except Exception as e:
                stats["errors"] += 1
                audit_log(
                    run_id=run_id,
                    agent_name=agent_name,
                    action="coordinator_error",
                    ref_type="agent_event",
                    ref_id=ev_id,
                    detail={"error": str(e)[:600]},
                )

        mark_agent_events_processed(processed_ids)

        finish_agent_run(run_id, status="success", stats=stats)
        return 0

    except Exception as e:
        finish_agent_run(run_id, status="failed", stats={**stats, "fatal": str(e)[:600]})
        raise


if __name__ == "__main__":
    raise SystemExit(main())