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
from .llm_client import get_llm_client, chat_json


SYSTEM_ANALYZE = (
    "You are the Coordinator AI agent for a digital fraud and risk intelligence system.\n"
    "Your job is to translate Android ecosystem security notes, policy updates, and integrity changes into detailed, implementable actions for:\n"
    "- risk model feature pipelines\n"
    "- monitoring and alerting\n"
    "- rule logic and enforcement checks\n"
    "- investigation workflows\n\n"
    "Hard rules:\n"
    "1) Do not invent facts. If unknown, say unknown.\n"
    "2) Ground recommendations in the provided CONTEXT snippets.\n"
    "3) Avoid generic advice. Each recommendation must reference a concrete signal, API, policy section, telemetry event, vulnerability theme, or control mentioned in CONTEXT.\n"
    "4) Output ONLY valid JSON. No markdown. No code fences."
)


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _as_float(x: Any, default: float = 0.65) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _as_list_str(x: Any, max_items: int, max_len: int) -> List[str]:
    if not isinstance(x, list):
        return []
    out: List[str] = []
    for item in x:
        s = str(item).strip()
        if not s:
            continue
        out.append(s[:max_len])
        if len(out) >= max_items:
            break
    return out


def _tags(ev: Dict[str, Any], max_items: int = 12) -> List[str]:
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


def _clamp_1_5(x: Any, default: int = 2) -> int:
    v = _as_int(x, default)
    if v < 1:
        return 1
    if v > 5:
        return 5
    return v


def _build_rag_query(ev_title: str, ev_summary: str, snap_text: str) -> str:
    """
    Focus retrieval on Sentinel summary + title first.
    Fallback adds a little snapshot head if needed.
    """
    base = (ev_title or "").strip() + "\n" + (ev_summary or "").strip()
    base = base.strip()
    if len(base) >= 220:
        return base[:1400]
    head = (snap_text or "").strip()[:1200]
    return (base + "\n" + head).strip()[:1400]


def rag_context_from_event(
    source_id: int,
    ev_title: str,
    ev_summary: str,
    snap_text: str,
    top_k: int,
) -> str:
    """
    768 + Nomic alignment:
    Always embed retrieval queries with is_query=True so prefixes are applied correctly.
    """
    q = _build_rag_query(ev_title, ev_summary, snap_text)
    if not q:
        return ""

    q_emb = embed_texts([q], is_query=True)[0]

    resp = vector_search(
        query_embedding=q_emb,
        match_count=top_k,
        filter_source_id=str(source_id) if source_id else None,
        filter_kind=None,
    )
    rows = resp.data or []

    out: List[str] = []
    for r in rows[:top_k]:
        chunk = (r.get("chunk_text") or "").strip()
        if chunk:
            out.append(chunk[:1100])

    return "\n\n---\n\n".join(out)


def build_prompt(url: str, title: str, summary: str, tags: List[str], context: str, final_risk: int) -> str:
    schema = {
        "title": "string",
        "summary": "string (3-6 sentences, concrete and grounded)",
        "category": "string (optional)",
        "affected_signals": ["string (up to 6)"],
        "recommended_actions": [
            "string (8-12 items, each must be implementable for risk models/monitoring and reference a concrete item from CONTEXT)"
        ],
        "evidence_snippets": ["string (3-6 short snippets copied or lightly quoted from CONTEXT, <= 220 chars each)"],
        "risk_score": "integer 1-5",
        "confidence": "number 0-1",
        "priority": "string (P0|P1|P2)"
    }

    instructions = (
        "Write recommendations for risk models and security monitoring.\n"
        "Recommendation quality bar:\n"
        "- Mention specific signals, APIs, policy clauses, enforcement mechanisms, or vulnerability themes present in CONTEXT.\n"
        "- Include checks we can implement: feature availability checks, telemetry assertions, regression tests, alert rules, denylist or threshold adjustments.\n"
        "- Prefer concrete verbs: add, instrument, validate, backfill, gate, alert, block, detect, review.\n"
        "- If you cannot find evidence, write 'unknown' explicitly and keep actions as validation steps.\n"
    )

    priority = "P0" if final_risk >= 85 else ("P1" if final_risk >= 70 else "P2")

    return (
        f"SOURCE URL: {url}\n\n"
        f"EVENT TITLE: {title}\n"
        f"EVENT SUMMARY (from Sentinel): {summary}\n"
        f"TAGS: {', '.join(tags) if tags else '[]'}\n"
        f"EVENT RISK (0-100): {final_risk}\n"
        f"DEFAULT PRIORITY: {priority}\n\n"
        f"CONTEXT (RAG snippets):\n{(context or '')[:8000]}\n\n"
        f"{instructions}\n"
        f"Return JSON only.\nSchema:\n{schema}"
    )


def main() -> int:
    agent_name = os.getenv("AGENT_NAME", "coordinator")
    model = os.getenv("MODEL_ANALYZE", os.getenv("GROQ_MODEL_ANALYZE", "llama-3.3-70b-versatile"))
    top_k = int(os.getenv("RAG_TOP_K", "6"))
    max_recs = int(os.getenv("MAX_RECOMMENDATIONS", "8"))

    client = get_llm_client()

    run_id = create_agent_run(
        run_name="coordinator",
        trigger="workflow_dispatch",
        llm_backend=os.getenv("LLM_BASE_URL", model),
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
            ev_id = _as_int(ev.get("id"), 0)
            if ev_id <= 0:
                continue
            processed_ids.append(ev_id)

            change_id = _as_int(ev.get("change_id"), 0)
            snapshot_id = _as_int(ev.get("snapshot_id"), 0)
            source_id = _as_int(ev.get("source_id"), 0)

            title = str(ev.get("title") or "Update detected")[:180]
            summary = str(ev.get("summary") or "")[:1400]
            tags = _tags(ev)

            final_risk = _as_int(ev.get("local_risk_score"), 50)
            priority_default = "P0" if final_risk >= 85 else ("P1" if final_risk >= 70 else "P2")

            try:
                url = get_source_url(source_id) or f"source_id={source_id}"
                snap_text = get_snapshot_text_by_id(snapshot_id) if snapshot_id else ""

                context = rag_context_from_event(
                    source_id=source_id,
                    ev_title=title,
                    ev_summary=summary,
                    snap_text=snap_text,
                    top_k=top_k,
                )

                prompt = build_prompt(url, title, summary, tags, context, final_risk)

                out = chat_json(
                    client=client,
                    model=model,
                    system=SYSTEM_ANALYZE,
                    user=prompt,
                    temperature=0.2,
                    max_tokens=1000,
                )

                affected = _as_list_str(out.get("affected_signals"), max_items=6, max_len=140)
                actions = _as_list_str(out.get("recommended_actions"), max_items=12, max_len=260)
                evidence = _as_list_str(out.get("evidence_snippets"), max_items=6, max_len=220)

                risk_score = _clamp_1_5(out.get("risk_score"), 2)
                confidence = _as_float(out.get("confidence"), 0.65)

                insight_title = str(out.get("title") or title)[:180]
                insight_summary = str(out.get("summary") or summary)[:2000]
                category = out.get("category")
                category = str(category).strip()[:120] if category else None

                priority = str(out.get("priority") or priority_default)[:8]

                # Put evidence into rationale so it is preserved in your recommendations table
                evidence_block = ""
                if evidence:
                    evidence_block = "\n\nEvidence snippets:\n- " + "\n- ".join(evidence[:6])

                rationale = (insight_summary + evidence_block)[:4000]

                # Write Insight
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
                        recommended_actions=actions[:6],
                        risk_score=risk_score,
                    )
                    stats["insights_written"] += 1
                    mark_change_analyzed(change_id)

                # Write Recommendation
                insert_recommendation(
                    run_id=run_id,
                    title=insight_title,
                    priority=priority,
                    final_risk_score=final_risk,
                    confidence=confidence,
                    event_ids=[ev_id],
                    change_ids=[change_id] if change_id > 0 else [],
                    rationale=rationale,
                    recommended_actions=actions
                    or [
                        "Add a telemetry assertion to confirm the referenced integrity or policy signal is still emitted and stable after this update.",
                        "Add a regression test that compares feature distributions and alert rates before vs after the change window.",
                        "Add an alert for sudden drops in key risk signals referenced in the security notes, and route to investigation queue.",
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
                        "model": model,
                        "rag_top_k": top_k,
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