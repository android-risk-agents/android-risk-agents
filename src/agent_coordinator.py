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
    "You are the Coordinator agent for a digital fraud and risk intelligence system. "
    "Your job is to translate platform ecosystem updates into detailed, implementable actions for risk monitoring and risk models. "
    "Do not invent facts. If something is unknown, say unknown. "
    "Ground your answer in the provided CONTEXT snippets. "
    "Avoid generic advice. Every recommendation must reference a specific signal, policy section, enforcement mechanism, API, telemetry event, "
    "or vulnerability theme mentioned in the context. "
    "Return ONLY valid JSON, no markdown. "
    "Requirements:\n"
    "- summary MUST be 10 to 14 bullets, each bullet begins with '-'.\n"
    "- recommended_actions MUST be 8 to 12 items, each in this exact format:\n"
    "  Owner=<role>; When=<timeframe>; How=<concrete steps>; Validate=<test or metric>\n"
    "- Include evidence_snippets: 3 to 6 short quotes from context (<=200 chars each).\n"
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


def build_deep_insight_prompt(url: str, title: str, summary: str, context: str, baseline: bool) -> str:
    schema = {
        "title": "string (specific, include feature or policy name when possible)",
        "summary": "string (10-14 bullets, each starts with '-')",
        "category": "string (vulnerability_intel | policy_intel | api_signal_change | enforcement_change | telemetry_change | other)",
        "affected_signals": ["string (up to 8, specific signals or controls)"],
        "recommended_actions": [
            "string (8-12 items, format: Owner=<role>; When=<timeframe>; How=<steps>; Validate=<test or metric>)"
        ],
        "risk_score": "integer 1-5",
        "confidence": "number 0-1",
        "evidence_snippets": ["string (3-6 quotes from context, <=200 chars each)"],
    }

    mode = "BASELINE_INTELLIGENCE" if baseline else "UPDATE_INTELLIGENCE"

    return (
        f"MODE: {mode}\n"
        f"SOURCE: {url}\n\n"
        f"EVENT TITLE: {title}\n"
        f"EVENT SUMMARY: {summary}\n\n"
        "Instructions:\n"
        "- Use CONTEXT snippets as evidence.\n"
        "- If MODE is BASELINE_INTELLIGENCE: extract key controls, policies, signals, and enforcement points present now, "
        "and propose monitoring and validation to detect future drift.\n"
        "- If MODE is UPDATE_INTELLIGENCE: explain what changed and how it could affect monitoring coverage and risk decisions.\n\n"
        f"CONTEXT (RAG snippets):\n{context[:9000]}\n\n"
        "Return JSON only.\n"
        "Do not add extra keys. Do not wrap in markdown.\n"
        f"Schema:\n{schema}"
    )


def rag_context_from_event(title: str, summary: str, text: str, top_k: int) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    # Better query: combine event title + summary + doc head
    q = (f"{title}\n{summary}\n{t[:1400]}").strip()[:1800]
    q_emb = embed_texts([q])[0]

    resp = vector_search(query_embedding=q_emb, match_count=top_k, filter_source_id=None, filter_kind=None)
    rows = resp.data or []

    out: List[str] = []
    for r in rows[:top_k]:
        chunk = (r.get("chunk_text") or "").strip()
        if chunk:
            out.append(chunk[:1200])

    return "\n\n---\n\n".join(out)


def main() -> int:
    agent_name = os.getenv("AGENT_NAME", "coordinator")
    model = os.getenv("MODEL_ANALYZE", "meta/llama3-8b-instruct")
    top_k = int(os.getenv("RAG_TOP_K", "6"))
    max_recs = int(os.getenv("MAX_RECOMMENDATIONS", "8"))

    debug = os.getenv("DEBUG_LLM", "").strip() in ("1", "true", "TRUE", "yes", "YES")

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

        # Prioritize by local_risk_score then relevance_score
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

            title = str(ev.get("title") or "Update detected")[:220]
            ev_summary = str(ev.get("summary") or "")[:2500]
            tags = _tags(ev)

            # Identify baseline events so prompt uses baseline intelligence mode
            baseline = str(ev.get("event_type") or "").lower() == "baseline_init"

            try:
                url = get_source_url(source_id) or f"source_id={source_id}"
                snap_text = get_snapshot_text_by_id(snapshot_id) if snapshot_id else ""

                context = rag_context_from_event(title, ev_summary, snap_text, top_k=top_k)

                prompt = build_deep_insight_prompt(url, title, ev_summary, context, baseline=baseline)

                out = client.chat_json(
                    model=model,
                    system=SYSTEM_ANALYZE,
                    user=prompt,
                    temperature=0.2,
                    max_tokens=900,  # increased for detailed actions
                    request_id=f"event={ev_id}",
                )

                affected = _as_list_str(out.get("affected_signals"), max_items=8, max_len=140) or []
                actions = _as_list_str(out.get("recommended_actions"), max_items=12, max_len=260) or []
                evidence = _as_list_str(out.get("evidence_snippets"), max_items=6, max_len=200) or []
                risk_score = _as_int(out.get("risk_score"), 2)
                confidence = _as_float(out.get("confidence"), 0.65)

                insight_title = str(out.get("title") or title)[:240]
                insight_summary = str(out.get("summary") or ev_summary)[:5000]
                category = out.get("category")
                category = str(category).strip()[:140] if category else None

                # Final risk in 0-100 scale comes from triage
                final_risk = _as_int(ev.get("local_risk_score"), 50)
                priority = "P0" if final_risk >= 85 else ("P1" if final_risk >= 70 else "P2")

                # Add evidence to rationale so it is not vague
                rationale_parts: List[str] = [insight_summary]
                if evidence:
                    rationale_parts.append("Evidence:")
                    for q in evidence[:6]:
                        rationale_parts.append(f"- {q}")
                rationale = "\n".join(rationale_parts)[:6000]

                # Write Insight (deep, structured)
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

                # Write Recommendation (prioritized feed)
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
                        "Owner=Risk Ops; When=This week; How=Review impacted controls and monitoring rules; Validate=Confirm alert coverage and false positive rate",
                        "Owner=Data Engineering; When=This week; How=Verify required signals are still available and correctly parsed; Validate=Backfill check and schema drift check",
                        "Owner=Risk Analytics; When=Next sprint; How=Run impact analysis on key decision points; Validate=Compare outcomes on a held-out period",
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
                        "baseline": baseline,
                    },
                )

                if debug:
                    print(f"[COORD] event_id={ev_id} wrote priority={priority} actions={len(actions)} evidence={len(evidence)}")

            except Exception as e:
                stats["errors"] += 1
                audit_log(
                    run_id=run_id,
                    agent_name=agent_name,
                    action="coordinator_error",
                    ref_type="agent_event",
                    ref_id=ev_id,
                    detail={"error": str(e)[:800]},
                )

        # Mark processed events so reruns do not duplicate outputs
        mark_agent_events_processed(processed_ids)

        finish_agent_run(run_id, status="success", stats=stats)
        return 0

    except Exception as e:
        finish_agent_run(run_id, status="failed", stats={**stats, "fatal": str(e)[:800]})
        raise


if __name__ == "__main__":
    raise SystemExit(main())