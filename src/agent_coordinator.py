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
    "You are the Coordinator agent for a digital fraud and risk intelligence system.\n"
    "Your job is to translate platform ecosystem updates into detailed, implementable actions for risk monitoring and risk models.\n"
    "Do not invent facts. If something is unknown, say unknown.\n"
    "Ground your answer in the provided CONTEXT snippets.\n"
    "Avoid generic advice. Each recommendation must reference a specific signal, policy section, enforcement mechanism, API, telemetry event,\n"
    "or vulnerability theme mentioned in the context.\n\n"
    "OUTPUT CONTRACT (STRICT):\n"
    "- Output EXACTLY one JSON object wrapped between markers.\n"
    "- Do not output code fences. Do not output any text outside the markers.\n"
    "- Use double quotes for all keys and all string values.\n"
    "- No trailing commas.\n"
    "- Never output the token sequence \",:\" anywhere.\n"
    "- Do not include raw newlines inside JSON string values.\n"
    "  For fields that require bullets, return JSON ARRAYS of strings (not a single string with \\n).\n\n"
    "Markers:\n"
    "<<<JSON>>>\n"
    "{...}\n"
    "<<<ENDJSON>>>\n\n"
    "Requirements:\n"
    "- summary MUST be 6 to 8 bullets.\n"
    "- recommended_actions MUST be 6 to 8 bullets, each is implementable.\n"
    "- Include evidence_snippets: 2 to 4 short quotes from context (<=200 chars each).\n\n"
    "Return ONLY valid JSON inside the markers.\n"
    "Do not add extra keys.\n\n"
    "If you cannot comply perfectly, output this fallback JSON inside the markers:\n"
    "<<<JSON>>>\n"
    "{\"title\":\"\",\"summary\":[],\"category\":\"other\",\"affected_signals\":[],"
    "\"recommended_actions\":[],\"risk_score\":2,\"confidence\":0.65,\"evidence_snippets\":[]}\n"
    "<<<ENDJSON>>>"
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
    # IMPORTANT: Keep schema aligned with how we parse later (lists, not newline-joined strings).
    schema = {
        "title": "string (specific, include feature or policy name when possible)",
        "summary": ["string (6-8 bullets, each starts with '-')"],
        "category": "string (vulnerability_intel | policy_intel | api_signal_change | enforcement_change | telemetry_change | other)",
        "affected_signals": ["string (up to 8, specific signals or controls)"],
        "recommended_actions": ["string (6-8 bullets, each starts with '-')"],
        "risk_score": "integer 1-5",
        "confidence": "number 0-1",
        "evidence_snippets": ["string (2-4 quotes from context, <=200 chars each)"],
    }

    mode = "BASELINE_INTELLIGENCE" if baseline else "UPDATE_INTELLIGENCE"

    return (
        f"MODE: {mode}\n"
        f"SOURCE: {url}\n\n"
        f"EVENT TITLE: {title}\n"
        f"EVENT SUMMARY: {summary}\n\n"
        "Instructions:\n"
        "- Use CONTEXT snippets as evidence.\n"
        "- If MODE is BASELINE_INTELLIGENCE: extract the key controls, policies, signals, and enforcement points present now. "
        "Propose monitoring and validation steps to detect future drift.\n"
        "- If MODE is UPDATE_INTELLIGENCE: explain what changed and how it could affect monitoring coverage and risk decisions.\n"
        "- Make recommendations implementable. Prefer concrete checks, alerts, rules, thresholds, and validation steps.\n\n"
        # Smaller context reduces drift and truncation.
        f"CONTEXT (RAG snippets):\n{context[:4500]}\n\n"
        "Return JSON only inside the markers described in the system instructions.\n"
        "Do not add extra keys.\n"
        f"Schema:\n{schema}"
    )


def rag_context_from_event(title: str, summary: str, text: str, top_k: int) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    # Smaller query reduces irrelevant retrieval while keeping intent.
    q = (f"{title}\n{summary}\n{t[:1000]}").strip()[:1500]
    q_emb = embed_texts([q])[0]

    resp = vector_search(query_embedding=q_emb, match_count=top_k, filter_source_id=None, filter_kind=None)
    rows = resp.data or []

    out: List[str] = []
    for r in rows[:top_k]:
        chunk = (r.get("chunk_text") or "").strip()
        if chunk:
            out.append(chunk[:700])

    return "\n\n---\n\n".join(out)


def main() -> int:
    agent_name = os.getenv("AGENT_NAME", "coordinator")

    # Stronger default model for JSON contract adherence
    model = os.getenv("MODEL_ANALYZE", "meta/llama3-70b-instruct")

    # Slightly lower retrieval breadth to reduce context size
    top_k = int(os.getenv("RAG_TOP_K", "4"))

    max_recs = int(os.getenv("MAX_RECOMMENDATIONS", "8"))

    debug = str(os.getenv("DEBUG_LLM", "")).strip().lower() in ("1", "true", "yes", "y")

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

            title = str(ev.get("title") or "Update detected")[:220]
            ev_summary = str(ev.get("summary") or "")[:2500]
            tags = _tags(ev)

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
                    max_tokens=900,
                    request_id=f"event={ev_id}",
                )

                affected = _as_list_str(out.get("affected_signals"), max_items=8, max_len=160) or []
                actions = _as_list_str(out.get("recommended_actions"), max_items=8, max_len=280) or []
                evidence = _as_list_str(out.get("evidence_snippets"), max_items=4, max_len=200) or []

                risk_score = _as_int(out.get("risk_score"), 2)
                confidence = _as_float(out.get("confidence"), 0.65)

                insight_title = str(out.get("title") or title)[:240]

                # summary is now expected as list[str]; if model returns string, fall back safely
                summary_list = _as_list_str(out.get("summary"), max_items=8, max_len=420)
                if summary_list:
                    insight_summary = "\n".join(summary_list)[:6000]
                else:
                    insight_summary = str(out.get("summary") or ev_summary)[:6000]

                category = out.get("category")
                category = str(category).strip()[:140] if category else None

                final_risk = _as_int(ev.get("local_risk_score"), 50)
                priority = "P0" if final_risk >= 85 else ("P1" if final_risk >= 70 else "P2")

                rationale_parts: List[str] = [insight_summary]
                if evidence:
                    rationale_parts.append("Evidence:")
                    for q in evidence[:4]:
                        rationale_parts.append(f"- {q}")
                rationale = "\n".join(rationale_parts)[:8000]

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
                    rationale=rationale,
                    recommended_actions=actions
                    or [
                        "- Identify which signals or controls are impacted and update monitoring coverage accordingly",
                        "- Add a validation check to confirm required fields are still present and correctly parsed",
                        "- Create an alert for drift in enforcement language or integrity requirements",
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
                    print(
                        f"[COORD] event_id={ev_id} wrote priority={priority} actions={len(actions)} evidence={len(evidence)}"
                    )

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

        mark_agent_events_processed(processed_ids)

        finish_agent_run(run_id, status="success", stats=stats)
        return 0

    except Exception as e:
        finish_agent_run(run_id, status="failed", stats={**stats, "fatal": str(e)[:800]})
        raise


if __name__ == "__main__":
    raise SystemExit(main())