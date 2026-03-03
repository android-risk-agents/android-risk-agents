# src/agent_coordinator.py
import os
import json
import re
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
from .llm_client import get_llm_client, chat_json  # ✅ CHANGED: import chat_json


SYSTEM_ANALYZE = (
    "You are the Coordinator AI agent for a digital fraud risk intelligence system. "
    "You prioritize platform ecosystem updates and translate them into concrete actions for risk monitoring and risk models. "
    "Do not invent facts. If something is unknown, say unknown. "
    "Return ONLY valid JSON, no markdown."
)

# Stricter system message used only when we detect parse issues
SYSTEM_ANALYZE_STRICT = (
    SYSTEM_ANALYZE
    + " CRITICAL: Output exactly ONE JSON object. No code fences. No extra text. "
      "Make JSON safe to parse: do not include raw control characters. "
      "If you need new lines inside strings, escape them as \\n."
)


# --- helpers ---

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_CTRL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")  # remove illegal JSON control chars


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


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = _CODE_FENCE_RE.sub("", s).strip()
    return s


def _sanitize_control_chars(s: str) -> str:
    # normalize newlines, then remove illegal control characters
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = _CTRL_CHARS_RE.sub("", s)
    return s


def _extract_first_json_value(s: str) -> Any:
    """
    Extract the first JSON value from a string (object or array) even if extra text exists.
    """
    s = (s or "").strip()
    if not s:
        raise ValueError("Empty LLM output")

    # If response starts with junk, find first { or [
    if not s.startswith("{") and not s.startswith("["):
        i1 = s.find("{")
        i2 = s.find("[")
        candidates = [i for i in (i1, i2) if i != -1]
        if not candidates:
            raise ValueError("No JSON start bracket found")
        s = s[min(candidates):]

    decoder = json.JSONDecoder()
    obj, _idx = decoder.raw_decode(s)
    return obj


def _safe_parse_json(content: str) -> Dict[str, Any]:
    """
    Parse coordinator output robustly:
    - strip code fences
    - sanitize illegal control chars
    - raw_decode first JSON value
    - unwrap [ {..} ] to {..}
    """
    content = _strip_code_fences(content)
    content = _sanitize_control_chars(content)

    obj = _extract_first_json_value(content)

    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        obj = obj[0]

    if not isinstance(obj, dict):
        raise ValueError(f"Parsed JSON is not an object (got {type(obj)}).")

    return obj


def _chat_json_coordinator(
    client,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """
    Coordinator-local chat_json to avoid 'Invalid control character' crashes.

    ✅ CHANGED: Instead of OpenAI/Groq-style client.chat.completions.create(),
    call the shared NIM wrapper chat_json() from llm_client.py.
    """
    return chat_json(
        client=client,
        model=model,
        system=system,
        user=user,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def build_deep_insight_prompt(url: str, title: str, summary: str, context: str) -> str:
    schema = {
        "title": "string",
        "summary": "string (3-6 sentences, concrete and actionable, security/risk-model focused)",
        "category": "string (optional)",
        "affected_signals": ["string (up to 6)"],
        "recommended_actions": [
            "string (actionable steps for risk monitoring / risk models, up to 8; be specific)"
        ],
        "risk_score": "integer 1-5",
        "confidence": "number 0-1",
    }
    return (
        f"SOURCE: {url}\n\n"
        f"EVENT TITLE: {title}\n"
        f"EVENT SUMMARY: {summary}\n\n"
        f"CONTEXT (RAG snippets):\n{context[:6500]}\n\n"
        "Return JSON only. Do not include markdown.\n"
        "Make recommendations specific to security notes and risk models: "
        "signals/telemetry to monitor, rules/features to update, tests to add, and what to validate.\n"
        f"Schema:\n{schema}"
    )


def rag_context_from_text(text: str, top_k: int) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    # Use is_query=True so Nomic uses search_query prefix
    q = t[:1200]
    q_emb = embed_texts([q], is_query=True)[0]

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
    model = os.getenv("MODEL_ANALYZE", os.getenv("GROQ_MODEL_ANALYZE", "llama-3.3-70b-versatile"))
    top_k = int(os.getenv("RAG_TOP_K", "6"))
    max_recs = int(os.getenv("MAX_RECOMMENDATIONS", "8"))

    client = get_llm_client()

    run_id = create_agent_run(
        run_name="coordinator",
        trigger="workflow_dispatch",
        llm_backend=os.getenv("LLM_BASE_URL", "default"),
    )

    stats = {
        "events_seen": 0,
        "events_processed": 0,
        "recommendations_written": 0,
        "insights_written": 0,
        "errors": 0,
        "parse_retries": 0,
    }

    try:
        events = get_pending_agent_events(limit=60)
        if not events:
            finish_agent_run(run_id, status="success", stats={**stats, "note": "no pending events"})
            return 0

        stats["events_seen"] = len(events)

        # v1 prioritization: sort by local_risk_score then relevance_score
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

                # First attempt: normal system
                try:
                    out = _chat_json_coordinator(
                        client=client,
                        model=model,
                        system=SYSTEM_ANALYZE,
                        user=prompt,
                        temperature=0.2,
                        max_tokens=650,
                    )
                except Exception as e:
                    # Second attempt: stricter system to reduce parse issues
                    stats["parse_retries"] += 1
                    out = _chat_json_coordinator(
                        client=client,
                        model=model,
                        system=SYSTEM_ANALYZE_STRICT,
                        user=prompt + "\n\nIMPORTANT: Output a single JSON object only.",
                        temperature=0.0,
                        max_tokens=650,
                    )

                affected = _as_list_str(out.get("affected_signals"), max_items=6, max_len=140) or []
                actions = _as_list_str(out.get("recommended_actions"), max_items=8, max_len=220) or []
                risk_score = _as_int(out.get("risk_score"), 2)
                confidence = _as_float(out.get("confidence"), 0.6)

                insight_title = str(out.get("title") or title)[:180]
                insight_summary = str(out.get("summary") or summary)[:2000]
                category = out.get("category")
                category = str(category).strip()[:120] if category else None

                # final risk for recommendation is 0-100 scale from event
                final_risk = _as_int(ev.get("local_risk_score"), 50)
                priority = "P0" if final_risk >= 85 else ("P1" if final_risk >= 70 else "P2")

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
                    rationale=insight_summary,
                    recommended_actions=actions
                    or [
                        "Add/adjust monitoring on impacted signals mentioned in the update",
                        "Update feature definitions or rules if the update changes data availability or integrity checks",
                        "Add regression tests and alerting for the impacted signal paths",
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