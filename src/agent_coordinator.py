# src/agent_coordinator.py
import os
import json
import re
from typing import Any, Dict, List, Optional

from supabase import create_client

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
    "You are the Coordinator AI agent for a digital fraud risk intelligence system. "
    "You prioritize platform ecosystem updates and translate them into concrete actions for risk monitoring and risk models. "
    "Do not invent facts. If something is unknown, say unknown. "
    "Return ONLY valid JSON, no markdown."
)

SYSTEM_ANALYZE_STRICT = (
    SYSTEM_ANALYZE
    + " CRITICAL: Output exactly ONE JSON object. No code fences. No extra text. "
      "Make JSON safe to parse: do not include raw control characters. "
      "If you need new lines inside strings, escape them as \\n."
)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_CTRL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def _debug_enabled() -> bool:
    return os.getenv("DEBUG_LLM", "false").strip().lower() == "true"


def _debug(msg: str) -> None:
    if _debug_enabled():
        print(msg)


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
        s = str(item).strip().lower()
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
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = _CTRL_CHARS_RE.sub("", s)
    return s


def _extract_first_json_value(s: str) -> Any:
    s = (s or "").strip()
    if not s:
        raise ValueError("Empty LLM output")

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
    return chat_json(
        client=client,
        model=model,
        system=system,
        user=user,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _csv_env(name: str, default: str = "") -> List[str]:
    raw = os.getenv(name, default)
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def build_deep_insight_prompt(
    url: str,
    title: str,
    summary: str,
    context: str,
    fingerprint_context: str = "",
) -> str:
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

    fp_section = (
        f"\nFINGERPRINT TECHNICAL EVIDENCE:\n{fingerprint_context[:3500]}\n"
        if fingerprint_context.strip()
        else "\nFINGERPRINT TECHNICAL EVIDENCE:\nNone available.\n"
    )

    return (
        f"SOURCE: {url}\n\n"
        f"EVENT TITLE: {title}\n"
        f"EVENT SUMMARY: {summary}\n\n"
        f"GENERAL CONTEXT (RAG snippets):\n{context[:6500]}\n"
        f"{fp_section}\n"
        "Return JSON only. Do not include markdown.\n"
        "Make recommendations specific to security notes and risk models: "
        "signals or telemetry to monitor, rules or features to update, tests to add, and what to validate.\n"
        "If Fingerprint technical evidence is available, use it to make recommendations more concrete. "
        "For example, reference identifier providers, signal collection logic, fallback behavior, "
        "device profiling, emulator or tamper signals, integrity-related checks, or related SDK modules.\n"
        "Do not overclaim. If evidence is weak, recommend validation, monitoring, or engineering review.\n"
        f"Schema:\n{schema}"
    )


def rag_context_from_text(text: str, top_k: int) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    q = t[:1200]
    q_emb = embed_texts([q], is_query=True)[0]

    resp = vector_search(
        query_embedding=q_emb,
        match_count=top_k,
        filter_source_id=None,
        filter_kind=None,
    )
    rows = resp.data or []

    out: List[str] = []
    for r in rows[:top_k]:
        chunk = (r.get("chunk_text") or "").strip()
        if chunk:
            out.append(chunk[:900])

    return "\n\n---\n\n".join(out)


def _get_supabase_client():
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is missing")
    return create_client(url, key)


def should_use_fingerprint(ev: Dict[str, Any]) -> bool:
    if not _bool_env("FINGERPRINT_ENABLED", True):
        return False

    min_local_risk = _as_int(os.getenv("FINGERPRINT_MIN_LOCAL_RISK", "40"), 40)
    min_relevance = _as_int(os.getenv("FINGERPRINT_MIN_RELEVANCE", "40"), 40)

    local_risk = _as_int(ev.get("local_risk_score"), 0)
    relevance = _as_int(ev.get("relevance_score"), 0)
    tags = _tags(ev)
    title = str(ev.get("title") or "").lower()
    summary = str(ev.get("summary") or "").lower()

    if local_risk >= min_local_risk:
        return True
    if relevance >= min_relevance:
        return True

    interesting_tags = set(
        _csv_env(
            "FINGERPRINT_TAG_MATCHES",
            "identifier,device,signal,signals,integrity,permission,permissions,"
            "sdk,api,root,emulator,attestation,privacy,telemetry,network,auth,"
            "security,policy,bulletin,vulnerability,patch",
        )
    )

    if any(t in interesting_tags for t in tags):
        return True

    keyword_text = f"{title} {summary}"
    keyword_matches = [
        "android id",
        "device id",
        "identifier",
        "signal",
        "integrity",
        "permission",
        "attestation",
        "sdk",
        "emulator",
        "root",
        "tamper",
        "fraud",
        "telemetry",
        "play integrity",
        "fingerprinting",
        "device profile",
        "security bulletin",
        "vulnerability",
        "policy",
        "patch",
    ]
    return any(k in keyword_text for k in keyword_matches)


def build_fingerprint_query(title: str, summary: str, tags: List[str]) -> str:
    bits: List[str] = []
    if title.strip():
        bits.append(title.strip())
    if summary.strip():
        bits.append(summary.strip())
    if tags:
        bits.append("Tags: " + ", ".join(tags[:8]))

    bits.append(
        "Focus on device identifiers, signal collection, fallback logic, "
        "device profiling, emulator detection, root or tamper signals, "
        "permissions, integrity checks, telemetry, SDK behavior, and fraud relevance."
    )
    q = "\n".join(bits).strip()
    return q[:1800]


def _fingerprint_vector_search(query_embedding: List[float], match_count: int) -> List[Dict[str, Any]]:
    rpc_name = os.getenv("FINGERPRINT_VECTOR_RPC", "match_fingerprint_library_chunks")
    client = _get_supabase_client()

    try:
        resp = client.rpc(
            rpc_name,
            {
                "query_embedding": query_embedding,
                "match_count": match_count,
            },
        ).execute()
        rows = resp.data or []
        _debug(f"[FP] rpc={rpc_name} rows_returned={len(rows)}")
        return rows
    except Exception as e:
        _debug(f"[FP] rpc={rpc_name} error={str(e)[:500]}")
        return []


def fingerprint_context_from_event(title: str, summary: str, tags: List[str], top_k: int = 4) -> str:
    query = build_fingerprint_query(title=title, summary=summary, tags=tags)
    if not query.strip():
        return ""

    _debug(f"[FP] query_head={query[:300]}")

    q_emb = embed_texts([query], is_query=True)[0]
    rows = _fingerprint_vector_search(query_embedding=q_emb, match_count=top_k)

    out: List[str] = []
    for idx, r in enumerate(rows[:top_k], start=1):
        chunk = str(r.get("chunk_text") or "").strip()
        if not chunk:
            continue

        file_name = (
            r.get("file_name")
            or r.get("path")
            or r.get("file_path")
            or r.get("relative_path")
            or "unknown_file"
        )
        topic = str(r.get("topic") or r.get("signal_family") or "").strip()
        summary_text = str(r.get("summary") or r.get("file_summary") or "").strip()
        score = r.get("similarity") or r.get("score")

        header_parts = [f"[Fingerprint evidence {idx}] file={file_name}"]
        if topic:
            header_parts.append(f"topic={topic}")
        if score is not None:
            try:
                header_parts.append(f"score={float(score):.4f}")
            except Exception:
                pass

        section = " | ".join(header_parts)

        body_parts: List[str] = []
        if summary_text:
            body_parts.append(f"Summary: {summary_text[:300]}")
        body_parts.append(f"Chunk: {chunk[:900]}")

        out.append(section + "\n" + "\n".join(body_parts))

    joined = "\n\n---\n\n".join(out)
    if joined.strip():
        _debug(f"[FP] context_used_head={joined[:700]}")
    else:
        _debug("[FP] no fingerprint context returned")
    return joined


def _default_actions(final_risk: int, has_fingerprint: bool) -> List[str]:
    base = [
        "Add or adjust monitoring on impacted signals mentioned in the update",
        "Update feature definitions or rules if the update changes data availability or integrity checks",
        "Add regression tests and alerting for the impacted signal paths",
    ]
    if has_fingerprint:
        base.insert(
            0,
            "Review the Fingerprint-related modules surfaced in retrieval and validate whether identifier, device profiling, or fallback logic is affected",
        )
    if final_risk >= 85:
        base.append("Escalate to engineering and fraud analytics for priority validation before the next production cycle")
    return base[:8]


def main() -> int:
    agent_name = os.getenv("AGENT_NAME", "coordinator")
    model = os.getenv("MODEL_ANALYZE", os.getenv("GROQ_MODEL_ANALYZE", "llama-3.3-70b-versatile"))
    top_k = int(os.getenv("RAG_TOP_K", "6"))
    max_recs = int(os.getenv("MAX_RECOMMENDATIONS", "8"))
    fingerprint_top_k = int(os.getenv("FINGERPRINT_TOP_K", "4"))

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
        "fingerprint_attempted": 0,
        "fingerprint_used": 0,
    }

    try:
        events = get_pending_agent_events(limit=60)
        if not events:
            finish_agent_run(run_id, status="success", stats={**stats, "note": "no pending events"})
            return 0

        stats["events_seen"] = len(events)
        _debug(f"[COORD] pending_events={len(events)}")

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

                fingerprint_context = ""
                use_fp = should_use_fingerprint(ev)

                _debug(
                    f"[FP] event_id={ev_id} use_fp={use_fp} "
                    f"local_risk={ev.get('local_risk_score')} "
                    f"relevance={ev.get('relevance_score')} tags={tags}"
                )

                if use_fp:
                    stats["fingerprint_attempted"] += 1
                    fingerprint_context = fingerprint_context_from_event(
                        title=title,
                        summary=summary,
                        tags=tags,
                        top_k=fingerprint_top_k,
                    )
                    if fingerprint_context.strip():
                        stats["fingerprint_used"] += 1

                prompt = build_deep_insight_prompt(
                    url=url,
                    title=title,
                    summary=summary,
                    context=context,
                    fingerprint_context=fingerprint_context,
                )

                try:
                    out = _chat_json_coordinator(
                        client=client,
                        model=model,
                        system=SYSTEM_ANALYZE,
                        user=prompt,
                        temperature=0.2,
                        max_tokens=800,
                    )
                except Exception:
                    stats["parse_retries"] += 1
                    out = _chat_json_coordinator(
                        client=client,
                        model=model,
                        system=SYSTEM_ANALYZE_STRICT,
                        user=prompt + "\n\nIMPORTANT: Output a single JSON object only.",
                        temperature=0.0,
                        max_tokens=800,
                    )

                affected = _as_list_str(out.get("affected_signals"), max_items=6, max_len=140) or []
                actions = _as_list_str(out.get("recommended_actions"), max_items=8, max_len=220) or []
                risk_score = _as_int(out.get("risk_score"), 2)
                confidence = _as_float(out.get("confidence"), 0.6)

                insight_title = str(out.get("title") or title)[:180]
                insight_summary = str(out.get("summary") or summary)[:2000]
                category = out.get("category")
                category = str(category).strip()[:120] if category else None

                final_risk = _as_int(ev.get("local_risk_score"), 50)
                priority = "P0" if final_risk >= 85 else ("P1" if final_risk >= 70 else "P2")

                final_actions = actions or _default_actions(
                    final_risk=final_risk,
                    has_fingerprint=bool(fingerprint_context.strip()),
                )

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
                        recommended_actions=final_actions,
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
                    recommended_actions=final_actions,
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
                        "fingerprint_used": bool(fingerprint_context.strip()),
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

        mark_agent_events_processed(processed_ids)

        _debug(f"[COORD] stats={json.dumps(stats)}")
        finish_agent_run(run_id, status="success", stats=stats)
        return 0

    except Exception as e:
        finish_agent_run(run_id, status="failed", stats={**stats, "fatal": str(e)[:800]})
        raise


if __name__ == "__main__":
    raise SystemExit(main())