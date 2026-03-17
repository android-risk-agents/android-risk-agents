# src/db.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from supabase import create_client

from .config import (
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
    validate_env,
    VECTOR_TABLE,
    VECTOR_RPC_MATCH,
)


def get_supabase_client():
    validate_env()
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_first(data: Any) -> Optional[Dict[str, Any]]:
    if isinstance(data, list) and len(data) > 0:
        return data[0]
    return None


# ---------------------------
# Sources
# ---------------------------

def get_source_by_id(source_id: int) -> Optional[Dict[str, Any]]:
    sb = get_supabase_client()
    resp = (
        sb.table("sources")
        .select("id, name, url, source_type, priority, active")
        .eq("id", int(source_id))
        .limit(1)
        .execute()
    )
    return _safe_first(resp.data)


def get_source_url(source_id: int) -> str:
    row = get_source_by_id(source_id)
    if not row:
        return ""
    return row.get("url") or ""


# ---------------------------
# Snapshots
# ---------------------------

def get_snapshot_text_by_id(snapshot_id: int) -> str:
    sb = get_supabase_client()
    resp = (
        sb.table("snapshots")
        .select("clean_text")
        .eq("id", int(snapshot_id))
        .limit(1)
        .execute()
    )
    row = _safe_first(resp.data)
    if not row:
        return ""
    return row.get("clean_text") or ""


def get_snapshot_text_and_hash_by_id(snapshot_id: int) -> Tuple[str, str]:
    sb = get_supabase_client()
    resp = (
        sb.table("snapshots")
        .select("clean_text, content_hash")
        .eq("id", int(snapshot_id))
        .limit(1)
        .execute()
    )
    row = _safe_first(resp.data) or {}
    return (row.get("clean_text") or "", row.get("content_hash") or "")


def get_latest_snapshot_for_source(source_id: int) -> Optional[Dict[str, Any]]:
    sb = get_supabase_client()
    resp = (
        sb.table("snapshots")
        .select("id, content_hash, fetched_at")
        .eq("source_id", int(source_id))
        .order("fetched_at", desc=True)
        .limit(1)
        .execute()
    )
    return _safe_first(resp.data)


# ---------------------------
# Changes
# ---------------------------

def get_pending_changes_for_triage(limit: int = 25) -> List[Dict[str, Any]]:
    sb = get_supabase_client()
    resp = (
        sb.table("changes")
        .select("id, source_id, prev_snapshot_id, new_snapshot_id, diff_json, created_at, status")
        .or_("status.is.null,status.eq.new,status.eq.pending")
        .order("created_at", desc=True)
        .limit(int(limit))
        .execute()
    )
    return resp.data or []


def update_change_triage_fields(
    change_id: int,
    status: str,
    relevance_score: int,
    local_risk_score: int,
    tags: List[str],
) -> None:
    sb = get_supabase_client()
    sb.table("changes").update(
        {
            "status": str(status),
            "triaged_at": _utc_now_iso(),
            "relevance_score": int(relevance_score),
            "local_risk_score": int(local_risk_score),
            "tags": tags,
        }
    ).eq("id", int(change_id)).execute()


def update_change_classification_fields(
    change_id: int,
    risk_category: str,
    risk_bucket: str,
    similarity_score: float,
    classification_method: str,
) -> None:
    """
    Write the deterministic classification results to the changes table.
    Called before update_change_triage_fields so classification data is
    always persisted even if the LLM rationale step later fails.
    Requires the following columns to exist on the changes table:
      risk_category TEXT, risk_bucket TEXT,
      similarity_score NUMERIC(5,4), classification_method TEXT
    """
    sb = get_supabase_client()
    sb.table("changes").update(
        {
            "risk_category":         str(risk_category),
            "risk_bucket":           str(risk_bucket),
            "similarity_score":      float(similarity_score),
            "classification_method": str(classification_method),
        }
    ).eq("id", int(change_id)).execute()


def mark_change_analyzed(change_id: int) -> None:
    sb = get_supabase_client()
    sb.table("changes").update(
        {
            "status": "analyzed",
            "analyzed_at": _utc_now_iso(),
        }
    ).eq("id", int(change_id)).execute()


# ---------------------------
# Vector chunks (pgvector)
# ---------------------------

def get_snapshot_embeddings(snapshot_id: int) -> List[List[float]]:
    """
    Fetch all chunk embeddings stored in vector_chunks for a given snapshot_id.
    Returns a list of embedding vectors (each a list of floats).
    Used by the sentinel triage agent to build a document-level embedding by
    averaging chunk embeddings - avoids re-embedding already-stored content.
    Returns empty list if no chunks found (e.g. snapshot was skipped by embedder).
    """
    sb = get_supabase_client()
    resp = (
        sb.table(VECTOR_TABLE)
        .select("embedding")
        .eq("snapshot_id", int(snapshot_id))
        .order("chunk_index")
        .execute()
    )
    rows = resp.data or []
    return [row["embedding"] for row in rows if row.get("embedding")]


def upsert_vector_chunks(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    sb = get_supabase_client()
    sb.table(VECTOR_TABLE).upsert(
        rows,
        on_conflict="source_id,snapshot_sha,kind,chunk_index",
    ).execute()


def vector_search(
    query_embedding: List[float],
    match_count: int = 8,
    filter_source_id: Optional[str] = None,
    filter_kind: Optional[str] = None,
):
    sb = get_supabase_client()
    payload: Dict[str, Any] = {
        "query_embedding": query_embedding,
        "match_count": int(match_count),
        "filter_source_id": filter_source_id,
        "filter_kind": filter_kind,
    }
    return sb.rpc(VECTOR_RPC_MATCH, payload).execute()


# ---------------------------
# Agent runs + audit
# ---------------------------

def create_agent_run(run_name: str, trigger: str, llm_backend: str) -> int:
    sb = get_supabase_client()
    payload = {
        "run_name": str(run_name),
        "trigger": str(trigger),
        "llm_backend": str(llm_backend),
        "started_at": _utc_now_iso(),
        "status": "running",
        "stats": {},
    }
    resp = sb.table("agent_runs").insert(payload).execute()
    row = _safe_first(resp.data) or {}
    rid = int(row.get("id") or 0)
    if rid <= 0:
        raise RuntimeError("agent_runs insert did not return id")
    return rid


def finish_agent_run(run_id: int, status: str, stats: Dict[str, Any]) -> None:
    sb = get_supabase_client()
    sb.table("agent_runs").update(
        {
            "finished_at": _utc_now_iso(),
            "status": str(status),
            "stats": stats or {},
        }
    ).eq("id", int(run_id)).execute()


def audit_log(
    run_id: int,
    agent_name: str,
    action: str,
    ref_type: str,
    ref_id: int,
    detail: Dict[str, Any],
) -> None:
    sb = get_supabase_client()
    sb.table("agent_audit_log").insert(
        {
            "run_id": int(run_id),
            "agent_name": str(agent_name),
            "action": str(action),
            "ref_type": str(ref_type),
            "ref_id": int(ref_id),
            "detail": detail or {},
            "created_at": _utc_now_iso(),
        }
    ).execute()


# ---------------------------
# Agent events
# ---------------------------

def insert_agent_event(payload: Dict[str, Any]) -> int:
    sb = get_supabase_client()
    resp = sb.table("agent_events").insert(payload).execute()
    row = _safe_first(resp.data) or {}
    return int(row.get("id") or 0)


def get_pending_agent_events(limit: int = 50) -> List[Dict[str, Any]]:
    sb = get_supabase_client()
    resp = (
        sb.table("agent_events")
        .select(
            "id, source_id, snapshot_id, change_id, agent_name, event_type, title, summary, tags, relevance_score, local_risk_score, status, created_at"
        )
        .or_("status.is.null,status.eq.new,status.eq.pending")
        .order("created_at", desc=True)
        .limit(int(limit))
        .execute()
    )
    return resp.data or []


def mark_agent_events_processed(event_ids: List[int]) -> None:
    if not event_ids:
        return
    sb = get_supabase_client()
    sb.table("agent_events").update(
        {"status": "processed"}
    ).in_("id", [int(x) for x in event_ids]).execute()


# ---------------------------
# Insights (includes source_id + snapshot_id)
# ---------------------------

def insert_insight(
    change_id: int,
    source_id: int,
    snapshot_id: int,
    agent_name: str,
    title: str,
    summary: str,
    confidence: float,
    category: Optional[str],
    affected_signals: Optional[List[str]],
    recommended_actions: Optional[List[str]],
    risk_score: int,
) -> None:
    sb = get_supabase_client()
    payload = {
        "change_id": int(change_id),
        "source_id": int(source_id),
        "snapshot_id": int(snapshot_id),
        "agent_name": str(agent_name),
        "title": str(title)[:180],
        "summary": str(summary)[:2000],
        "category": (str(category)[:120] if category else None),
        "affected_signals": affected_signals if affected_signals is not None else [],
        "recommended_actions": recommended_actions if recommended_actions is not None else [],
        "confidence": float(confidence),
        "risk_score": int(risk_score),
        "created_at": _utc_now_iso(),
    }
    sb.table("insights").insert(payload).execute()


# ---------------------------
# Recommendations (matches your schema)
# ---------------------------

def insert_recommendation(
    run_id: int,
    title: str,
    priority: str,
    final_risk_score: int,
    confidence: float,
    event_ids: List[int],
    change_ids: List[int],
    rationale: str,
    recommended_actions: List[str],
    related_tags: List[str],
) -> None:
    sb = get_supabase_client()
    payload = {
        "run_id": int(run_id),
        "title": str(title)[:180],
        "priority": str(priority)[:8],  # P0 / P1 / P2
        "final_risk_score": int(final_risk_score),
        "confidence": float(confidence),
        "event_ids": event_ids,
        "change_ids": change_ids,
        "rationale": str(rationale)[:4000],
        "recommended_actions": recommended_actions[:10],
        "related_tags": related_tags[:12],
        "created_at": _utc_now_iso(),
    }
    sb.table("recommendations").insert(payload).execute()