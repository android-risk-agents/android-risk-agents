import os
from typing import Any, Dict, List

from .db import (
    create_agent_run,
    finish_agent_run,
    audit_log,
    get_pending_changes_for_triage,
    get_snapshot_text_by_id,
    update_change_triage_fields,
    insert_agent_event,
    get_source_url,
)
from .nim_client import NimClient


SYSTEM_TRIAGE = (
    "You are a Sentinel triage AI agent for a digital fraud risk team. "
    "Given OLD vs NEW platform text, decide if this update is relevant to digital fraud risk monitoring. "
    "Focus on platform capability changes, identity or device signals, policy enforcement, telemetry, SDK changes, and attacker opportunities. "
    "Be strict. Prefer false negatives if uncertain. "
    "Return ONLY valid JSON, no markdown."
)


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _as_tags(x: Any, max_items: int = 8) -> List[str]:
    if not isinstance(x, list):
        return []
    out: List[str] = []
    for item in x:
        s = str(item).strip()
        if not s:
            continue
        out.append(s[:60])
        if len(out) >= max_items:
            break
    return out


def _is_baseline_init(change_row: Dict[str, Any]) -> bool:
    dj = change_row.get("diff_json") or {}
    if isinstance(dj, dict) and str(dj.get("type", "")).lower() == "baseline_init":
        return True
    return change_row.get("prev_snapshot_id") is None


def build_prompt(old_text: str, new_text: str, url: str, baseline: bool) -> str:
    schema = {
        "is_relevant": "boolean",
        "relevance_score": "integer 0-100",
        "local_risk_score": "integer 0-100 (impact to fraud signals)",
        "event_type": "string (policy_change | api_signal_change | security_update | enforcement_change | baseline_init | other)",
        "title": "string (short)",
        "summary": "string (1-3 sentences, concrete)",
        "tags": ["string (short tags)"],
        "what_changed_hint": "string (short hint)",
    }

    header = "BASELINE INITIAL INGESTION (no previous snapshot)" if baseline else "DIFF UPDATE (previous vs new)"

    return (
        f"{header}\n"
        f"SOURCE: {url}\n\n"
        f"OLD (trimmed):\n{(old_text or '')[:4500]}\n\n"
        f"NEW (trimmed):\n{(new_text or '')[:4500]}\n\n"
        "Return JSON only.\n"
        f"Schema:\n{schema}"
    )


def main() -> int:
    agent_name = os.getenv("AGENT_NAME", "sentinel-triage")
    threshold = int(os.getenv("RELEVANCE_THRESHOLD", "70"))
    model = os.getenv("MODEL_TRIAGE", "meta/llama3-8b-instruct")

    # Baseline behavior toggle for testing
    # If "true": always create triage events for baseline-init changes.
    baseline_force = os.getenv("BASELINE_FORCE_TRIAGE", "true").strip().lower() in ("1", "true", "yes")

    debug = os.getenv("DEBUG_LLM", "").strip() in ("1", "true", "TRUE", "yes", "YES")

    client = NimClient()

    run_id = create_agent_run(
        run_name="sentinel-triage",
        trigger="workflow_dispatch",
        llm_backend="nvidia-nim",
    )

    stats = {"triaged": 0, "ignored": 0, "events_created": 0, "errors": 0}

    try:
        changes = get_pending_changes_for_triage(limit=25)
        if not changes:
            finish_agent_run(run_id, status="success", stats={**stats, "note": "no pending changes"})
            return 0

        for ch in changes:
            change_id = int(ch["id"])
            source_id = int(ch["source_id"])
            old_id = ch.get("prev_snapshot_id")
            new_id = ch.get("new_snapshot_id")

            baseline = _is_baseline_init(ch)

            old_text = "" if baseline else (get_snapshot_text_by_id(int(old_id)) if old_id else "")
            new_text = get_snapshot_text_by_id(int(new_id)) if new_id else ""

            url = get_source_url(source_id) or f"source_id={source_id}"

            try:
                prompt = build_prompt(old_text, new_text, url, baseline=baseline)

                out = client.chat_json(
                    model=model,
                    system=SYSTEM_TRIAGE,
                    user=prompt,
                    temperature=0.0,
                    max_tokens=320,
                    request_id=f"change={change_id}",
                )

                is_rel = bool(out.get("is_relevant", False))
                rel_score = _as_int(out.get("relevance_score", 0), 0)
                local_risk = _as_int(out.get("local_risk_score", 0), 0)
                tags = _as_tags(out.get("tags", []))
                event_type = str(out.get("event_type", "baseline_init" if baseline else "other"))[:40]
                title = str(out.get("title", "Baseline captured" if baseline else "Update detected")).strip()[:120]
                summary = str(out.get("summary", "")).strip()[:900] or str(out.get("what_changed_hint", "")).strip()[:260]

                # ---- KEY FIX FOR YOUR TESTING MODE ----
                # If baseline-init and you're truncating tables each run,
                # force at least one triaged event so coordinator always writes outputs.
                if baseline and baseline_force:
                    is_rel = True
                    if rel_score == 0:
                        rel_score = 75
                    if local_risk == 0:
                        local_risk = 55
                    if event_type == "other":
                        event_type = "baseline_init"
                    if not summary:
                        summary = "Baseline captured for monitoring."

                if debug:
                    print(
                        f"[TRIAGE] change_id={change_id} baseline={baseline} baseline_force={baseline_force} "
                        f"is_relevant_raw={out.get('is_relevant', None)} relevance_score_raw={out.get('relevance_score', None)} "
                        f"parsed_is_rel={is_rel} parsed_rel_score={rel_score} threshold={threshold}"
                    )

                # Decision gate
                if (not is_rel) or (rel_score < threshold):
                    update_change_triage_fields(
                        change_id=change_id,
                        status="ignored",
                        relevance_score=rel_score,
                        local_risk_score=local_risk,
                        tags=tags,
                    )
                    audit_log(
                        run_id=run_id,
                        agent_name=agent_name,
                        action="triage_ignored",
                        ref_type="change",
                        ref_id=change_id,
                        detail={
                            "relevance_score": rel_score,
                            "local_risk_score": local_risk,
                            "tags": tags,
                            "baseline": baseline,
                        },
                    )
                    stats["ignored"] += 1
                    continue

                update_change_triage_fields(
                    change_id=change_id,
                    status="triaged",
                    relevance_score=rel_score,
                    local_risk_score=local_risk,
                    tags=tags,
                )

                event_payload: Dict[str, Any] = {
                    "source_id": source_id,
                    "snapshot_id": int(new_id),
                    "change_id": change_id,
                    "agent_name": agent_name,
                    "event_type": event_type,
                    "title": title,
                    "summary": summary,
                    "tags": tags,
                    "relevance_score": rel_score,
                    "local_risk_score": local_risk,
                    "status": "new",
                }

                event_id = insert_agent_event(event_payload)

                audit_log(
                    run_id=run_id,
                    agent_name=agent_name,
                    action="triage_event_created",
                    ref_type="agent_event",
                    ref_id=int(event_id or 0),
                    detail={"change_id": change_id, "baseline": baseline},
                )

                stats["triaged"] += 1
                stats["events_created"] += 1

            except Exception as e:
                stats["errors"] += 1
                audit_log(
                    run_id=run_id,
                    agent_name=agent_name,
                    action="triage_error",
                    ref_type="change",
                    ref_id=change_id,
                    detail={"error": str(e)[:500]},
                )

        finish_agent_run(run_id, status="success", stats=stats)
        return 0

    except Exception as e:
        finish_agent_run(run_id, status="failed", stats={**stats, "fatal": str(e)[:500]})
        raise


if __name__ == "__main__":
    raise SystemExit(main())