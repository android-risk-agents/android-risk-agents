# src/agents/sentinel_agent.py
import os
import json
import logging
import requests
from typing import Dict, Any

from src.db import (
    create_agent_run,
    finish_agent_run,
    get_pending_changes_for_triage,
    get_source_url,
    get_snapshot_text_by_id,
    update_change_triage_fields,
    insert_agent_event,
    audit_log
)

from .prompts import SYSTEM_TRIAGE, build_prompt
from .validate import parse_and_validate_triage_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def _is_baseline_init(change: Dict[str, Any]) -> bool:
    """Detects baseline case (first-time snapshot)"""
    prev_id = change.get("prev_snapshot_id")
    # Also check if diff_json type says baseline_init if needed
    diff_json = change.get("diff_json", {})
    if isinstance(diff_json, dict) and diff_json.get("type") == "baseline_init":
        return True
    return prev_id is None

def call_vllm_json(prompt: str, base_url: str, model: str) -> str:
    """
    Calls the vLLM OpenAI-compatible endpoint directly using the requests library.
    Requests strictly JSON output from the model.
    """
    base_url = base_url.rstrip('/')
    if not base_url.endswith('/v1'):
        base_url = f"{base_url}/v1"
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }

    # Pass API key if one exists, though often not strictly required for local vLLM 
    api_key = os.getenv("VLLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_TRIAGE},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 400
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    
    data = response.json()
    return data["choices"][0]["message"]["content"]

def main():
    agent_name = os.getenv("AGENT_NAME", "sentinel-triage")
    threshold = float(os.getenv("TRIAGE_THRESHOLD", "0.7"))
    
    # Read vLLM credentials from env as requested
    vllm_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
    vllm_model = os.getenv("VLLM_MODEL", "llama-3.1-8b-instant")
    
    # Needs valid Supabase credentials stored in ENV (handled cleanly by src.db under the hood)
    logger.info(f"Starting {agent_name} run with threshold {threshold} on model {vllm_model}")

    run_id = create_agent_run(run_name=agent_name, trigger="cron", llm_backend=vllm_model)

    stats = {
        "processed": 0,
        "ignored": 0,
        "triaged": 0,
        "errors": 0,
        "events_created": 0
    }

    try:
        changes = get_pending_changes_for_triage(limit=25)
        if not changes:
            logger.info("No pending changes found.")
            finish_agent_run(run_id, "success", stats)
            return

        for ch in changes:
            change_id = ch.get("id")
            source_id = ch.get("source_id")
            prev_id = ch.get("prev_snapshot_id")
            new_id = ch.get("new_snapshot_id")

            if not change_id or not source_id or not new_id:
                logger.warning(f"Skipping change {change_id}: missing identifiers")
                stats["errors"] += 1
                continue
            
            baseline = _is_baseline_init(ch)

            # Load texts
            old_text = "" if baseline else get_snapshot_text_by_id(prev_id)
            new_text = get_snapshot_text_by_id(new_id)
            url = get_source_url(source_id)

            user_prompt = build_prompt(old_text, new_text, url, baseline, change_id)
            
            # Call vLLM endpoint directly using requests
            try:
                raw_llm_json = call_vllm_json(user_prompt, vllm_base_url, vllm_model)
                parsed_result = parse_and_validate_triage_json(raw_llm_json, fallback_change_id=change_id)
            except requests.exceptions.RequestException as e:
                logger.error(f"vLLM Network Error on change {change_id}: {e}")
                stats["errors"] += 1
                audit_log(run_id, agent_name, "triage_error", "changes", change_id, {"error": str(e)})
                continue
            except Exception as e:
                logger.error(f"Failed to process LLM output for change {change_id}: {e}")
                stats["errors"] += 1
                audit_log(run_id, agent_name, "triage_error", "changes", change_id, {"error": str(e)})
                continue
                
            stats["processed"] += 1
            
            # Extract final scores & map severity to local_risk_score since db schema wants local_risk_score
            decision = parsed_result["decision"]
            rel_score = parsed_result["relevance_score"]
            risk_score = parsed_result["severity_score"] 
            tags = parsed_result["tags"]
            
            # NOTE: We scale 0.0-1.0 to 0-100 since `update_change_triage_fields` expects `int(relevance_score)`
            rel_score_int = int(rel_score * 100)
            risk_score_int = int(risk_score * 100)

            # Ignore path (never ignore baseline/fresh runs)
            if not baseline and (decision == "ignore" or rel_score < threshold):
                update_change_triage_fields(
                    change_id=change_id,
                    status="ignored",
                    relevance_score=rel_score_int,
                    local_risk_score=risk_score_int,
                    tags=tags
                )
                audit_log(run_id, agent_name, "triage_ignored", "changes", change_id, parsed_result)
                stats["ignored"] += 1
                logger.info(f"Change {change_id} IGNORED (Score: {rel_score})")
            
            # Triage path
            else:
                update_change_triage_fields(
                    change_id=change_id,
                    status="triaged",
                    relevance_score=rel_score_int,
                    local_risk_score=risk_score_int,
                    tags=tags
                )
                
                # Insert Agent Event for Coordinator
                payload = {
                    "source_id": source_id,
                    "snapshot_id": new_id,
                    "change_id": change_id,
                    "agent_name": agent_name,
                    "event_type": "baseline_init" if baseline else "diff_update",
                    "title": f"Triage Alert: {parsed_result['category']}",
                    "summary": parsed_result["rationale"],
                    "tags": tags,
                    "relevance_score": rel_score_int,
                    "local_risk_score": risk_score_int,
                    "status": "new"
                }
                
                event_id = insert_agent_event(payload)
                audit_log(run_id, agent_name, "triage_event_created", "agent_events", event_id, parsed_result)
                stats["triaged"] += 1
                stats["events_created"] += 1
                logger.info(f"Change {change_id} TRIAGED -> Event {event_id} created.")

        finish_agent_run(run_id, "success", stats)
        logger.info(f"Run completed successfully: {stats}")

    except Exception as e:
        logger.error(f"Fatal error during {agent_name} run: {e}")
        finish_agent_run(run_id, "failed", stats)
        raise

if __name__ == "__main__":
    main()
