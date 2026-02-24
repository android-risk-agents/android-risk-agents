import os
import json
import logging
from typing import Dict, Any, TypedDict, List

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
from src.llm_client import get_llm_client, chat_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- schemas.py ---
class SentinelTriageResult(TypedDict):
    """
    Schema validating the strict JSON output required from the Sentinel Triage Agent.
    """
    change_id: int
    relevance_score: float      # Range: 0.0 to 1.0
    severity_score: float       # Range: 0.0 to 1.0
    confidence_score: float     # Range: 0.0 to 1.0
    tags: List[str]             # Max 8 items, each max 60 chars
    category: str               # e.g., permissions, device integrity, policy, network, malware, auth
    decision: str               # Must be exactly one of: "triage", "ignore", "needs_review"
    rationale: str              # Brief explanation of the decision

# --- prompts.py ---
SYSTEM_TRIAGE = """You are the Sentinel Triage Agent, an expert in Android and digital fraud risk monitoring.
Your job is to compare an OLD snapshot text with a NEW snapshot text and decide its relevance to fraud, malware, security, or compliance risk.

CRITICAL INSTRUCTIONS:
1. Be conservative: if unsure, set decision to 'needs_review' (do not ignore uncertain items).
2. You MUST output your response in strictly valid JSON format.
3. Provide absolutely NO markdown formatting, NO conversational text, and NO markdown code fences (like ```json). Just the raw JSON object.
4. If this is a BASELINE analysis, you MUST set the decision to "triage" and relevance_score >= 0.75, because fresh runs must always be recorded.
5. Provide a 3-4 sentence summary in the rationale. The Coordinator Agent heavily relies on this summary for actionable insights.
6. Use exactly this JSON schema:

{
  "change_id": <int, the change_id provided in the prompt>,
  "relevance_score": <float 0.0-1.0>,
  "severity_score": <float 0.0-1.0>,
  "confidence_score": <float 0.0-1.0>,
  "tags": [<list of up to 8 short lowercase string tags>],
  "category": <string, one of: permissions, device integrity, policy, network, malware, auth, general>,
  "decision": <string, exactly one of: "triage", "ignore", "needs_review">,
  "rationale": <string, a 3-4 sentence summary of your findings. This summary will be used directly by the coordinator agent as context for actionable insights. Make it comprehensive.>
}
"""

def build_prompt(old_text: str, new_text: str, url: str, baseline: bool, change_id: int) -> str:
    """
    Creates the exact user prompt sent to the LLM.
    """
    # Trim to ~4500 characters to ensure we stay within prompt context limits
    MAX_CHARS = 4500
    safe_old = (old_text or "")[:MAX_CHARS]
    safe_new = (new_text or "")[:MAX_CHARS]

    header = (
        "This is a BASELINE analysis (no previous snapshot exists)."
        if baseline else
        "This is a DIFF analysis (comparing previous snapshot to new snapshot)."
    )

    prompt = f"{header}\n\nSource URL: {url}\nChange ID: {change_id}\n\n=== OLD TEXT (Trimmed) ===\n{safe_old if safe_old else '[NONE]'}\n\n=== NEW TEXT (Trimmed) ===\n{safe_new if safe_new else '[NONE]'}\n\nAnalyze the changes (or the new text if baseline). Output exactly ONE raw JSON object matching the required schema. Ensure you include the change_id!"

    return prompt

# --- validate.py ---
def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        val = float(v)
        return max(0.0, min(1.0, val))  # Clamp between 0.0 and 1.0
    except (ValueError, TypeError):
        return default

def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return default

def _as_tags(v: Any) -> list[str]:
    if not isinstance(v, list):
        return []
    safe_tags = []
    for t in v:
        if isinstance(t, str):
            clean = t.strip()[:60].lower()
            if clean:
                safe_tags.append(clean)
    return safe_tags[:8]

def parse_and_validate_triage_json(raw_text: str, fallback_change_id: int) -> Dict[str, Any]:
    """
    Takes the raw string from the LLM, attempts to extract JSON,
    and returns a guaranteed structured dict matching SentinelTriageResult.
    """
    raw_text = (raw_text or "").strip()
    
    # Attempt to strip accidental markdown fences
    if raw_text.startswith("```"):
        parts = raw_text.split("```")
        if len(parts) >= 2:
            raw_text = parts[1].replace("json", "", 1).strip()

    parsed = {}
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # Fallback to finding the first and last curly braces if the LLM outputted conversational garbage
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(raw_text[start : end + 1])
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse inner JSON: {e}")
        else:
            logger.error(f"Could not locate JSON brackets in LLM output: {raw_text[:200]}...")

    if not isinstance(parsed, dict):
        parsed = {}

    decision = parsed.get("decision", "ignore").lower()
    if decision not in ["triage", "ignore", "needs_review"]:
        decision = "ignore"
        
    category = parsed.get("category", "general")
    if not isinstance(category, str):
        category = "general"
    else:
        category = category.lower()
        valid_categories = {"permissions", "device integrity", "policy", "network", "malware", "auth", "general"}
        if category not in valid_categories:
            category = "general"

    # Enforce constraints and return the structured schema
    return {
        "change_id": _as_int(parsed.get("change_id"), fallback_change_id),
        "relevance_score": _as_float(parsed.get("relevance_score"), 0.0),
        "severity_score": _as_float(parsed.get("severity_score"), 0.0),
        "confidence_score": _as_float(parsed.get("confidence_score"), 0.0),
        "tags": _as_tags(parsed.get("tags")),
        "category": category[:120],
        "decision": decision,
        "rationale": str(parsed.get("rationale", "No rationale provided"))[:2000]
    }

# --- sentinel_agent.py ---
def _is_baseline_init(change: Dict[str, Any]) -> bool:
    """Detects baseline case (first-time snapshot)"""
    prev_id = change.get("prev_snapshot_id")
    # Also check if diff_json type says baseline_init if needed
    diff_json = change.get("diff_json", {})
    if isinstance(diff_json, dict) and diff_json.get("type") == "baseline_init":
        return True
    return prev_id is None

def main():
    agent_name = os.getenv("AGENT_NAME", "sentinel-triage")
    threshold = float(os.getenv("TRIAGE_THRESHOLD", "0.7"))
    
    vllm_model = os.getenv("VLLM_MODEL", os.getenv("MODEL_TRIAGE", "llama-3.1-8b-instant"))
    
    try:
        llm_client = get_llm_client()
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        return
        
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
            
            # Route through the centralized llm_client
            try:
                raw_llm_dict = chat_json(
                    client=llm_client,
                    model=vllm_model,
                    system=SYSTEM_TRIAGE,
                    user=user_prompt,
                    temperature=0.0,
                    max_tokens=400
                )
                raw_llm_json = json.dumps(raw_llm_dict)
                parsed_result = parse_and_validate_triage_json(raw_llm_json, fallback_change_id=change_id)
                
                # Fix 2: Enforce baseline rule in code
                if baseline:
                    parsed_result["decision"] = "triage"
                    if parsed_result["relevance_score"] < 0.75:
                        parsed_result["relevance_score"] = 0.75
                        
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