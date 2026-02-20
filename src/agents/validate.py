# src/agents/validate.py
import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

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
