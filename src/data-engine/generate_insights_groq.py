# src/generate_insights_groq.py
import os
import json
import time
from typing import Any, Dict, Optional, List

from openai import OpenAI

from .db import (
    get_uninsighted_changes,
    get_snapshot_text_by_id,
    insert_insight,
    create_baseline_changes,
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing. Add it as a GitHub Actions secret.")

GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

MODEL_TRIAGE = os.getenv("GROQ_MODEL_TRIAGE", "llama-3.1-8b-instant")
MODEL_ANALYZE = os.getenv("GROQ_MODEL_ANALYZE", "llama-3.3-70b-versatile")

AGENT_NAME = os.getenv("AGENT_NAME", "groq-digital-risk-agent")
RELEVANCE_THRESHOLD = int(os.getenv("RELEVANCE_THRESHOLD", "70"))

SYSTEM_TRIAGE = (
    "You are a Digital Risk Intelligence triage agent for a fraud prevention team. "
    "Decide if the change between OLD and NEW text is relevant to digital fraud risk monitoring. "
    "Be strict. Prefer false negatives over false positives if uncertain. "
    "Return ONLY valid JSON. Do not include markdown."
)

SYSTEM_ANALYZE = (
    "You are a Digital Risk Intelligence Agent supporting a fraud prevention team. "
    "You do NOT write generic cybersecurity mitigation advice. "
    "You focus on platform ecosystem changes and how they impact fraud detection and digital risk solutions. "
    "Be concrete. Do not invent facts. If unknown, say unknown. "
    "Return ONLY valid JSON. Do not include markdown."
)


def extract_json_only(s: str) -> Dict[str, Any]:
    s = (s or "").strip()

    # Handle accidental fenced blocks
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1].replace("json", "", 1).strip()

    try:
        return json.loads(s)
    except Exception:
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(s[start : end + 1])


def _as_list_of_str(x: Any, max_items: int, max_len: int) -> Optional[List[str]]:
    if x is None:
        return None
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
    return out if out else None


def safe_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    title = str(obj.get("title", "")).strip() or "Update detected"
    summary = str(obj.get("summary", "")).strip() or "Update detected. Details unknown."

    category = obj.get("category")
    if category is not None:
        category = str(category).strip()[:80]
        if category == "":
            category = None

    affected_signals = _as_list_of_str(obj.get("affected_signals"), max_items=5, max_len=120)
    recommended_actions = _as_list_of_str(obj.get("recommended_actions"), max_items=5, max_len=140)

    try:
        confidence = float(obj.get("confidence", 0.6))
    except Exception:
        confidence = 0.6
    confidence = max(0.0, min(1.0, confidence))

    risk_score = obj.get("risk_score", None)
    if risk_score is not None:
        try:
            risk_score = int(risk_score)
            risk_score = max(1, min(5, risk_score))
        except Exception:
            risk_score = None

    return {
        "title": title[:120],
        "summary": summary[:1200],
        "category": category,
        "affected_signals": affected_signals,
        "recommended_actions": recommended_actions,
        "risk_score": risk_score,
        "confidence": confidence,
    }


def build_triage_prompt(old_text: str, new_text: str, url: str) -> str:
    schema_hint = {
        "is_relevant": "boolean",
        "relevance_score": "integer 0..100",
        "primary_theme": "platform_change|data_access|policy_change|threat_actor|vulnerability|fraud_tactic|other",
        "reasons": ["string (up to 3)"],
        "what_changed_hint": "string (one sentence)",
    }

    return (
        f"SOURCE: {url}\n\n"
        f"OLD TEXT (trimmed):\n{old_text[:3500]}\n\n"
        f"NEW TEXT (trimmed):\n{new_text[:3500]}\n\n"
        "Task: Decide relevance to digital fraud risk monitoring and detection.\n"
        "Relevant means it likely affects data collection, device or identity signals, platform APIs or policies, "
        "attacker capabilities, fraud tactics, automation, or detection opportunities.\n\n"
        "Return JSON only.\n"
        f"Schema:\n{json.dumps(schema_hint)}"
    )


def build_baseline_briefing_prompt(new_text: str, url: str) -> str:
    """
    Baseline briefing for a newly monitored source (first run, no meaningful diff).
    This MUST still produce actionable insight (what the source is + how to leverage it).
    """
    schema_hint = {
        "title": "string",
        "summary": "string (2-5 sentences, what this source is + why it matters for digital risk)",
        "category": "string (e.g., vulnerability_intel|platform_change|policy_change|threat_actor|other)",
        "affected_signals": ["string (signals or intel outputs we can derive, up to 5)"],
        "recommended_actions": ["string (leverage ideas for risk pipeline, up to 5)"],
        "risk_score": "integer 1..5 (urgency to pay attention to this source)",
        "confidence": "number 0..1",
    }

    return (
        f"SOURCE: {url}\n\n"
        "This is an INITIAL BASELINE BRIEFING for a newly monitored source.\n"
        "Do not claim something changed.\n"
        "Extract what this source provides and how a digital fraud risk team can leverage it.\n\n"
        "Specific requirement:\n"
        "- If this is a list or catalog (e.g., CVE/KEV), explain what it tracks and pull out 3-5 notable recent entries "
        "from the visible text (if present).\n"
        "- recommended_actions must be leverage opportunities (features, scoring, rules, monitoring), not generic patch advice.\n\n"
        f"CONTENT (trimmed):\n{new_text[:7000]}\n\n"
        "Return JSON only.\n"
        f"Schema:\n{json.dumps(schema_hint)}"
    )


def build_analysis_prompt(old_text: str, new_text: str, url: str, triage: Dict[str, Any]) -> str:
    schema_hint = {
        "title": "string",
        "summary": "string (2-5 sentences, concrete and actionable)",
        "category": "string (optional, use primary_theme if helpful)",
        "affected_signals": ["string (new or changed data signals, up to 5)"],
        "recommended_actions": [
            "string (specific leverage ideas for fraud/digital risk, not generic mitigations, up to 5)"
        ],
        "risk_score": "integer 1..5 (digital risk urgency, not CVSS)",
        "confidence": "number 0..1",
    }

    triage_hint = {
        "primary_theme": triage.get("primary_theme"),
        "what_changed_hint": triage.get("what_changed_hint"),
        "reasons": triage.get("reasons"),
        "relevance_score": triage.get("relevance_score"),
    }

    return (
        f"SOURCE: {url}\n\n"
        f"TRIAGE CONTEXT:\n{json.dumps(triage_hint)}\n\n"
        f"OLD TEXT (trimmed):\n{old_text[:4500]}\n\n"
        f"NEW TEXT (trimmed):\n{new_text[:4500]}\n\n"
        "Instructions:\n"
        "1) Focus on platform capability changes, policy updates, SDK or API changes, or attacker capability shifts.\n"
        "2) affected_signals should be new or restricted signals (device, identity, behavioral, telemetry, permissions).\n"
        "3) recommended_actions must be leverage opportunities (collection, features, scoring, rules, investigations).\n"
        "4) Avoid generic advice like 'apply mitigations' unless it directly enables detection improvements.\n"
        "5) If unknown, say unknown.\n\n"
        "Return JSON only.\n"
        f"Schema:\n{json.dumps(schema_hint)}"
    )


def _call_llm(
    client: OpenAI,
    model: str,
    system: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content or "{}"
    return extract_json_only(content)


def run() -> int:
    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

    changes = get_uninsighted_changes(limit=25)

    if not changes:
        created = create_baseline_changes(limit=50)
        print(f"No changes pending insights. Created baseline changes: {created}")
        changes = get_uninsighted_changes(limit=25)

    if not changes:
        print("Still no changes available after baseline creation.")
        return 0

    created_insights = 0

    for ch in changes:
        try:
            old_text = ""
            if ch.old_snapshot_id is not None:
                old_text = get_snapshot_text_by_id(int(ch.old_snapshot_id)) or ""

            new_text = get_snapshot_text_by_id(int(ch.new_snapshot_id)) or ""

            # Baseline detection: prev == new snapshot id
            is_baseline = False
            if ch.old_snapshot_id is not None and int(ch.old_snapshot_id) == int(ch.new_snapshot_id):
                is_baseline = True

            # ✅ Baseline bypass: do NOT triage based on diffs. Always produce a baseline briefing from NEW text.
            if is_baseline:
                baseline_prompt = build_baseline_briefing_prompt(new_text, ch.url)
                baseline_raw = _call_llm(
                    client=client,
                    model=MODEL_ANALYZE,
                    system=SYSTEM_ANALYZE,
                    prompt=baseline_prompt,
                    temperature=0.2,
                    max_tokens=520,
                )
                out = safe_output(baseline_raw)

                insert_insight(
                    change_id=ch.id,
                    agent_name=AGENT_NAME,
                    title=out["title"],
                    summary=out["summary"],
                    confidence=out["confidence"],
                    category=out["category"],
                    affected_signals=out["affected_signals"],
                    recommended_actions=out["recommended_actions"],
                    risk_score=out["risk_score"],
                )

                created_insights += 1
                print(f"Baseline briefing insight upserted for change_id={ch.id}")
                time.sleep(0.25)
                continue

            # Stage 1: strict triage for real diffs
            triage_prompt = build_triage_prompt(old_text, new_text, ch.url)
            triage_raw = _call_llm(
                client=client,
                model=MODEL_TRIAGE,
                system=SYSTEM_TRIAGE,
                prompt=triage_prompt,
                temperature=0.0,
                max_tokens=250,
            )

            is_relevant = bool(triage_raw.get("is_relevant", False))
            try:
                rel_score = int(triage_raw.get("relevance_score", 0))
            except Exception:
                rel_score = 0

            primary_theme = str(triage_raw.get("primary_theme", "other"))[:80]

            if (not is_relevant) or (rel_score < RELEVANCE_THRESHOLD):
                title = f"Not relevant to digital risk (score {rel_score})"
                summary = (
                    "This update does not appear to change data collection capabilities, platform policies, "
                    "or attacker and detection dynamics in a way that is actionable for digital fraud risk monitoring. "
                    f"Theme: {primary_theme}. "
                    f"Hint: {str(triage_raw.get('what_changed_hint', 'unknown'))[:200]}"
                )
                out = safe_output(
                    {
                        "title": title,
                        "summary": summary,
                        "category": primary_theme,
                        "affected_signals": [],
                        "recommended_actions": [],
                        "risk_score": 1,
                        "confidence": 0.55,
                    }
                )

                insert_insight(
                    change_id=ch.id,
                    agent_name=AGENT_NAME,
                    title=out["title"],
                    summary=out["summary"],
                    confidence=out["confidence"],
                    category=out["category"],
                    affected_signals=out["affected_signals"],
                    recommended_actions=out["recommended_actions"],
                    risk_score=out["risk_score"],
                )

                created_insights += 1
                print(f"Triage-only insight upserted for change_id={ch.id} (score={rel_score})")
                time.sleep(0.20)
                continue

            # Stage 2: deep analysis for relevant diffs
            analysis_prompt = build_analysis_prompt(old_text, new_text, ch.url, triage_raw)
            analysis_raw = _call_llm(
                client=client,
                model=MODEL_ANALYZE,
                system=SYSTEM_ANALYZE,
                prompt=analysis_prompt,
                temperature=0.2,
                max_tokens=520,
            )

            out = safe_output(analysis_raw)

            if not out.get("category"):
                out["category"] = primary_theme

            if out["title"] and "relevance" not in out["title"].lower():
                out["title"] = f"{out['title']} (relevance {rel_score})"[:120]

            insert_insight(
                change_id=ch.id,
                agent_name=AGENT_NAME,
                title=out["title"],
                summary=out["summary"],
                confidence=out["confidence"],
                category=out["category"],
                affected_signals=out["affected_signals"],
                recommended_actions=out["recommended_actions"],
                risk_score=out["risk_score"],
            )

            created_insights += 1
            print(f"Insight upserted for change_id={ch.id} (score={rel_score})")
            time.sleep(0.25)

        except Exception as e:
            print(f"Insight failed for change_id={getattr(ch, 'id', 'unknown')}: {e}")
            continue

    print(f"Done. Upserted {created_insights}/{len(changes)} insights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())