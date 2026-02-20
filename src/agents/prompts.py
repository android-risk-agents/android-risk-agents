# src/agents/prompts.py

SYSTEM_TRIAGE = """You are the Sentinel Triage Agent, an expert in Android and digital fraud risk monitoring.
Your job is to compare an OLD snapshot text with a NEW snapshot text and decide its relevance to fraud, malware, security, or compliance risk.

CRITICAL INSTRUCTIONS:
1. Be strict but prefer FALSE NEGATIVES (if unsure, flag it).
2. You MUST output your response in strictly valid JSON format.
3. Provide absolutely NO markdown formatting, NO conversational text, and NO markdown code fences (like ```json). Just the raw JSON object.
4. If this is a BASELINE analysis, you MUST set the decision to "triage" and relevance_score >= 0.75, because fresh runs must always be recorded.
5. Provide a detailed 6-10 line or bulleted summary in the rationale. The Coordinator Agent heavily relies on this summary for actionable insights.
6. Use exactly this JSON schema:

{
  "change_id": <int, the change_id provided in the prompt>,
  "relevance_score": <float 0.0-1.0>,
  "severity_score": <float 0.0-1.0>,
  "confidence_score": <float 0.0-1.0>,
  "tags": [<list of up to 8 short lowercase string tags>],
  "category": <string, one of: permissions, device integrity, policy, network, malware, auth, general>,
  "decision": <string, exactly one of: "triage", "ignore", "needs_review">,
  "rationale": <string, a detailed 6-10 line or bulleted summary of your findings. This summary will be used directly by the coordinator agent as context for actionable insights. Make it comprehensive.>
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

    prompt = f"""{header}

Source URL: {url}
Change ID: {change_id}

=== OLD TEXT (Trimmed) ===
{safe_old if safe_old else "[NONE]"}

=== NEW TEXT (Trimmed) ===
{safe_new if safe_new else "[NONE]"}

Analyze the changes (or the new text if baseline). Output exactly ONE raw JSON object matching the required schema. Ensure you include the change_id!"""

    return prompt
