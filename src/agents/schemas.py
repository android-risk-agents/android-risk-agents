# src/agents/schemas.py
from typing import TypedDict, List

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
