import os
import json
import logging
import math
from typing import Dict, Any, TypedDict, List, Tuple, Optional

from src.db import (
    create_agent_run,
    finish_agent_run,
    get_pending_changes_for_triage,
    get_source_url,
    get_snapshot_text_by_id,
    get_snapshot_embeddings,
    update_change_triage_fields,
    update_change_classification_fields,
    insert_agent_event,
    audit_log,
)
from src.llm_client import get_llm_client, chat_json
from src.embedder import embed_texts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


RISK_CATEGORIES: List[Dict[str, Any]] = [
    {
        "id": "device_integrity",
        "label": "Device Integrity",
        "base_risk": "high",
        "description": (
            "Changes related to Android device attestation, Play Integrity API, "
            "SafetyNet, bootloader state, root detection, emulator detection, "
            "hardware-backed keystore, TEE (Trusted Execution Environment), "
            "verified boot, device fingerprinting signals, tamper detection, "
            "anti-cheat mechanisms, and device authenticity verification."
        ),
    },
    {
        "id": "fraud_signal_degradation",
        "label": "Fraud Signal Degradation",
        "base_risk": "high",
        "description": (
            "Changes that reduce the reliability or availability of fraud detection "
            "signals on Android: deprecation of Android ID, changes to IMEI or "
            "hardware identifier access, advertising ID opt-out policy changes, "
            "sensor permission restrictions, reduction of device signal fidelity, "
            "identifier randomisation, privacy sandbox changes affecting attribution, "
            "and removal of previously available risk signals."
        ),
    },
    {
        "id": "permission_changes",
        "label": "Permission Changes",
        "base_risk": "medium",
        "description": (
            "New or modified Android runtime permissions, dangerous permission "
            "groups, background location restrictions, microphone and camera "
            "indicators, READ_PHONE_STATE changes, MANAGE_EXTERNAL_STORAGE, "
            "notification permission requirements, health and fitness permissions, "
            "nearby devices permissions, and permission auto-revocation policy changes."
        ),
    },
    {
        "id": "policy_compliance",
        "label": "Policy and Compliance",
        "base_risk": "medium",
        "description": (
            "Google Play Store policy updates, developer content policy changes, "
            "financial services app requirements, loan app regulations, target SDK "
            "version enforcement deadlines, data safety section requirements, "
            "Play billing policy, app review policy changes, GDPR or CCPA related "
            "platform changes, and regulatory compliance requirements for Android apps."
        ),
    },
    {
        "id": "network_security",
        "label": "Network Security",
        "base_risk": "medium",
        "description": (
            "Android network security configuration changes, cleartext traffic "
            "restrictions, certificate pinning, TLS version enforcement, "
            "Private DNS (DNS-over-TLS), VPN permission changes, network "
            "permission restrictions, WebView security updates, and changes "
            "to how Android handles SSL/TLS certificates or proxies."
        ),
    },
    {
        "id": "malware_exposure",
        "label": "Malware Exposure",
        "base_risk": "high",
        "description": (
            "Active Android malware campaigns, exploit-in-the-wild CVEs targeting "
            "Android, banking trojans, spyware, adware, ransomware targeting Android "
            "devices, zero-day vulnerabilities in Android framework or kernel, "
            "privilege escalation exploits, remote code execution vulnerabilities, "
            "and CISA Known Exploited Vulnerabilities affecting Android."
        ),
    },
    {
        "id": "platform_update",
        "label": "Platform Update",
        "base_risk": "low",
        "description": (
            "General Android OS version releases, Android security patch level "
            "updates without active exploits, new Android API additions, SDK "
            "updates, Android Studio tooling changes, Gradle plugin updates, "
            "and routine monthly security bulletins with no critical or high "
            "severity vulnerabilities in use by attackers."
        ),
    },
    {
        "id": "general",
        "label": "General",
        "base_risk": "low",
        "description": (
            "Miscellaneous Android ecosystem news, blog posts, documentation "
            "updates, community announcements, and changes that do not directly "
            "affect security, fraud risk, device integrity, or compliance."
        ),
    },
]


MIN_SIMILARITY_THRESHOLD = float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.30"))
TRIAGE_THRESHOLD = float(os.getenv("TRIAGE_THRESHOLD", "0.35"))

BUCKET_RULES: Dict[str, List[Tuple[float, str]]] = {
    "high": [(0.45, "high"), (0.35, "medium"), (0.0, "low")],
    "medium": [(0.55, "high"), (0.40, "medium"), (0.0, "low")],
    "low": [(0.0, "low")],
}

SENTINEL_MAX_TOKENS = int(os.getenv("SENTINEL_MAX_TOKENS", "240"))

_CATEGORY_EMBEDDINGS: Optional[List[Tuple[Dict[str, Any], List[float]]]] = None


def _get_category_embeddings() -> List[Tuple[Dict[str, Any], List[float]]]:
    global _CATEGORY_EMBEDDINGS
    if _CATEGORY_EMBEDDINGS is not None:
        return _CATEGORY_EMBEDDINGS

    descriptions = [c["description"] for c in RISK_CATEGORIES]
    embeddings = embed_texts(descriptions, is_query=False)

    _CATEGORY_EMBEDDINGS = list(zip(RISK_CATEGORIES, embeddings))
    logger.info(f"Category embeddings computed for {len(_CATEGORY_EMBEDDINGS)} categories.")
    return _CATEGORY_EMBEDDINGS


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return max(-1.0, min(1.0, dot))


def _average_embeddings(embeddings: List[List[float]]) -> List[float]:
    if not embeddings:
        return []
    dim = len(embeddings[0])
    avg = [0.0] * dim
    for emb in embeddings:
        for i, v in enumerate(emb):
            avg[i] += v
    n = len(embeddings)
    avg = [v / n for v in avg]

    magnitude = math.sqrt(sum(v * v for v in avg))
    if magnitude > 0:
        avg = [v / magnitude for v in avg]
    return avg


def classify_change(snapshot_id: int, fallback_text: str = "") -> Dict[str, Any]:
    category_embeddings = _get_category_embeddings()
    chunk_embeddings = get_snapshot_embeddings(snapshot_id)

    if chunk_embeddings:
        doc_embedding = _average_embeddings(chunk_embeddings)
        embedding_source = f"averaged_{len(chunk_embeddings)}_chunks"
    elif fallback_text:
        logger.warning(
            f"No chunks found for snapshot_id={snapshot_id}, "
            f"falling back to direct embedding."
        )
        doc_embedding = embed_texts([fallback_text[:6000]], is_query=True)[0]
        embedding_source = "fallback_direct_embed"
    else:
        logger.error(f"No chunks and no fallback text for snapshot_id={snapshot_id}.")
        return {
            "risk_category": "general",
            "risk_category_label": "General",
            "risk_bucket": "low",
            "similarity_score": 0.0,
            "classification_method": "embedding_similarity",
            "decision": "ignore",
            "all_scores": {},
            "embedding_source": "none",
        }

    scores: Dict[str, float] = {}
    for cat, cat_emb in category_embeddings:
        scores[cat["id"]] = _cosine_similarity(doc_embedding, cat_emb)

    best_id = max(scores, key=lambda k: scores[k])
    best_score = scores[best_id]
    best_cat = next(c for c in RISK_CATEGORIES if c["id"] == best_id)

    if best_score < MIN_SIMILARITY_THRESHOLD:
        best_cat = next(c for c in RISK_CATEGORIES if c["id"] == "general")
        best_id = "general"
        best_score = scores.get("general", best_score)

    base_risk = best_cat["base_risk"]
    risk_bucket = "low"
    for threshold, bucket in BUCKET_RULES[base_risk]:
        if best_score >= threshold:
            risk_bucket = bucket
            break

    if risk_bucket == "high":
        decision = "triage"
    elif best_score >= TRIAGE_THRESHOLD:
        decision = "triage"
    else:
        decision = "ignore"

    return {
        "risk_category": best_id,
        "risk_category_label": best_cat["label"],
        "risk_bucket": risk_bucket,
        "similarity_score": round(best_score, 4),
        "classification_method": "embedding_similarity",
        "decision": decision,
        "all_scores": {k: round(v, 4) for k, v in scores.items()},
        "embedding_source": embedding_source,
    }


SYSTEM_RATIONALE = """You are the Sentinel Triage Agent for Android fraud risk and mobile security intelligence.

A deterministic classifier has already assigned the risk category and risk bucket.
Your job is to return a concise structured JSON summary for the Coordinator Agent.

RULES:
1. Return ONLY one raw JSON object.
2. Do NOT override the provided risk_category or risk_bucket.
3. Ground every claim in the provided content.
4. Be specific when CVE IDs, Android components, versions, APIs, or policy sections are present.
5. affected_signals must name concrete Android fraud or device signals when relevant.
6. Use exactly this schema:

{
  "rationale": "<3-4 sentences on what changed, what Android components or issues are involved, why it matters for fraud risk or device integrity, and likely downstream impact>",
  "insight": "<1-2 sentences on broader implication or trend>",
  "affected_signals": ["<signal 1>", "<signal 2>"],
  "recommended_actions": ["<specific action 1>", "<specific action 2>", "<specific action 3>"]
}"""


def _detect_source_type(url: str) -> str:
    u = url.lower()
    if "cisa.gov" in u:
        return "CISA Known Exploited Vulnerabilities (KEV) catalog"
    if "nvd.nist.gov" in u or "nvd" in u:
        return "NVD CVE database (NIST)"
    if "osv" in u or "osv-vulnerabilities" in u:
        return "OSV Android vulnerability database (Google)"
    if "source.android.com" in u and "bulletin" in u:
        return "Android Security Bulletin (Google AOSP)"
    if "android-developers.googleblog.com" in u or "atom.xml" in u:
        return "Android Developers Blog"
    if "play.google" in u or "developer-content-policy" in u:
        return "Google Play Developer Policy Center"
    if "developer.android.com" in u and "integrity" in u:
        return "Play Integrity API documentation"
    return "Android ecosystem source"


def build_rationale_prompt(
    change_text: str,
    url: str,
    classification: Dict[str, Any],
    baseline: bool,
    change_id: int,
) -> str:
    max_chars = int(os.getenv("SENTINEL_CONTENT_MAX_CHARS", "2400"))
    safe_text = (change_text or "")[:max_chars]
    source_type = _detect_source_type(url)

    return (
        f"CHANGE ID: {change_id}\n"
        f"SOURCE TYPE: {source_type}\n"
        f"SOURCE URL: {url}\n"
        f"ANALYSIS TYPE: {'BASELINE' if baseline else 'DIFF'}\n"
        f"RISK CATEGORY: {classification['risk_category_label']} ({classification['risk_category']})\n"
        f"RISK BUCKET: {classification['risk_bucket'].upper()}\n"
        f"SIMILARITY SCORE: {classification['similarity_score']}\n"
        f"EMBEDDING SOURCE: {classification.get('embedding_source', 'unknown')}\n\n"
        f"CONTENT:\n{safe_text}\n\n"
        "Return the JSON fields rationale, insight, affected_signals, and recommended_actions.\n"
        "Do not invent CVEs, component names, versions, or policy sections."
    )


def _parse_rationale_response(raw: Dict[str, Any]) -> Tuple[str, str, List[str], List[str]]:
    rationale = str(raw.get("rationale", "No rationale provided."))[:2000]
    insight = str(raw.get("insight", ""))[:1000]

    affected = raw.get("affected_signals", [])
    if not isinstance(affected, list):
        affected = []
    affected = [str(s)[:120] for s in affected[:6]]

    actions = raw.get("recommended_actions", [])
    if not isinstance(actions, list):
        actions = []
    actions = [str(a)[:300] for a in actions[:5]]

    return rationale, insight, affected, actions


class SentinelTriageResult(TypedDict):
    change_id: int
    risk_category: str
    risk_category_label: str
    risk_bucket: str
    similarity_score: float
    classification_method: str
    decision: str
    rationale: str
    recommended_actions: List[str]
    tags: List[str]
    relevance_score: int
    local_risk_score: int


def _bucket_to_score(bucket: str) -> int:
    return {"high": 90, "medium": 60, "low": 30}.get(bucket, 30)


def _derive_tags(classification: Dict[str, Any], url: str) -> List[str]:
    tags = [
        classification["risk_category"],
        classification["risk_bucket"],
        "embedding-classified",
    ]
    if "android" in url.lower() or "source.android" in url.lower():
        tags.append("android-bulletin")
    if "cisa" in url.lower():
        tags.append("cisa-kev")
    if "nvd" in url.lower():
        tags.append("nvd-cve")
    if "osv" in url.lower():
        tags.append("osv")
    return tags[:8]


def _is_baseline_init(change: Dict[str, Any]) -> bool:
    prev_id = change.get("prev_snapshot_id")
    diff_json = change.get("diff_json", {})
    if isinstance(diff_json, dict) and diff_json.get("type") == "baseline_init":
        return True
    return prev_id is None


def main():
    agent_name = os.getenv("AGENT_NAME", "sentinel-triage")
    vllm_model = os.getenv("VLLM_MODEL", os.getenv("MODEL_TRIAGE", "llama-3.1-8b-instant"))

    logger.info(
        f"Starting {agent_name} | model={vllm_model} "
        f"| triage_threshold={TRIAGE_THRESHOLD} "
        f"| min_similarity={MIN_SIMILARITY_THRESHOLD} "
        f"| classification=embedding_similarity "
        f"| sentinel_max_tokens={SENTINEL_MAX_TOKENS}"
    )

    try:
        llm_client = get_llm_client()
    except Exception as e:
        logger.error(f"Failed to initialise LLM client: {e}")
        return

    run_id = create_agent_run(run_name=agent_name, trigger="cron", llm_backend=vllm_model)

    stats = {
        "processed": 0,
        "ignored": 0,
        "triaged": 0,
        "errors": 0,
        "events_created": 0,
        "llm_calls": 0,
    }

    try:
        _get_category_embeddings()

        changes = get_pending_changes_for_triage(limit=25)
        if not changes:
            logger.info("No pending changes found.")
            finish_agent_run(run_id, "success", stats)
            return

        for ch in changes:
            change_id = ch.get("id")
            source_id = ch.get("source_id")
            new_id = ch.get("new_snapshot_id")
            prev_id = ch.get("prev_snapshot_id")

            if not change_id or not source_id or not new_id:
                logger.warning(f"Skipping change {change_id}: missing identifiers")
                stats["errors"] += 1
                continue

            baseline = _is_baseline_init(ch)

            new_text = get_snapshot_text_by_id(new_id)
            url = get_source_url(source_id)

            if not new_text:
                logger.warning(f"Change {change_id}: empty snapshot text, skipping")
                stats["errors"] += 1
                continue

            try:
                classification = classify_change(
                    snapshot_id=new_id,
                    fallback_text=new_text[:6000] if new_text else "",
                )
            except Exception as e:
                logger.error(f"Classification failed for change {change_id}: {e}")
                stats["errors"] += 1
                audit_log(run_id, agent_name, "classification_error", "changes", change_id, {"error": str(e)})
                continue

            logger.info(
                f"Change {change_id} | category={classification['risk_category']} "
                f"| bucket={classification['risk_bucket']} "
                f"| sim={classification['similarity_score']} "
                f"| decision={classification['decision']}"
            )

            if baseline and classification["decision"] == "ignore":
                classification["decision"] = "triage"
                if classification["risk_bucket"] == "low":
                    classification["risk_bucket"] = "medium"

            decision = classification["decision"]
            score_int = _bucket_to_score(classification["risk_bucket"])
            tags = _derive_tags(classification, url)

            update_change_classification_fields(
                change_id=change_id,
                risk_category=classification["risk_category"],
                risk_bucket=classification["risk_bucket"],
                similarity_score=classification["similarity_score"],
                classification_method=classification["classification_method"],
            )

            stats["processed"] += 1

            if decision == "ignore":
                update_change_triage_fields(
                    change_id=change_id,
                    status="ignored",
                    relevance_score=score_int,
                    local_risk_score=score_int,
                    tags=tags,
                )
                audit_log(run_id, agent_name, "triage_ignored", "changes", change_id, classification)
                stats["ignored"] += 1
                logger.info(f"Change {change_id} IGNORED.")
                continue

            rationale = "No rationale generated."
            insight = ""
            affected_signals: List[str] = []
            rec_actions: List[str] = []

            try:
                rationale_prompt = build_rationale_prompt(
                    change_text=new_text,
                    url=url,
                    classification=classification,
                    baseline=baseline,
                    change_id=change_id,
                )
                raw_llm = chat_json(
                    client=llm_client,
                    model=vllm_model,
                    system=SYSTEM_RATIONALE,
                    user=rationale_prompt,
                    temperature=0.1,
                    max_tokens=SENTINEL_MAX_TOKENS,
                )
                rationale, insight, affected_signals, rec_actions = _parse_rationale_response(raw_llm)
                stats["llm_calls"] += 1
            except Exception as e:
                logger.warning(f"LLM rationale failed for change {change_id}: {e} - using fallback.")
                rationale = (
                    f"Deterministic classifier assigned category "
                    f"'{classification['risk_category_label']}' "
                    f"(bucket={classification['risk_bucket']}, "
                    f"similarity={classification['similarity_score']}). "
                    f"LLM rationale unavailable."
                )

            update_change_triage_fields(
                change_id=change_id,
                status="triaged",
                relevance_score=score_int,
                local_risk_score=score_int,
                tags=tags,
            )

            coordinator_summary = rationale
            if insight:
                coordinator_summary = f"{rationale}\n\nInsight: {insight}"

            event_payload = {
                "source_id": source_id,
                "snapshot_id": new_id,
                "change_id": change_id,
                "agent_name": agent_name,
                "event_type": "baseline_init" if baseline else "diff_update",
                "title": f"[{classification['risk_bucket'].upper()}] {classification['risk_category_label']}",
                "summary": coordinator_summary,
                "tags": tags,
                "relevance_score": score_int,
                "local_risk_score": score_int,
                "status": "new",
            }

            event_id = insert_agent_event(event_payload)
            audit_log(
                run_id,
                agent_name,
                "triage_event_created",
                "agent_events",
                event_id,
                {
                    **classification,
                    "rationale": rationale,
                    "insight": insight,
                    "affected_signals": affected_signals,
                    "recommended_actions": rec_actions,
                },
            )
            stats["triaged"] += 1
            stats["events_created"] += 1
            logger.info(f"Change {change_id} TRIAGED -> Event {event_id}.")

        finish_agent_run(run_id, "success", stats)
        logger.info(f"Run complete: {stats}")

    except Exception as e:
        logger.error(f"Fatal error in {agent_name}: {e}")
        finish_agent_run(run_id, "failed", stats)
        raise


if __name__ == "__main__":
    main()