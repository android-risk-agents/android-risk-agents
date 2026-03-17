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
from src.embedder import embed_texts  # used for category description embeddings only

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# RISK TAXONOMY
# Fixed category definitions used for deterministic embedding-similarity
# classification. These descriptions are embedded once at agent startup and
# reused for every change in the run. Adding or editing a category here is
# the only change needed to extend the taxonomy.
# =============================================================================

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

# =============================================================================
# RISK BUCKET RULES
# Maps (base_risk, similarity_score) -> risk_bucket and decision.
# All thresholds are configurable via ENV so you can tune without code changes.
# =============================================================================

# Similarity score below which a category match is considered unreliable.
# Below this threshold the change is classified as "general" regardless of
# which category had the highest cosine similarity.
MIN_SIMILARITY_THRESHOLD = float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.30"))

# Relevance score (0-1 cosine similarity) thresholds for triage decision.
# Changes with similarity >= TRIAGE_THRESHOLD go to triage; below -> ignore.
# This replaces the old LLM-generated relevance_score threshold.
TRIAGE_THRESHOLD = float(os.getenv("TRIAGE_THRESHOLD", "0.35"))

# Risk bucket boundaries by base_risk level + similarity score.
# base_risk=high: high bucket if sim >= 0.45, medium if >= 0.35, else low
# base_risk=medium: high bucket if sim >= 0.55, medium if >= 0.40, else low
# base_risk=low: always low bucket regardless of similarity
BUCKET_RULES: Dict[str, List[Tuple[float, str]]] = {
    "high":   [(0.45, "high"), (0.35, "medium"), (0.0, "low")],
    "medium": [(0.55, "high"), (0.40, "medium"), (0.0, "low")],
    "low":    [(0.0, "low")],
}


# =============================================================================
# EMBEDDING CACHE
# Category descriptions are embedded once per agent run and cached here.
# This avoids re-embedding the same 8 descriptions for every change.
# =============================================================================

_CATEGORY_EMBEDDINGS: Optional[List[Tuple[Dict[str, Any], List[float]]]] = None


def _get_category_embeddings() -> List[Tuple[Dict[str, Any], List[float]]]:
    """
    Embed all risk category descriptions and cache the result.
    Returns list of (category_dict, embedding_vector) pairs.
    """
    global _CATEGORY_EMBEDDINGS
    if _CATEGORY_EMBEDDINGS is not None:
        return _CATEGORY_EMBEDDINGS

    descriptions = [c["description"] for c in RISK_CATEGORIES]
    # embed_texts returns normalised embeddings (nomic uses search_document prefix)
    embeddings = embed_texts(descriptions, is_query=False)

    _CATEGORY_EMBEDDINGS = list(zip(RISK_CATEGORIES, embeddings))
    logger.info(f"Category embeddings computed for {len(_CATEGORY_EMBEDDINGS)} categories.")
    return _CATEGORY_EMBEDDINGS


# =============================================================================
# CLASSIFICATION LOGIC
# Pure functions - no LLM, fully deterministic given the same embedder.
# =============================================================================


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two pre-normalised vectors.
    embed_texts() with normalize_embeddings=True means dot product == cosine sim.
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    # Clamp to [-1, 1] to handle floating point drift
    return max(-1.0, min(1.0, dot))


def _average_embeddings(embeddings: List[List[float]]) -> List[float]:
    """
    Average a list of chunk embeddings into a single document-level vector.
    Since nomic embeddings are already L2-normalised per chunk, averaging
    gives a reasonable centroid. We re-normalise the result so cosine
    similarity comparisons remain valid.
    """
    if not embeddings:
        return []
    dim = len(embeddings[0])
    avg = [0.0] * dim
    for emb in embeddings:
        for i, v in enumerate(emb):
            avg[i] += v
    n = len(embeddings)
    avg = [v / n for v in avg]

    # Re-normalise
    magnitude = math.sqrt(sum(v * v for v in avg))
    if magnitude > 0:
        avg = [v / magnitude for v in avg]
    return avg


def classify_change(snapshot_id: int, fallback_text: str = "") -> Dict[str, Any]:
    """
    Classify a change using pre-stored vector_chunks embeddings for the
    snapshot. Averages all chunk embeddings into one document-level vector,
    then compares cosine similarity against all risk category embeddings.

    Falls back to embedding `fallback_text` directly if no chunks are found
    in vector_chunks (e.g. snapshot was too short and skipped by embedder,
    or EMBED_BASELINE_ON_FIRST_SNAPSHOT=false).

    Returns deterministic classification dict - same inputs always produce
    the same output given the same embedder model.
    """
    category_embeddings = _get_category_embeddings()

    # ── Get document embedding ────────────────────────────────────────────────
    chunk_embeddings = get_snapshot_embeddings(snapshot_id)

    if chunk_embeddings:
        doc_embedding = _average_embeddings(chunk_embeddings)
        embedding_source = f"averaged_{len(chunk_embeddings)}_chunks"
    elif fallback_text:
        # Re-embed on the fly only as last resort
        logger.warning(
            f"No chunks found for snapshot_id={snapshot_id}, "
            f"falling back to direct embedding."
        )
        doc_embedding = embed_texts([fallback_text[:6000]], is_query=True)[0]
        embedding_source = "fallback_direct_embed"
    else:
        logger.error(f"No chunks and no fallback text for snapshot_id={snapshot_id}.")
        return {
            "risk_category":         "general",
            "risk_category_label":   "General",
            "risk_bucket":           "low",
            "similarity_score":      0.0,
            "classification_method": "embedding_similarity",
            "decision":              "ignore",
            "all_scores":            {},
            "embedding_source":      "none",
        }

    # ── Score every category ──────────────────────────────────────────────────
    scores: Dict[str, float] = {}
    for cat, cat_emb in category_embeddings:
        scores[cat["id"]] = _cosine_similarity(doc_embedding, cat_emb)

    best_id    = max(scores, key=lambda k: scores[k])
    best_score = scores[best_id]
    best_cat   = next(c for c in RISK_CATEGORIES if c["id"] == best_id)

    # Low-confidence fallback
    if best_score < MIN_SIMILARITY_THRESHOLD:
        best_cat   = next(c for c in RISK_CATEGORIES if c["id"] == "general")
        best_id    = "general"
        best_score = scores.get("general", best_score)

    # ── Risk bucket ───────────────────────────────────────────────────────────
    base_risk   = best_cat["base_risk"]
    risk_bucket = "low"
    for threshold, bucket in BUCKET_RULES[base_risk]:
        if best_score >= threshold:
            risk_bucket = bucket
            break

    # ── Decision ──────────────────────────────────────────────────────────────
    if risk_bucket == "high":
        decision = "triage"
    elif best_score >= TRIAGE_THRESHOLD:
        decision = "triage"
    else:
        decision = "ignore"

    return {
        "risk_category":         best_id,
        "risk_category_label":   best_cat["label"],
        "risk_bucket":           risk_bucket,
        "similarity_score":      round(best_score, 4),
        "classification_method": "embedding_similarity",
        "decision":              decision,
        "all_scores":            {k: round(v, 4) for k, v in scores.items()},
        "embedding_source":      embedding_source,
    }


# =============================================================================
# LLM: RATIONALE + SUMMARY ONLY
# The LLM no longer determines any scores. It only produces human-readable
# narrative that the Coordinator agent uses for recommendations.
# =============================================================================

SYSTEM_RATIONALE = """You are the Sentinel Triage Agent, a senior analyst in Android fraud risk and mobile security intelligence.

A deterministic classifier has already assigned a risk category and risk bucket to this event.
Your only job is to produce a structured JSON summary that will be used by the Coordinator Agent
to generate recommendations for fraud and security engineering teams.

OUTPUT RULES:
1. Return ONLY a raw JSON object. No markdown, no code fences, no preamble, no extra text.
2. Do NOT override or question the risk_category or risk_bucket. They are fixed inputs.
3. Write as if briefing a fraud analyst or risk engineer who needs to act immediately.
4. Be specific: name CVE IDs, affected components, Android versions, API names, or policy sections
   that appear in the content. Do not generalise when specifics are available.
5. affected_signals must name concrete Android fraud/device signals that this event impacts
   (e.g. "Android ID", "Play Integrity verdict", "SafetyNet attestation", "kernel version signal",
   "IMEI access", "advertising ID", "bootloader state"). Leave empty array if none are affected.
6. Use exactly this JSON schema - no extra fields:

{
  "rationale": "<4-5 sentences: what changed or was disclosed, which Android components or CVEs are involved, why it matters for fraud risk or device integrity, and what the downstream impact is on risk models or detection signals>",
  "insight": "<2-3 sentences: the broader security implication beyond this single event - e.g. trend this fits into, attacker capability it enables, or compliance pressure it creates>",
  "affected_signals": ["<signal 1>", "<signal 2>"],
  "recommended_actions": ["<specific action 1>", "<specific action 2>", "<specific action 3>"]
}"""


def _detect_source_type(url: str) -> str:
    """Derive a human-readable source label from the URL for prompt context."""
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
    MAX_CHARS = 4000
    safe_text = (change_text or "")[:MAX_CHARS]
    source_type = _detect_source_type(url)
    all_scores = classification.get("all_scores", {})

    # Format all category scores for transparency so the LLM understands
    # the classification context without being able to change it
    scores_str = "  " + "\n  ".join(
        f"{k}: {v}" for k, v in sorted(all_scores.items(), key=lambda x: -x[1])
    ) if all_scores else "  (not available)"

    return (
        f"CHANGE ID: {change_id}\n"
        f"SOURCE TYPE: {source_type}\n"
        f"SOURCE URL: {url}\n"
        f"ANALYSIS TYPE: {'BASELINE - first time this source has been ingested' if baseline else 'DIFF - content changed since last snapshot'}\n\n"
        f"DETERMINISTIC CLASSIFICATION RESULTS:\n"
        f"  Risk Category: {classification['risk_category_label']} ({classification['risk_category']})\n"
        f"  Risk Bucket: {classification['risk_bucket'].upper()}\n"
        f"  Similarity Score: {classification['similarity_score']} "
        f"(cosine similarity to category description embedding)\n"
        f"  Embedding Source: {classification.get('embedding_source', 'unknown')}\n"
        f"  All Category Scores:\n{scores_str}\n\n"
        f"CONTENT (trimmed to {MAX_CHARS} chars):\n"
        f"{safe_text}\n\n"
        f"Write the rationale, insight, affected_signals, and recommended_actions JSON for this event.\n"
        f"Ground every claim in the content above. Do not invent CVEs, component names, or version numbers."
    )


def _parse_rationale_response(raw: Dict[str, Any]) -> Tuple[str, str, List[str], List[str]]:
    """
    Extract rationale, insight, affected_signals, and recommended_actions
    from LLM response dict. Returns (rationale, insight, affected_signals, actions).
    """
    rationale = str(raw.get("rationale", "No rationale provided."))[:2000]
    insight   = str(raw.get("insight", ""))[:1000]

    affected = raw.get("affected_signals", [])
    if not isinstance(affected, list):
        affected = []
    affected = [str(s)[:120] for s in affected[:6]]

    actions = raw.get("recommended_actions", [])
    if not isinstance(actions, list):
        actions = []
    actions = [str(a)[:300] for a in actions[:5]]

    return rationale, insight, affected, actions


# =============================================================================
# SCHEMA (updated to include new deterministic fields)
# =============================================================================

class SentinelTriageResult(TypedDict):
    change_id:              int
    risk_category:          str
    risk_category_label:    str
    risk_bucket:            str
    similarity_score:       float
    classification_method:  str
    decision:               str
    rationale:              str
    recommended_actions:    List[str]
    tags:                   List[str]
    # Legacy fields preserved for coordinator compatibility
    relevance_score:        int    # mapped from risk_bucket (high=90, medium=60, low=30)
    local_risk_score:       int    # same mapping


def _bucket_to_score(bucket: str) -> int:
    """Map risk bucket to legacy int score (0-100) for coordinator compatibility."""
    return {"high": 90, "medium": 60, "low": 30}.get(bucket, 30)


def _derive_tags(classification: Dict[str, Any], url: str) -> List[str]:
    """Derive tags from classification result for filtering/search."""
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


# =============================================================================
# BASELINE ENFORCEMENT
# Baseline changes (first snapshot ever for a source) are always triaged
# regardless of classification score. This preserves the existing behaviour.
# =============================================================================

def _is_baseline_init(change: Dict[str, Any]) -> bool:
    prev_id = change.get("prev_snapshot_id")
    diff_json = change.get("diff_json", {})
    if isinstance(diff_json, dict) and diff_json.get("type") == "baseline_init":
        return True
    return prev_id is None


# =============================================================================
# MAIN AGENT LOOP
# =============================================================================

def main():
    agent_name = os.getenv("AGENT_NAME", "sentinel-triage")
    vllm_model = os.getenv("VLLM_MODEL", os.getenv("MODEL_TRIAGE", "llama-3.1-8b-instant"))

    logger.info(
        f"Starting {agent_name} | model={vllm_model} "
        f"| triage_threshold={TRIAGE_THRESHOLD} "
        f"| min_similarity={MIN_SIMILARITY_THRESHOLD} "
        f"| classification=embedding_similarity"
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
        # Pre-compute category embeddings once for the whole run
        _get_category_embeddings()

        changes = get_pending_changes_for_triage(limit=25)
        if not changes:
            logger.info("No pending changes found.")
            finish_agent_run(run_id, "success", stats)
            return

        for ch in changes:
            change_id = ch.get("id")
            source_id = ch.get("source_id")
            new_id    = ch.get("new_snapshot_id")
            prev_id   = ch.get("prev_snapshot_id")

            if not change_id or not source_id or not new_id:
                logger.warning(f"Skipping change {change_id}: missing identifiers")
                stats["errors"] += 1
                continue

            baseline = _is_baseline_init(ch)

            # ── Load snapshot texts ───────────────────────────────────────────
            old_text = "" if baseline else get_snapshot_text_by_id(prev_id)
            new_text = get_snapshot_text_by_id(new_id)
            url      = get_source_url(source_id)

            if not new_text:
                logger.warning(f"Change {change_id}: empty snapshot text, skipping")
                stats["errors"] += 1
                continue

            # ── Step 1: Deterministic embedding-similarity classification ─────
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

            # ── Baseline override: always triage first snapshots ──────────────
            if baseline and classification["decision"] == "ignore":
                classification["decision"] = "triage"
                if classification["risk_bucket"] == "low":
                    classification["risk_bucket"] = "medium"

            decision = classification["decision"]

            # ── Map to legacy int scores for coordinator compatibility ─────────
            score_int = _bucket_to_score(classification["risk_bucket"])
            tags      = _derive_tags(classification, url)

            # ── Write deterministic classification fields to DB ───────────────
            update_change_classification_fields(
                change_id=change_id,
                risk_category=classification["risk_category"],
                risk_bucket=classification["risk_bucket"],
                similarity_score=classification["similarity_score"],
                classification_method=classification["classification_method"],
            )

            stats["processed"] += 1

            # ── Ignore path ───────────────────────────────────────────────────
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

            # ── Triage path: call LLM for rationale only ──────────────────────
            rationale      = "No rationale generated."
            insight        = ""
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
                    max_tokens=700,
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
                # LLM failure does NOT block triage - classification already done

            # Write triage status
            update_change_triage_fields(
                change_id=change_id,
                status="triaged",
                relevance_score=score_int,
                local_risk_score=score_int,
                tags=tags,
            )

            # Build summary for coordinator: combine rationale + insight
            # so coordinator's EVENT SUMMARY field is as rich as possible
            coordinator_summary = rationale
            if insight:
                coordinator_summary = f"{rationale}\n\nInsight: {insight}"

            # Insert agent event for Coordinator (same schema as before)
            event_payload = {
                "source_id":    source_id,
                "snapshot_id":  new_id,
                "change_id":    change_id,
                "agent_name":   agent_name,
                "event_type":   "baseline_init" if baseline else "diff_update",
                "title":        f"[{classification['risk_bucket'].upper()}] {classification['risk_category_label']}",
                "summary":      coordinator_summary,
                "tags":         tags,
                "relevance_score":   score_int,
                "local_risk_score":  score_int,
                "status":       "new",
            }

            event_id = insert_agent_event(event_payload)
            audit_log(
                run_id, agent_name, "triage_event_created",
                "agent_events", event_id,
                {
                    **classification,
                    "rationale":        rationale,
                    "insight":          insight,
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