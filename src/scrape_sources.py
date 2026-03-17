import hashlib
import io
import json
import os
import re
import time
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Tuple, Dict, Any, List

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

from .db import get_supabase_client, get_latest_snapshot_for_source, upsert_vector_chunks
from .config import USER_AGENT, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS, EMBED_BASELINE_ON_FIRST_SNAPSHOT
from .embedder import chunk_text, embed_texts

HEADERS = {"User-Agent": USER_AGENT}

# ── Quality thresholds ─────────────────────────────────────────────────────────
MIN_CLEAN_TEXT_LEN   = 1200
MAX_CLEAN_TEXT_CHARS = 25_000
REQUEST_TIMEOUT_S    = 30

# ── HTML junk tags (pure navigation / chrome) ──────────────────────────────────
# UPDATED: Added Google DevSite-specific web components
_JUNK_TAGS = [
    "script", "style", "noscript", "svg", "canvas", "iframe",
    "button", "form", "input", "select", "textarea",
    "nav", "footer", "header", "aside",
    "devsite-nav", "devsite-toc", "devsite-language-selector",
    "devsite-footer", "devsite-header", "devsite-feedback",
    "devsite-filter", "devsite-snackbar",
]

# Role / class / id fragments that signal non-content containers
# UPDATED: Added CISA (usa-banner) and Android Blog (widget, archive) targets
_JUNK_ATTRS = re.compile(
    r"(breadcrumb|cookie|consent|banner|promo|sidebar|toc-nav|"
    r"pagination|share|social|print|feedback|rating|related|"
    r"newsletter|subscribe|popup|modal|overlay|ads?|sponsor|"
    r"announcement|alert-bar|skip-link|widget|archive|devsite-page-nav|"
    r"usa-banner|usa-nav|usa-footer)",
    re.IGNORECASE,
)

# Legal / boilerplate patterns to strip from final text
_LEGAL_PATTERNS = [
    r"Except as otherwise noted,\s*the content of this page.*?(?=\n\n|\Z)",
    r"Terms of Service.*?Privacy Policy",
    r"Was this (helpful|page helpful)\?.*?(?=\n\n|\Z)",
    r"Last updated\s+\d{4}[-/]\d{2}[-/]\d{2}.*?UTC",
    r"Portions of this page are modifications based on work.*?(?=\n\n|\Z)",
    r"Creative Commons 2\.5 Attribution License.*?(?=\n\n|\Z)",
    r"Send feedback.*?(?=\n\n|\Z)",
    r"Content is available under.*?(?=\n\n|\Z)",
    r"\[?\s*Edit\s*(this\s*page)?\s*\]?",
    r"Home\s*/\s*Security\s*/.*?(?=\n)",
    r"Sign in to your Google Account.*?(?=\n\n|\Z)",
]

# ── Helpers ────────────────────────────────────────────────────────────────────


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _is_junk_element(tag: Tag) -> bool:
    """Return True if a tag looks like navigation/chrome rather than content."""
    for attr in ("class", "id", "role", "aria-label"):
        val = tag.get(attr, "")
        if isinstance(val, list):
            val = " ".join(val)
        if _JUNK_ATTRS.search(val):
            return True
    return False


def _remove_junk_tags(root: Tag) -> None:
    """Decompose tags that are definitively non-content."""
    for tag in root.find_all(_JUNK_TAGS):
        tag.decompose()


def _remove_junk_containers(root: Tag) -> None:
    """Decompose divs/sections whose class/id signal chrome."""
    for tag in root.find_all(True):
        if tag.name in ("div", "section", "span", "ul", "ol", "li") and _is_junk_element(tag):
            tag.decompose()


def _pick_root(soup: BeautifulSoup) -> Tag:
    """
    Prefer the most specific semantic content container.
    UPDATED: Prioritises Google DevSite and Android Blog architectures.
    """
    candidates = [
        soup.find("devsite-content"),              # Priority 1: Google/Android Docs
        soup.find("div", {"class": "post-body"}),  # Priority 2: Android Blog content
        soup.find("article"),
        soup.find("main"),                          # Priority 3: CISA / generic
        soup.find("div", {"id": re.compile(r"(content|main|article|body)", re.I)}),
        soup.find("div", {"class": re.compile(
            r"(devsite-article|article-body|page-content|"
            r"document-content|entry-content|post-content)", re.I,
        )}),
        soup.find("div", {"role": "main"}),
        soup.find("body"),
        soup,
    ]
    return next((c for c in candidates if c is not None), soup)


def _table_to_text(table: Tag) -> str:
    """
    Convert an HTML table to pipe-delimited text rows so 768-dim models
    can embed structured data (CVE tables, KEV catalog entries) properly.
    """
    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(separator=" ", strip=True) for td in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append("[TABLE_ROW] " + " | ".join(cells))
    return "\n".join(rows)


def _semantic_cleaner(root: Tag) -> str:
    """
    Walk the cleaned DOM and emit semantically structured plain text.
    Handles: headings -> [SECTION:], tables -> pipe rows, lists -> bullets.
    """
    parts: List[str] = []

    def _walk(node):
        if isinstance(node, NavigableString):
            text = str(node)
            text = re.sub(r"[\u200b\u200c\u200d\ufeff\xa0]+", " ", text)
            stripped = text.strip()
            if stripped:
                parts.append(stripped)
            return

        if not isinstance(node, Tag):
            return

        name = node.name

        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            heading_text = node.get_text(separator=" ", strip=True)
            if heading_text:
                prefix = "\n\n" if name in ("h1", "h2") else "\n"
                parts.append(f"{prefix}[SECTION: {heading_text.upper()}]")
            return

        if name == "table":
            table_text = _table_to_text(node)
            if table_text:
                parts.append("\n" + table_text + "\n")
            return

        if name == "li":
            item_text = node.get_text(separator=" ", strip=True)
            if item_text:
                parts.append(f"\n* {item_text}")
            return

        if name in ("p", "div", "section", "blockquote", "pre", "code",
                    "dd", "dt", "figcaption", "caption"):
            parts.append("\n")

        for child in node.children:
            _walk(child)

        if name in ("p", "div", "section", "blockquote"):
            parts.append("\n")

    _walk(root)

    text = " ".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\(https?://[^\)]{0,200}\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\(https?://[^\)]*\)", r"\1", text)

    return text.strip()


def _remove_legal_noise(text: str) -> str:
    """Strip boilerplate / legal footer text that dilutes embeddings."""
    for pattern in _LEGAL_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def _deduplicate_lines(text: str) -> str:
    """
    Remove duplicate or near-duplicate lines (common from repeated nav
    items that survive tag removal).
    """
    seen: set = set()
    output: List[str] = []
    for line in text.splitlines():
        key = line.strip()
        if not key:
            output.append(line)
            continue
        if key not in seen:
            seen.add(key)
            output.append(line)
    return "\n".join(output)


def _cap_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.7)]
    tail = text[-int(max_chars * 0.3):]
    return head.rstrip() + "\n\n[...truncated...]\n\n" + tail.lstrip()


# ── HTML fetch + clean ─────────────────────────────────────────────────────────


def fetch_raw_and_clean(url: str) -> Tuple[str, str]:
    """Fetch an HTML page and return (raw_html, clean_text)."""
    resp = requests.get(
        url,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT_S,
        allow_redirects=True,
    )
    resp.raise_for_status()

    raw_html = resp.text or ""
    soup = BeautifulSoup(raw_html, "html.parser")

    root = _pick_root(soup)
    _remove_junk_tags(root)
    _remove_junk_containers(root)
    clean_text = _semantic_cleaner(root)
    clean_text = _remove_legal_noise(clean_text)
    clean_text = _deduplicate_lines(clean_text)

    # Strip out empty [SECTION: ] tags left behind after nav removal
    clean_text = re.sub(r"\[SECTION: \]\n+", "", clean_text)
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()
    clean_text = _cap_text(clean_text, MAX_CLEAN_TEXT_CHARS)

    return raw_html, clean_text


# ── JSON fetch + clean (CISA KEV feed) ────────────────────────────────────────


def _format_kev_entry(v: Dict[str, Any]) -> str:
    """
    Render a single KEV vulnerability dict as structured plain text suitable
    for chunking and embedding. All fields kept; the triage agent filters by
    relevance rather than us pre-filtering here.
    """
    lines = [
        f"[CVE] {v.get('cveID', 'N/A')}",
        f"Vendor: {v.get('vendorProject', '')}",
        f"Product: {v.get('product', '')}",
        f"Vulnerability Name: {v.get('vulnerabilityName', '')}",
        f"Date Added: {v.get('dateAdded', '')}",
        f"Short Description: {v.get('shortDescription', '')}",
        f"Required Action: {v.get('requiredAction', '')}",
        f"Due Date: {v.get('dueDate', '')}",
        f"Ransomware Campaign Use: {v.get('knownRansomwareCampaignUse', 'Unknown')}",
        f"Notes: {v.get('notes', '').strip()}",
        "---",
    ]
    return "\n".join(line for line in lines if line.strip() != "---" or True)


def fetch_json_and_clean(url: str) -> Tuple[str, str]:
    """
    Fetch the CISA KEV JSON feed and return (raw_json_str, clean_text).

    The raw JSON string is stored verbatim in the snapshots.raw_text column
    (mirrors the role of raw_html for HTML sources). The clean_text is a
    human-readable, embedding-friendly rendering of every entry in the feed.

    No vendor filtering is applied here - the full catalog is ingested so
    the triage agent can classify relevance itself. The clean text is capped
    to MAX_CLEAN_TEXT_CHARS to stay within snapshot storage limits.
    """
    resp = requests.get(
        url,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT_S,
        allow_redirects=True,
    )
    resp.raise_for_status()

    raw_json_str = resp.text or "{}"

    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        raise ValueError(f"CISA KEV feed returned invalid JSON: {e}") from e

    vulnerabilities: List[Dict[str, Any]] = data.get("vulnerabilities", [])
    if not vulnerabilities:
        raise ValueError("CISA KEV feed contained no 'vulnerabilities' array.")

    catalog_version = data.get("catalogVersion", "unknown")
    date_released   = data.get("dateReleased", "unknown")
    total_count      = data.get("count", len(vulnerabilities))

    # Header block so the snapshot has high-level context for the triage agent
    header = (
        f"[SECTION: CISA KNOWN EXPLOITED VULNERABILITIES CATALOG]\n"
        f"Catalog Version: {catalog_version}\n"
        f"Date Released: {date_released}\n"
        f"Total Entries: {total_count}\n\n"
    )

    # Render each entry as structured text; newest entries last in feed order
    entry_blocks = [_format_kev_entry(v) for v in vulnerabilities]
    body = "\n".join(entry_blocks)

    clean_text = header + body
    clean_text = _cap_text(clean_text, MAX_CLEAN_TEXT_CHARS)

    return raw_json_str, clean_text


# ── RSS / Atom fetch + clean (Android Developers Blog) ────────────────────────


def fetch_rss_and_clean(url: str) -> Tuple[str, str]:
    """
    Fetch an RSS or Atom feed and return (raw_xml_str, clean_text).

    Parses standard RSS 2.0 and Atom 1.0 without external libraries.
    Each entry is rendered as a structured block: title, date, summary/content.
    Used for the Android Developers Blog Atom feed which is static XML and
    contains full post content including security and policy announcements.
    """
    resp = requests.get(
        url,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT_S,
        allow_redirects=True,
    )
    resp.raise_for_status()
    raw_xml = resp.text or ""

    soup = BeautifulSoup(raw_xml, "xml")

    entries = soup.find_all("entry") or soup.find_all("item")
    if not entries:
        raise ValueError(f"RSS/Atom feed at {url} contained no entries.")

    blocks: List[str] = [f"[SECTION: ANDROID DEVELOPERS BLOG FEED]\nTotal entries: {len(entries)}\n"]

    for entry in entries:
        # Title
        title_tag = entry.find("title")
        title = title_tag.get_text(strip=True) if title_tag else "Untitled"

        # Date - try multiple field names
        date = ""
        for tag in ("published", "updated", "pubDate", "dc:date"):
            t = entry.find(tag)
            if t:
                date = t.get_text(strip=True)
                break

        # Content - prefer full content over summary
        content = ""
        for tag in ("content", "content:encoded", "summary", "description"):
            t = entry.find(tag)
            if t:
                raw = t.get_text(separator=" ", strip=True)
                if len(raw) > len(content):
                    content = raw

        # Link
        link_tag = entry.find("link")
        link = ""
        if link_tag:
            link = link_tag.get("href") or link_tag.get_text(strip=True)

        block = (
            f"[ENTRY] {title}\n"
            f"Date: {date}\n"
            f"URL: {link}\n"
            f"Content: {content[:2000]}\n"
            f"---"
        )
        blocks.append(block)

    clean_text = "\n".join(blocks)
    clean_text = _cap_text(clean_text, MAX_CLEAN_TEXT_CHARS)
    return raw_xml, clean_text


# ── NVD CVE API 2.0 fetch + clean ─────────────────────────────────────────────

# How many days back to query on each pipeline run.
# 30 days catches any recently added/modified Android CVEs including
# retroactive NVD enrichment of older CVE IDs.
NVD_LOOKBACK_DAYS = int(os.getenv("NVD_LOOKBACK_DAYS", "30"))
NVD_RESULTS_PER_PAGE = 100
# Polite delay between paginated NVD requests (keyless = 5 req / 30s)
NVD_REQUEST_DELAY_S = 7.0


def _format_nvd_cve(cve: Dict[str, Any]) -> str:
    """Render a single NVD CVE 2.0 record as structured plain text."""
    cve_id = cve.get("id", "N/A")
    status = cve.get("vulnStatus", "")
    published = cve.get("published", "")
    last_modified = cve.get("lastModified", "")

    # English description
    desc = ""
    for d in (cve.get("descriptions") or []):
        if d.get("lang") == "en":
            desc = d.get("value", "")
            break

    # CVSS scores - prefer v3.1, fall back to v3.0, then v2
    severity = ""
    base_score = ""
    vector = ""
    metrics = cve.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        entries = metrics.get(key)
        if entries:
            m = entries[0].get("cvssData", {})
            severity = entries[0].get("baseSeverity", m.get("baseSeverity", ""))
            base_score = str(m.get("baseScore", ""))
            vector = m.get("vectorString", "")
            break

    # CWE
    weaknesses = cve.get("weaknesses") or []
    cwes = []
    for w in weaknesses:
        for desc_item in (w.get("description") or []):
            if desc_item.get("lang") == "en":
                cwes.append(desc_item.get("value", ""))
    cwe_str = ", ".join(cwes[:3])

    # CISA KEV flag
    cisa = cve.get("cisaExploitAdd", "")
    cisa_action = cve.get("cisaRequiredAction", "")

    lines = [
        f"[CVE] {cve_id}",
        f"Status: {status}",
        f"Published: {published}  |  Last Modified: {last_modified}",
        f"Severity: {severity}  |  CVSS Score: {base_score}  |  Vector: {vector}",
        f"CWE: {cwe_str}",
        f"Description: {desc[:800]}",
    ]
    if cisa:
        lines.append(f"CISA KEV Date Added: {cisa}  |  Action: {cisa_action}")
    lines.append("---")
    return "\n".join(lines)


def fetch_nvd_and_clean(url: str) -> Tuple[str, str]:
    """
    Query the NVD CVE 2.0 API for Android-related CVEs modified in the last
    NVD_LOOKBACK_DAYS days. Handles pagination automatically.

    The `url` parameter is used as the base URL (so seed_sources controls the
    keyword) and date range params are appended dynamically each run.

    Returns (raw_json_str, clean_text).
    The raw_json_str is the concatenated raw responses (newline-separated JSON).
    """
    now_utc = datetime.now(timezone.utc)
    start = now_utc - timedelta(days=NVD_LOOKBACK_DAYS)

    # ISO-8601 format NVD expects: 2026-01-01T00:00:00.000
    fmt = "%Y-%m-%dT%H:%M:%S.000"
    start_str = start.strftime(fmt)
    end_str = now_utc.strftime(fmt)

    all_cves: List[Dict[str, Any]] = []
    raw_parts: List[str] = []
    start_index = 0
    total_results = None

    while True:
        params = {
            "keywordSearch": "android",
            "resultsPerPage": NVD_RESULTS_PER_PAGE,
            "startIndex": start_index,
            "lastModStartDate": start_str,
            "lastModEndDate": end_str,
        }

        resp = requests.get(
            "https://services.nvd.nist.gov/rest/json/cves/2.0",
            headers={**HEADERS, "Accept": "application/json"},
            params=params,
            timeout=60,
        )
        resp.raise_for_status()
        raw_parts.append(resp.text)

        data = resp.json()
        if total_results is None:
            total_results = int(data.get("totalResults", 0))
            print(
                f"  [NVD] totalResults={total_results} "
                f"window={start_str} -> {end_str}",
                flush=True,
            )

        batch = data.get("vulnerabilities", [])
        for item in batch:
            all_cves.append(item.get("cve", {}))

        start_index += len(batch)
        if start_index >= total_results or not batch:
            break

        # Polite delay between pages to stay within keyless rate limit
        time.sleep(NVD_REQUEST_DELAY_S)

    if not all_cves:
        raise ValueError(
            f"NVD API returned 0 Android CVEs for window "
            f"{start_str} -> {end_str}. Check rate limit or window."
        )

    header = (
        f"[SECTION: NVD CVE FEED - ANDROID (LAST {NVD_LOOKBACK_DAYS} DAYS)]\n"
        f"Query window: {start_str} to {end_str}\n"
        f"Total results: {total_results}  |  Retrieved: {len(all_cves)}\n\n"
    )
    body = "\n".join(_format_nvd_cve(c) for c in all_cves)
    clean_text = _cap_text(header + body, MAX_CLEAN_TEXT_CHARS)

    return "\n".join(raw_parts), clean_text


# ── OSV Android bulk ZIP fetch + clean ────────────────────────────────────────

# How many OSV records to include per snapshot. The Android ecosystem zip
# contains thousands of entries; cap to keep clean_text within storage limits.
OSV_MAX_RECORDS = int(os.getenv("OSV_MAX_RECORDS", "500"))


def _format_osv_entry(entry: Dict[str, Any]) -> str:
    """Render a single OSV record as structured plain text."""
    osv_id = entry.get("id", "N/A")
    aliases = ", ".join(entry.get("aliases") or [])
    summary = entry.get("summary", "")
    details = (entry.get("details") or "")[:600]
    modified = entry.get("modified", "")
    published = entry.get("published", "")
    severity = ""
    for s in (entry.get("severity") or []):
        severity = f"{s.get('type','')} {s.get('score','')}".strip()
        break

    lines = [
        f"[OSV] {osv_id}",
        f"Aliases: {aliases}",
        f"Published: {published}  |  Modified: {modified}",
        f"Severity: {severity}",
        f"Summary: {summary}",
        f"Details: {details}",
        "---",
    ]
    return "\n".join(lines)


def fetch_osv_and_clean(url: str) -> Tuple[str, str]:
    """
    Download the OSV Android ecosystem bulk ZIP from Google's GCS bucket,
    extract all JSON records, and render as structured plain text.

    Sorted by `modified` descending so the most recent vulnerabilities appear
    first and survive the MAX_CLEAN_TEXT_CHARS cap.

    Returns (raw_manifest_json, clean_text) where raw_manifest_json is a
    lightweight JSON summary (not the full zip bytes) for the raw_text column.
    """
    resp = requests.get(
        url,
        headers=HEADERS,
        timeout=120,   # zip can be several MB
        allow_redirects=True,
    )
    resp.raise_for_status()

    zip_bytes = resp.content
    entries: List[Dict[str, Any]] = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            try:
                with zf.open(name) as f:
                    entry = json.load(f)
                    if isinstance(entry, dict):
                        entries.append(entry)
            except Exception:
                continue

    if not entries:
        raise ValueError("OSV Android zip contained no parseable JSON records.")

    # Sort newest-modified first
    entries.sort(key=lambda e: e.get("modified", ""), reverse=True)
    top = entries[:OSV_MAX_RECORDS]

    raw_manifest = json.dumps({
        "source": url,
        "total_in_zip": len(entries),
        "records_rendered": len(top),
        "newest_modified": entries[0].get("modified", "") if entries else "",
    })

    header = (
        f"[SECTION: OSV ANDROID VULNERABILITY DATABASE]\n"
        f"Total records in zip: {len(entries)}\n"
        f"Records rendered (newest first): {len(top)}\n\n"
    )
    body = "\n".join(_format_osv_entry(e) for e in top)
    clean_text = _cap_text(header + body, MAX_CLEAN_TEXT_CHARS)

    return raw_manifest, clean_text


# ── Vector storage ─────────────────────────────────────────────────────────────


def _store_vectors_for_snapshot(
    source_id_int: int,
    snapshot_id: int,
    snapshot_sha: str,
    kind: str,
    clean_text: str,
) -> int:
    chunks = chunk_text(
        clean_text,
        chunk_size_chars=CHUNK_SIZE_CHARS,
        overlap_chars=CHUNK_OVERLAP_CHARS,
    )
    if not chunks:
        return 0

    embs = embed_texts(chunks)
    rows: List[Dict[str, Any]] = [
        {
            "source_id": str(source_id_int),
            "snapshot_sha": str(snapshot_sha),
            "kind": kind,
            "chunk_index": int(i),
            "chunk_text": ch,
            "embedding": emb,
            "source_id_int": int(source_id_int),
            "snapshot_id": int(snapshot_id),
        }
        for i, (ch, emb) in enumerate(zip(chunks, embs))
    ]

    upsert_vector_chunks(rows)
    return len(rows)


# ── ETL entry point ────────────────────────────────────────────────────────────


def main():
    sb = get_supabase_client()
    now = _utc_now_iso()

    sources = (
        sb.table("sources")
        .select("id,name,url,fetch_type")
        .eq("active", True)
        .execute()
        .data
    )

    print(f"🚀 Found {len(sources)} active sources. Starting ingestion...", flush=True)

    inserted = 0
    skipped  = 0
    embedded = 0

    for s in sources:
        src_id     = int(s["id"])
        name       = s["name"]
        url        = s["url"]
        fetch_type = (s.get("fetch_type") or "html").lower().strip()

        prev_latest       = get_latest_snapshot_for_source(src_id)
        is_first_snapshot = prev_latest is None

        # ── Dispatch by fetch_type ───────────────────────────────────────────
        try:
            if fetch_type == "json":
                raw_text, clean_text = fetch_json_and_clean(url)
            elif fetch_type == "rss":
                raw_text, clean_text = fetch_rss_and_clean(url)
            elif fetch_type == "api_nvd":
                raw_text, clean_text = fetch_nvd_and_clean(url)
            elif fetch_type == "api_osv":
                raw_text, clean_text = fetch_osv_and_clean(url)
            else:
                # Default: HTML (covers fetch_type="html" and unknown values)
                raw_text, clean_text = fetch_raw_and_clean(url)
        except Exception as e:
            print(f"⚠️  Fetch failed for source='{name}' (fetch_type={fetch_type}): {e}", flush=True)
            continue

        if len(clean_text) < MIN_CLEAN_TEXT_LEN:
            skipped += 1
            print(f"⏩ Skipped (too short): {name}  len={len(clean_text)}", flush=True)
            continue

        content_hash = _sha256(clean_text)

        payload = {
            "source_id":    src_id,
            "fetched_at":   now,
            "content_hash": content_hash,
            # raw_text holds raw_html for HTML sources, raw JSON for JSON sources
            "raw_text":     raw_text,
            "clean_text":   clean_text,
        }

        ins = sb.table("snapshots").insert(payload).execute()
        row = (ins.data or [{}])[0]
        snapshot_id = int(row.get("id") or 0)

        if snapshot_id <= 0:
            print(f"❌ Failed to get snapshot ID for '{name}'", flush=True)
            continue

        inserted += 1
        print(f"📦 Stored snapshot: {name} (ID: {snapshot_id}, type={fetch_type})", flush=True)

        if is_first_snapshot and not EMBED_BASELINE_ON_FIRST_SNAPSHOT:
            continue

        kind = "baseline" if is_first_snapshot else "snapshot"
        try:
            nvec = _store_vectors_for_snapshot(
                source_id_int=src_id,
                snapshot_id=snapshot_id,
                snapshot_sha=content_hash,
                kind=kind,
                clean_text=clean_text,
            )
            embedded += nvec
            print(f"✨ Embedded {nvec} chunks ({kind}): {name}", flush=True)
        except Exception as e:
            print(f"❌ Vector embed failed for '{name}': {e}", flush=True)

    print(
        f"\n✅ ETL CYCLE COMPLETE. "
        f"inserted={inserted}  skipped={skipped}  total_chunks={embedded}",
        flush=True,
    )


if __name__ == "__main__":
    main()