# src/scrape_sources.py
import hashlib
import re
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, List

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

from .db import get_supabase_client, get_latest_snapshot_for_source, upsert_vector_chunks
from .config import USER_AGENT, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS, EMBED_BASELINE_ON_FIRST_SNAPSHOT
from .embedder import embed_texts  # Removed chunk_text, keeping our semantic chunker

HEADERS = {"User-Agent": USER_AGENT}

# ── Quality thresholds ─────────────────────────────────────────────────────────
MIN_CLEAN_TEXT_LEN   = 1200  # Lowered back to 1200 so we don't skip short security bulletins!
MAX_CLEAN_TEXT_CHARS = 25_000 
REQUEST_TIMEOUT_S    = 30

# Tags whose *content* is pure navigation / chrome
_JUNK_TAGS = [
    "script", "style", "noscript", "svg", "canvas", "iframe",
    "button", "form", "input", "select", "textarea",
    "nav", "footer", "header", "aside",
    "devsite-nav", "devsite-toc", "devsite-language-selector", 
    "devsite-footer", "devsite-header", "devsite-feedback", 
    "devsite-filter", "devsite-snackbar"
]

# Role / class / id fragments that signal non-content containers
_JUNK_ATTRS = re.compile(
    r"(breadcrumb|cookie|consent|banner|promo|sidebar|toc-nav|"
    r"pagination|share|social|print|feedback|rating|related|"
    r"newsletter|subscribe|popup|modal|overlay|ads?|sponsor|"
    r"announcement|alert-bar|skip-link|widget|archive|devsite-page-nav|"
    r"usa-banner|usa-nav|usa-footer|language|lang)",
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
    r"Sign in to your Google Account.*?(?=\n\n|\Z)"
]

# ── Helpers ────────────────────────────────────────────────────────────────────

_JSON_BLOB_RE = re.compile(r"^\s*[\{\[](.|\n){80,}[\}\]]\s*$")

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def _is_junk_element(tag: Tag) -> bool:
    for attr in ("class", "id", "role", "aria-label"):
        val = tag.get(attr, "")
        if isinstance(val, list):
            val = " ".join(val)
        if _JUNK_ATTRS.search(val):
            return True
    return False

def _remove_junk_tags(root: Tag) -> None:
    for tag in root.find_all(_JUNK_TAGS):
        tag.decompose()

def _remove_junk_containers(root: Tag) -> None:
    for tag in root.find_all(True):
        if tag.name in ("div", "section", "span", "ul", "ol", "li") and _is_junk_element(tag):
            tag.decompose()

def _pick_root(soup: BeautifulSoup) -> Tag:
    candidates = [
        soup.find("div", {"class": re.compile(r"devsite-article-body", re.I)}),  
        soup.find("main", {"class": re.compile(r"devsite-main-content", re.I)}), 
        soup.find("devsite-content"),             
        soup.find("div", {"class": "post-body"}), 
        soup.find("article"),
        soup.find("main"),                        
        soup.find("div", {"id": re.compile(r"(content|main|article|body)", re.I)}),
        soup.find("div", {"class": re.compile(r"(devsite-article|article-body|page-content|"
                                               r"document-content|entry-content|post-content)", re.I)}),
        soup.find("div", {"role": "main"}),
        soup.find("body"),
        soup,
    ]
    return next((c for c in candidates if c is not None), soup)

def _table_to_text(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(separator=" ", strip=True) for td in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append("[TABLE_ROW] " + " | ".join(cells))
    return "\n".join(rows)

def _semantic_cleaner(root: Tag) -> str:
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

def _remove_json_blobs(text: str) -> str:
    out = []
    for line in text.splitlines():
        if _JSON_BLOB_RE.match(line):
            continue
        out.append(line)
    return "\n".join(out).strip()

def _remove_legal_noise(text: str) -> str:
    for pattern in _LEGAL_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()

def _deduplicate_lines(text: str) -> str:
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

_LANG_NOISE_RE = re.compile(
    r"(English\s*/\s*日本語\s*/\s*한국어\s*/\s*русск(ий|ий)\s*/\s*简体中文\s*/\s*繁體中文)",
    re.IGNORECASE,
)

def _remove_language_selector_lines(text: str) -> str:
    out: List[str] = []
    for line in text.splitlines():
        if _LANG_NOISE_RE.search(line):
            continue
        out.append(line)
    return "\n".join(out).strip()

def _cap_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.7)]
    tail = text[-int(max_chars * 0.3):]
    return head.rstrip() + "\n\n[...truncated...]\n\n" + tail.lstrip()

# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_raw_and_clean(url: str) -> Tuple[str, str]:
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
    clean_text = _remove_language_selector_lines(clean_text)
    
    clean_text = re.sub(r"\[SECTION: \]\n+", "", clean_text)
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()
    clean_text = _cap_text(clean_text, MAX_CLEAN_TEXT_CHARS)
    clean_text = _remove_json_blobs(clean_text) 
    return raw_html, clean_text

# ── Semantic Chunking ──────────────────────────────────────────────────────────

def semantic_chunk_text(text: str, max_chars: int = 1600, overlap_chars: int = 200) -> List[str]:
    paragraphs = re.split(r'\n{2,}', text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            for i in range(0, len(para), max_chars - overlap_chars):
                chunks.append(para[i:i + max_chars].strip())
            continue

        if len(current_chunk) + len(para) + 2 > max_chars:
            chunks.append(current_chunk.strip())
            
            overlap_text = current_chunk[-overlap_chars:] if len(current_chunk) > overlap_chars else current_chunk
            space_idx = overlap_text.find(" ")
            if space_idx != -1:
                overlap_text = overlap_text[space_idx:].strip()
            
            current_chunk = overlap_text + "\n\n" + para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# ── Vector storage ─────────────────────────────────────────────────────────────

def _store_vectors_for_snapshot(
    source_id_int: int,
    snapshot_id: int,
    snapshot_sha: str,
    kind: str,
    clean_text: str,
) -> int:
    
    chunks = semantic_chunk_text(
        clean_text,
        max_chars=CHUNK_SIZE_CHARS,
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
        .select("id,name,url")
        .eq("active", True)
        .execute()
        .data
    )

    print(f"🚀 Found {len(sources)} active sources. Starting ingestion...", flush=True)

    inserted = 0
    skipped  = 0
    embedded = 0

    for s in sources:
        src_id = int(s["id"])
        name   = s["name"]
        url    = s["url"]

        prev_latest      = get_latest_snapshot_for_source(src_id)
        is_first_snapshot = prev_latest is None

        try:
            raw_html, clean_text = fetch_raw_and_clean(url)
        except Exception as e:
            print(f"⚠️  Fetch failed for source='{name}': {e}", flush=True)
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
            "raw_text":     raw_html,
            "clean_text":   clean_text,
        }

        ins = sb.table("snapshots").insert(payload).execute()
        row = (ins.data or [{}])[0]
        snapshot_id = int(row.get("id") or 0)

        if snapshot_id <= 0:
            print(f"❌ Failed to get snapshot ID for '{name}'", flush=True)
            continue

        inserted += 1
        print(f"📦 Stored snapshot: {name} (ID: {snapshot_id})", flush=True)

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
        f"inserted={inserted}  skipped={skipped}  total_chunks={embedded}"
    )

if __name__ == "__main__":
    main()