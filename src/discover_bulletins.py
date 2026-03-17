# src/discover_bulletins.py
#
# Discovers Android Security Bulletin URLs at two depths:
#
#   Depth 0  (index page)  - The monthly bulletin index (asb-overview).
#                            Finds the TOP_N most recent month-level URLs,
#                            e.g. /docs/security/bulletin/2026/2026-03-01
#
#   Depth 1  (bulletin page) - For each month-level URL, fetches the full
#                              bulletin and extracts every hyperlink that
#                              represents a sub-page worth scraping:
#                                - Patch-level variant pages  (2026-03-01, 2026-03-05)
#                                - Any other /security/bulletin/* sub-pages
#                                - AOSP commit/diff links (android.googlesource.com)
#                                - Android CVE tracker entries (issuetracker.google.com)
#
# All discovered URLs are upserted as active sources so scrape_sources.py
# picks them up on the next run.

import re
import time
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse
from typing import List, Tuple

import requests
from bs4 import BeautifulSoup

from .db import get_supabase_client
from .config import USER_AGENT

HEADERS = {"User-Agent": USER_AGENT}

AGENT_NAME = "android-risk-agent"
BULLETIN_INDEX_NAME = "Android Security Bulletins"

# How many recent months to deep-scrape. Keep low (2-3) to stay within
# GitHub Actions timeouts; historical months rarely change.
TOP_N = 2

# Seconds to wait between HTTP requests to be polite to source.android.com
REQUEST_DELAY_S = 1.0

# Maximum number of sub-links to register per bulletin page (safety cap).
MAX_SUB_LINKS_PER_BULLETIN = 30

# ── Regex patterns ──────────────────────────────────────────────────────────────

# Matches month-name links on the index page, e.g. "March 2026"
_MONTH_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}",
    re.I,
)

# Matches patch-level date slugs in a URL path, e.g. /2026-03-01 or /2026-03-05
_PATCH_DATE_RE = re.compile(r"/(\d{4}-\d{2}-\d{2})(?:/|$)")

# Matches any URL that is a child of the /security/bulletin/ tree
_BULLETIN_PATH_RE = re.compile(r"/docs/security/bulletin/", re.I)

# AOSP Gerrit / Googlesource commit URLs
_AOSP_COMMIT_RE = re.compile(
    r"https?://(android\.googlesource\.com|googlesource\.com)/.*\+/",
    re.I,
)

# Google Issue Tracker - Android CVE tracker entries
_ISSUE_TRACKER_RE = re.compile(
    r"https?://issuetracker\.google\.com/issues/\d+",
    re.I,
)

# ── Helpers ─────────────────────────────────────────────────────────────────────


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
    resp.raise_for_status()
    return resp.text or ""


def _dedupe(links: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Deduplicate (name, url) pairs by URL, preserving order."""
    seen: set = set()
    out = []
    for name, url in links:
        if url not in seen:
            seen.add(url)
            out.append((name, url))
    return out


# ── Depth-0: extract month-level bulletin links from the index page ─────────────


def _extract_month_bulletin_links(html: str, base_url: str) -> List[Tuple[str, str]]:
    """
    Scan the ASB overview page for links whose visible text contains a month
    name and year (e.g. "March 2026"). Returns (display_title, absolute_url).
    Skips Pixel/Nexus bulletin links (they contain /pixel/ in the href).
    """
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("main") or soup.find("article") or soup.find("body") or soup

    links: List[Tuple[str, str]] = []
    for a in root.find_all("a", href=True):
        href = a["href"].strip()

        # Skip Pixel/Nexus sub-bulletins at this depth
        if "/pixel/" in href.lower():
            continue

        text = " ".join(a.get_text(" ", strip=True).split())
        if not text or not _MONTH_RE.search(text):
            continue

        abs_url = urljoin(base_url, href)
        links.append((text, abs_url))

    return _dedupe(links)


# ── Depth-1: extract sub-links from a single bulletin page ──────────────────────


def _extract_bulletin_sub_links(
    html: str,
    bulletin_url: str,
    month_label: str,
) -> List[Tuple[str, str]]:
    """
    Given the HTML of a month-level bulletin page, find all links that point
    to content worth ingesting as separate sources:

      1. Patch-level variant pages on source.android.com
         e.g.  /docs/security/bulletin/2026/2026-03-01
               /docs/security/bulletin/2026/2026-03-05
      2. Any other /docs/security/bulletin/* pages linked from this bulletin
      3. AOSP Gerrit commit links (android.googlesource.com)
      4. Google Issue Tracker CVE entries (issuetracker.google.com)

    Returns (descriptive_name, absolute_url) pairs.
    """
    soup = BeautifulSoup(html, "html.parser")
    base = "https://source.android.com"

    sub_links: List[Tuple[str, str]] = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue

        text = " ".join(a.get_text(" ", strip=True).split())
        abs_url = urljoin(bulletin_url, href)
        parsed = urlparse(abs_url)

        # 1 + 2. Patch-level / sibling bulletin pages on source.android.com
        if parsed.netloc in ("source.android.com", "") and _BULLETIN_PATH_RE.search(parsed.path):
            # Skip Pixel bulletins (user preference)
            if "/pixel/" in parsed.path.lower():
                continue
            # Skip the index / overview pages - we only want leaf bulletin pages
            if parsed.path.rstrip("/").endswith(("bulletin", "asb-overview", "bulletin/")):
                continue
            # Must look like a dated leaf: contains a date slug in the path
            if not _PATCH_DATE_RE.search(parsed.path):
                continue

            m = _PATCH_DATE_RE.search(parsed.path)
            date_slug = m.group(1) if m else "unknown"
            name = f"Android Security Bulletin - {month_label} ({date_slug})"
            sub_links.append((name, abs_url))
            continue

        # 3. AOSP Gerrit commit links
        if _AOSP_COMMIT_RE.match(abs_url):
            # Use the path tail as a short identifier
            path_tail = parsed.path.rstrip("/").split("/")[-1][:60]
            name = f"AOSP Commit [{month_label}] {path_tail}"
            sub_links.append((name, abs_url))
            continue

        # 4. Google Issue Tracker CVE entries
        if _ISSUE_TRACKER_RE.match(abs_url):
            issue_id = parsed.path.rstrip("/").split("/")[-1]
            name = f"CVE Tracker [{month_label}] #{issue_id}"
            sub_links.append((name, abs_url))
            continue

    deduped = _dedupe(sub_links)
    return deduped[:MAX_SUB_LINKS_PER_BULLETIN]


# ── Main ─────────────────────────────────────────────────────────────────────────


def main():
    sb = get_supabase_client()
    now = _utc_now_iso()

    # Fetch the index source row so we know the starting URL
    idx = (
        sb.table("sources")
        .select("id,name,url")
        .eq("name", BULLETIN_INDEX_NAME)
        .limit(1)
        .execute()
        .data
    )

    if not idx:
        raise RuntimeError(f"Index source not found: name='{BULLETIN_INDEX_NAME}'")

    index_url = idx[0]["url"]

    # ── Depth 0: get TOP_N most recent month-level bulletins ────────────────────
    print(f"📡 Fetching bulletin index: {index_url}", flush=True)
    index_html = _fetch_html(index_url)
    month_links = _extract_month_bulletin_links(index_html, index_url)

    if not month_links:
        raise RuntimeError("No month bulletin links found on bulletin index page.")

    top_months = month_links[:TOP_N]
    print(f"🗓  Found {len(month_links)} month links; processing top {TOP_N}.", flush=True)

    total_registered = 0

    for month_title, month_url in top_months:
        month_label = month_title  # e.g. "March 2026"

        # Upsert the month-level bulletin page itself as a source
        sb.table("sources").upsert(
            {
                "agent_name": AGENT_NAME,
                "name": f"Android Security Bulletin - {month_label}",
                "url": month_url,
                "fetch_type": "html",
                "active": True,
                "created_at": now,
            },
            on_conflict="url",
        ).execute()
        total_registered += 1
        print(f"  ✅ Month page registered: {month_label} -> {month_url}", flush=True)

        # ── Depth 1: fetch the bulletin and extract all sub-links ───────────────
        time.sleep(REQUEST_DELAY_S)
        try:
            bulletin_html = _fetch_html(month_url)
        except Exception as e:
            print(f"  ⚠️  Could not fetch bulletin page for {month_label}: {e}", flush=True)
            continue

        sub_links = _extract_bulletin_sub_links(bulletin_html, month_url, month_label)
        print(
            f"  🔍 Found {len(sub_links)} sub-links inside {month_label} bulletin.",
            flush=True,
        )

        for sub_name, sub_url in sub_links:
            # Decide fetch_type: AOSP / Issue Tracker links are external HTML;
            # source.android.com bulletin pages are also HTML.
            # (No JSON sources discovered at depth-1.)
            fetch_type = "html"

            sb.table("sources").upsert(
                {
                    "agent_name": AGENT_NAME,
                    "name": sub_name,
                    "url": sub_url,
                    "fetch_type": fetch_type,
                    "active": True,
                    "created_at": now,
                },
                on_conflict="url",
            ).execute()
            total_registered += 1
            print(f"    ↳ {sub_name}", flush=True)

        time.sleep(REQUEST_DELAY_S)

    print(
        f"\n✅ Bulletin discovery done. "
        f"months_processed={len(top_months)}  total_sources_registered={total_registered}",
        flush=True,
    )


if __name__ == "__main__":
    main()