# src/discover_bulletins.py
#
# Discovers Android Security Bulletin URLs at two levels:
#
#   Level 0  (index page)  - The monthly bulletin index (asb-overview).
#                            Finds the TOP_N most recent month-level URLs,
#                            e.g. /docs/security/bulletin/2026/2026-03-01
#
#   Level 1  (derived)     - For each month URL, derives the standard pair of
#                            patch-level sub-pages deterministically:
#                              - YYYY-MM-01  (base patch level)
#                              - YYYY-MM-05  (extended patch level, always the 5th)
#                            Google has used this exact pattern consistently
#                            since 2016. No scraping needed - the bulletin page
#                            is JS-rendered so requests only gets the nav shell.
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

# How many recent months to process. Keep at 2-3; historical bulletins
# rarely change and each month adds 2 sources (month page + -05 patch level).
TOP_N = 2

# Polite delay between HTTP requests to source.android.com
REQUEST_DELAY_S = 1.0

# ── Regex patterns ───────────────────────────────────────────────────────────────

# Matches month-name links on the index page, e.g. "March 2026"
_MONTH_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}",
    re.I,
)

# Extracts YYYY-MM from a bulletin URL path like /2026/2026-03-01
_YEAR_MONTH_RE = re.compile(r"/(\d{4})-(\d{2})-\d{2}(?:/|$)")

# ── Helpers ──────────────────────────────────────────────────────────────────────


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
    resp.raise_for_status()
    return resp.text or ""


def _dedupe(links: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen: set = set()
    out = []
    for name, url in links:
        if url not in seen:
            seen.add(url)
            out.append((name, url))
    return out


# ── Level 0: extract month-level bulletin links from the index page ──────────────


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

        # Skip Pixel/Nexus sub-bulletins
        if "/pixel/" in href.lower():
            continue

        text = " ".join(a.get_text(" ", strip=True).split())
        if not text or not _MONTH_RE.search(text):
            continue

        abs_url = urljoin(base_url, href)
        links.append((text, abs_url))

    return _dedupe(links)


# ── Level 1: derive patch-level sub-page URLs deterministically ──────────────────


def _derive_patch_level_urls(
    month_url: str,
    month_label: str,
) -> List[Tuple[str, str]]:
    """
    Given a month-level bulletin URL like:
        https://source.android.com/docs/security/bulletin/2026/2026-03-01

    The bulletin page is JS-rendered - requests only gets the nav shell, so
    scraping for sub-links is not possible. Instead we derive the -05 patch
    level URL deterministically. Google has used this exact YYYY-MM-01 /
    YYYY-MM-05 split consistently since 2016.

    The month page URL itself is already the -01 patch level page, so we only
    register the -05 variant as an additional source.

    Returns (descriptive_name, absolute_url) for the -05 sub-page only.
    """
    parsed = urlparse(month_url)
    path = parsed.path.rstrip("/")

    m = _YEAR_MONTH_RE.search(path)
    if not m:
        return []

    year  = m.group(1)   # e.g. "2026"
    month = m.group(2)   # e.g. "03"

    # Directory portion of the path, e.g. /docs/security/bulletin/2026/
    path_dir = path[: path.rfind("/") + 1]
    base = f"{parsed.scheme}://{parsed.netloc}"

    return [
        (
            f"Android Security Bulletin - {month_label} ({year}-{month}-05 patch level)",
            f"{base}{path_dir}{year}-{month}-05",
        )
    ]


# ── Main ─────────────────────────────────────────────────────────────────────────


def main():
    sb = get_supabase_client()
    now = _utc_now_iso()

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

    print(f"📡 Fetching bulletin index: {index_url}", flush=True)
    index_html = _fetch_html(index_url)
    month_links = _extract_month_bulletin_links(index_html, index_url)

    if not month_links:
        raise RuntimeError("No month bulletin links found on bulletin index page.")

    top_months = month_links[:TOP_N]
    print(f"🗓  Found {len(month_links)} month links; processing top {TOP_N}.", flush=True)

    total_registered = 0

    for month_title, month_url in top_months:
        month_label = month_title

        # Register the month-level page (this IS the -01 patch level page)
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

        # Derive and register the -05 patch level variant
        sub_pages = _derive_patch_level_urls(month_url, month_label)
        for sub_name, sub_url in sub_pages:
            sb.table("sources").upsert(
                {
                    "agent_name": AGENT_NAME,
                    "name": sub_name,
                    "url": sub_url,
                    "fetch_type": "html",
                    "active": True,
                    "created_at": now,
                },
                on_conflict="url",
            ).execute()
            total_registered += 1
            print(f"    ↳ Patch level registered: {sub_name}", flush=True)

        time.sleep(REQUEST_DELAY_S)

    print(
        f"\n✅ Bulletin discovery done. "
        f"months_processed={len(top_months)}  total_sources_registered={total_registered}",
        flush=True,
    )


if __name__ == "__main__":
    main()