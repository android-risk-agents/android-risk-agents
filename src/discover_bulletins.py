# src/discover_bulletins.py

import re
from datetime import datetime, timezone
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .db import get_supabase_client
from .config import USER_AGENT

HEADERS = {"User-Agent": USER_AGENT}

AGENT_NAME = "android-risk-agent"
BULLETIN_INDEX_NAME = "Android Security Bulletins"

MONTH_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}",
    re.I,
)

TOP_N = 2  # keep minimal; change to 3 if you want


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _extract_month_bulletin_links(html: str, base_url: str):
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("main") or soup.find("article") or soup.find("body") or soup

    links = []
    for a in root.find_all("a", href=True):
        text = " ".join(a.get_text(" ", strip=True).split())
        href = a["href"].strip()

        if not text:
            continue
        if not MONTH_RE.search(text):
            continue

        abs_url = urljoin(base_url, href)
        links.append((text, abs_url))

    # Deduplicate by URL
    seen = set()
    uniq = []
    for t, u in links:
        if u in seen:
            continue
        seen.add(u)
        uniq.append((t, u))

    return uniq


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

    base_url = idx[0]["url"]

    resp = requests.get(base_url, headers=HEADERS, timeout=30, allow_redirects=True)
    resp.raise_for_status()

    links = _extract_month_bulletin_links(resp.text or "", base_url)

    if not links:
        raise RuntimeError("No month bulletin links found on bulletin index page.")

    top = links[:TOP_N]

    for title, url in top:
        name = f"Android Security Bulletin - {title}"

        payload = {
            "agent_name": AGENT_NAME,
            "name": name,
            "url": url,
            "fetch_type": "html",
            "active": True,
            "created_at": now,
        }

        # Requires UNIQUE constraint on sources(url)
        sb.table("sources").upsert(payload, on_conflict="url").execute()

    print(f"✅ Bulletin discovery done. tracked={len(top)}", flush=True)


if __name__ == "__main__":
    main()