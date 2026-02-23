# src/discover_ios_security_updates.py
# Self-contained discovery:
# - ensures Apple Security Updates hub (100100) exists in sources
# - crawls it to discover iOS-specific update pages
# - upserts discovered URLs into sources

import re
from datetime import datetime, timezone
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .db import get_supabase_client
from .config import USER_AGENT

HEADERS = {"User-Agent": USER_AGENT}

AGENT_NAME = "ios-risk-agent"
HUB_NAME = "Apple Security Updates Hub (100100)"
HUB_URL = "https://support.apple.com/en-us/100100"

# Tune this for dry run
TOP_N = 10

# Matches strings like:
# "iOS 18.7", "iOS 18.6.2", and optionally "iPadOS 18.6.2"
IOS_TITLE_RE = re.compile(r"\b(iOS|iPadOS)\s+(\d+(?:\.\d+){0,2})\b", re.IGNORECASE)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_hub_source(sb) -> int:
    now = _utc_now_iso()
    payload = {
        "agent_name": AGENT_NAME,
        "name": HUB_NAME,
        "url": HUB_URL,
        "fetch_type": "html",
        "active": True,
        "created_at": now,
    }

    # Requires UNIQUE on sources(url) - you already have this
    sb.table("sources").upsert(payload, on_conflict="url").execute()

    row = (
        sb.table("sources")
        .select("id")
        .eq("url", HUB_URL)
        .limit(1)
        .execute()
        .data
    )
    if not row:
        raise RuntimeError("Could not read back hub source id after upsert.")
    return int(row[0]["id"])


def _extract_candidate_links(html: str, base_url: str):
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("main") or soup.find("article") or soup.find("body") or soup

    links = []
    for a in root.find_all("a", href=True):
        text = " ".join(a.get_text(" ", strip=True).split())
        if not text:
            continue

        # Only keep links that mention iOS/iPadOS + a version
        m = IOS_TITLE_RE.search(text)
        if not m:
            continue

        href = a["href"].strip()
        abs_url = urljoin(base_url, href)

        # Capture normalized label like "iOS 18.6.2"
        os_name = m.group(1).lower()  # ios / ipados
        ver = m.group(2)
        label = f"{os_name.upper()} {ver}"

        links.append((label, abs_url, text))

    # Deduplicate by URL
    seen = set()
    uniq = []
    for label, url, full_text in links:
        if url in seen:
            continue
        seen.add(url)
        uniq.append((label, url, full_text))

    return uniq


def main():
    sb = get_supabase_client()
    now = _utc_now_iso()

    hub_source_id = _ensure_hub_source(sb)

    resp = requests.get(HUB_URL, headers=HEADERS, timeout=30, allow_redirects=True)
    resp.raise_for_status()

    candidates = _extract_candidate_links(resp.text or "", HUB_URL)
    if not candidates:
        raise RuntimeError("No iOS/iPadOS links found on 100100 hub page.")

    # Keep newest-ish first based on page order
    top = candidates[:TOP_N]

    upserted = 0
    for label, url, full_text in top:
        name = f"Apple Security Update - {label}"

        payload = {
            "agent_name": AGENT_NAME,
            "name": name,
            "url": url,
            "fetch_type": "html",
            "active": True,
            "created_at": now,

            # Optional: if you later add columns like parent_source_id, you can store it.
            # For now schema stays unchanged.
            # "parent_source_id": hub_source_id
        }

        sb.table("sources").upsert(payload, on_conflict="url").execute()
        upserted += 1

        print(f"Discovered source: {name} -> {url}", flush=True)

    print(f"✅ iOS security discovery done. hub_source_id={hub_source_id} upserted={upserted}", flush=True)


if __name__ == "__main__":
    main()
