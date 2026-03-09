# src/seed_sources.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime, timezone
from .db import get_supabase_client

AGENT_NAME = "android-risk-agent"

# --- 1. STATIC SOURCES ---
STATIC_SOURCES = [
    {"name": "Android Security Bulletins Overview", "url": "https://source.android.com/docs/security/bulletin/asb-overview", "fetch_type": "html"},
    {"name": "Android Developers Blog", "url": "https://android-developers.googleblog.com/", "fetch_type": "html"},
    {"name": "Google Play Developer Policy Center", "url": "https://play.google/developer-content-policy/", "fetch_type": "html"},
    {"name": "Play Integrity API Docs", "url": "https://developer.android.com/google/play/integrity", "fetch_type": "html"},
    {"name": "CISA KEV Catalog", "url": "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json", "fetch_type": "html"}
]

# --- 2. DYNAMIC CRAWLER ---
MAIN_BULLETIN_URL = "https://source.android.com/docs/security/bulletin"

def discover_bulletins(base_url: str) -> list:
    """
    Crawls the main overview page to find links to recent monthly bulletins (2025 and 2026).
    """
    print(f"🕸️ Crawling for dynamic links at: {base_url}")
    try:
        response = requests.get(base_url, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"⚠️ Failed to crawl {base_url}: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    discovered_sources = []
    seen_urls = set()
    
    for link in soup.find_all("a", href=True):
        href = link.get("href", "")
        
        # 🚨 FILTER: Only keep 2025 and 2026 links!
        if "/security/bulletin/2025" in href or "/security/bulletin/2026" in href:
            full_url = urljoin(base_url, href)
            
            if full_url != base_url and full_url not in seen_urls:
                seen_urls.add(full_url)
                
                # Extract the date for the database name (e.g., 2026-03-01)
                name_part = full_url.split("/")[-1] 
                discovered_sources.append({
                    "name": f"Android Security Bulletins-{name_part}",
                    "url": full_url,
                    "fetch_type": "html"
                })
                
    return discovered_sources

# --- 3. MAIN EXECUTION ---
def main():
    sb = get_supabase_client()
    now = datetime.now(timezone.utc).isoformat()

    print("🚀 Starting Hybrid Seeding Process (2025-2026 only)...")
    
    # Run the crawler to find relevant monthly bulletins
    dynamic_sources = discover_bulletins(MAIN_BULLETIN_URL)
    print(f"✅ Found {len(dynamic_sources)} dynamic monthly bulletins.")

    # Combine static sites with the newly discovered dynamic links
    ALL_SOURCES = STATIC_SOURCES + dynamic_sources

    for s in ALL_SOURCES:
        payload = {
            "agent_name": AGENT_NAME,
            "name": s["name"],
            "url": s["url"],
            "fetch_type": s["fetch_type"],
            "active": True,
            "created_at": now,
        }

        # Requires UNIQUE constraint on sources(url)
        sb.table("sources").upsert(payload, on_conflict="url").execute()

    print(f"✅ {len(ALL_SOURCES)} total sources seeded/updated (idempotent via upsert).", flush=True)

if __name__ == "__main__":
    main()