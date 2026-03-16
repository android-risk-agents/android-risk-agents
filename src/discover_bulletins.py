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

# Regex to find "Month Year" (e.g., "March 2026")
MONTH_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
    re.I,
)

def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def _is_within_last_14_months(match_obj) -> bool:
    """
    Mathematically calculates if the matched 'Month Year' is strictly within 
    the rolling 14-month window from the current execution date.
    """
    month_str = match_obj.group(1)
    year_str = match_obj.group(2)
    
    try:
        date_obj = datetime.strptime(f"{month_str} {year_str}", "%B %Y")
        now = datetime.now()
        
        # Calculate the absolute difference in months
        months_diff = (now.year - date_obj.year) * 12 + now.month - date_obj.month
        
        # 0 = Current month, 14 = Exactly 14 months ago
        return 0 <= months_diff <= 14
    except ValueError:
        return False

def _crawl_android_bulletins(base_url: str) -> list:
    """Crawls Android Security Bulletins and strictly filters for the last 14 months."""
    print(f"🕸️ Crawling Android Security Bulletins: {base_url}")
    resp = requests.get(base_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    
    links = []
    for a in soup.find_all("a", href=True):
        text = " ".join(a.get_text(" ", strip=True).split())
        href = a["href"].strip()
        
        match = MONTH_RE.search(text)
        if match and _is_within_last_14_months(match):
            abs_url = urljoin(base_url, href)
            links.append((f"Android Security Bulletin - {text}", abs_url))
            
    return _deduplicate(links)

def _crawl_android_dev_blog() -> list:
    """Crawls the Android Developers Blog homepage for recent article links."""
    base_url = "https://android-developers.googleblog.com/"
    print(f"🕸️ Crawling Android Dev Blog: {base_url}")
    resp = requests.get(base_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    
    links = []
    # Regex to find standard Google Blogspot URL structures (e.g., /2026/02/article-name.html)
    blog_url_re = re.compile(r"/\d{4}/\d{2}/.*\.html$")
    
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if blog_url_re.search(href):
            text = " ".join(a.get_text(" ", strip=True).split())
            # Filter out empty or generic "Read more" link text
            if text and len(text) > 10: 
                links.append((f"Android Dev Blog - {text}", href))
                
    return _deduplicate(links)

def _crawl_cisa_kev() -> list:
    """
    Maps the CISA Known Exploited Vulnerabilities (KEV) Catalog.
    Since CISA KEV is a constantly updating, authoritative JSON feed, 
    we map directly to it rather than scraping arbitrary HTML links.
    """
    print("🕸️ Mapping CISA KEV Catalog...")
    return [("CISA Known Exploited Vulnerabilities Catalog", "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json")]

def _deduplicate(links_list: list) -> list:
    """Removes duplicate URLs found during the crawl."""
    seen = set()
    uniq = []
    for text, url in links_list:
        if url not in seen:
            seen.add(url)
            uniq.append((text, url))
    return uniq

def main():
    sb = get_supabase_client()
    now = _utc_now_iso()

    # 1. Android Bulletins (Last 14 Months)
    bulletin_base_url = "https://source.android.com/docs/security/bulletin"
    bulletin_links = _crawl_android_bulletins(bulletin_base_url)
    
    # 2. Android Dev Blog
    blog_links = _crawl_android_dev_blog()
    
    # 3. CISA Catalog
    cisa_links = _crawl_cisa_kev()

    # Combine all discovered threat intel sources
    all_links = bulletin_links + blog_links + cisa_links
    
    if not all_links:
        raise RuntimeError("No links found during the crawl.")

    print(f"🔍 Compiling list of {len(all_links)} active sources to database...")

    inserted = 0
    for name, url in all_links:
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
        inserted += 1

    print(f"✅ Discovery complete. Tracked and updated {inserted} targeted sources.", flush=True)

if __name__ == "__main__":
    main()