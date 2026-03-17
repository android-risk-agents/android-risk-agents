# src/seed_sources.py

from datetime import datetime, timezone
from .db import get_supabase_client

AGENT_NAME = "android-risk-agent"

SOURCES = [
    # ── Android ecosystem sources ──────────────────────────────────────────────
    {
        "name": "Android Security Bulletins",
        "url": "https://source.android.com/docs/security/bulletin/asb-overview",
        "fetch_type": "html",
    },
    {
        # Changed from html to rss - the blog is JS-rendered as HTML but
        # publishes a clean static Atom feed with full post content.
        "name": "Android Developers Blog",
        "url": "https://android-developers.googleblog.com/atom.xml",
        "fetch_type": "rss",
    },
    {
        "name": "Google Play Developer Policy Center",
        "url": "https://play.google/developer-content-policy/",
        "fetch_type": "html",
    },
    {
        "name": "Play Integrity API Docs",
        "url": "https://developer.android.com/google/play/integrity",
        "fetch_type": "html",
    },
    # ── CISA KEV: official JSON feed (HTML catalog blocks automated requests) ──
    {
        "name": "CISA KEV Feed",
        "url": "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
        "fetch_type": "json",
    },
    # ── NVD CVE API 2.0: Android CVEs modified in last 30 days ────────────────
    # fetch_type="api_nvd" triggers fetch_nvd_and_clean() in scrape_sources.py
    # which appends lastModStartDate/lastModEndDate dynamically each run.
    # The URL here is just a semantic label used for logging; the actual
    # endpoint is hardcoded in fetch_nvd_and_clean() to keep seed_sources clean.
    {
        "name": "NVD CVE Feed - Android",
        "url": "https://services.nvd.nist.gov/rest/json/cves/2.0",
        "fetch_type": "api_nvd",
    },
    # ── OSV Android bulk vulnerability database (Google GCS, no auth) ─────────
    # fetch_type="api_osv" triggers fetch_osv_and_clean() which downloads the
    # full Android ecosystem ZIP and renders the newest OSV_MAX_RECORDS entries.
    {
        "name": "OSV Android Vulnerability Database",
        "url": "https://osv-vulnerabilities.storage.googleapis.com/Android/all.zip",
        "fetch_type": "api_osv",
    },
]


def main():
    sb = get_supabase_client()
    now = datetime.now(timezone.utc).isoformat()

    for s in SOURCES:
        payload = {
            "agent_name": AGENT_NAME,
            "name": s["name"],
            "url": s["url"],
            "fetch_type": s["fetch_type"],
            "active": True,
            "created_at": now,
        }

        # Idempotent: upsert on unique URL constraint.
        # The old CISA HTML row (different URL) will remain in the DB but
        # should be set to active=False manually or via a one-time migration.
        sb.table("sources").upsert(payload, on_conflict="url").execute()

    print("✅ Sources seeded/updated (idempotent via upsert).", flush=True)


if __name__ == "__main__":
    main()