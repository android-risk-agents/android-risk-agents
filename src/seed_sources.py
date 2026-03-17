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
        "name": "Android Developers Blog",
        "url": "https://android-developers.googleblog.com/",
        "fetch_type": "html",
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
    # ── CISA KEV: use official JSON feed instead of the HTML catalog page ──────
    # The HTML catalog (cisa.gov/known-exploited-vulnerabilities-catalog) blocks
    # automated requests with a 403. CISA publishes the full dataset as a
    # machine-readable JSON file at the URL below; it updates within minutes of
    # any addition to the canonical catalog and is CC0 (public domain).
    # fetch_type="json" is handled by scrape_sources.fetch_json_and_clean().
    {
        "name": "CISA KEV Feed",
        "url": "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
        "fetch_type": "json",
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