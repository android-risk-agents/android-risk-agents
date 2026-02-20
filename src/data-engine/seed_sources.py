# src/seed_sources.py

from datetime import datetime, timezone
from .db import get_supabase_client

AGENT_NAME = "android-risk-agent"

SOURCES = [
    {"name": "Android Security Bulletins", "url": "https://source.android.com/docs/security/bulletin/asb-overview", "fetch_type": "html"},
    {"name": "Android Developers Blog", "url": "https://android-developers.googleblog.com/", "fetch_type": "html"},
    {"name": "Google Play Developer Policy Center", "url": "https://play.google/developer-content-policy/", "fetch_type": "html"},
    {"name": "Play Integrity API Docs", "url": "https://developer.android.com/google/play/integrity", "fetch_type": "html"},
    {"name": "CISA KEV Catalog", "url": "https://www.cisa.gov/known-exploited-vulnerabilities-catalog", "fetch_type": "html"},
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

        # Requires UNIQUE constraint on sources(url)
        sb.table("sources").upsert(payload, on_conflict="url").execute()

    print("✅ Sources seeded/updated (idempotent via upsert).", flush=True)


if __name__ == "__main__":
    main()