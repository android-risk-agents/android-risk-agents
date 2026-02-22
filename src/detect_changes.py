# src/detect_changes.py
from __future__ import annotations

import os
from datetime import datetime, timezone
import difflib
from typing import List, Dict, Any

from .db import (
    get_supabase_client,
    get_snapshot_text_and_hash_by_id,
    upsert_vector_chunks,
)
from .config import CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS, EMBED_DELTAS_ON_CHANGE
from .embedder import chunk_text, embed_texts

# First-run demo: treat first snapshot as a "baseline init change"
INIT_BASELINE_AS_CHANGE = os.getenv("INIT_BASELINE_AS_CHANGE", "false").lower() == "true"
BASELINE_MAX_SOURCES = int(os.getenv("BASELINE_MAX_SOURCES", "10"))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _delta_added_text(old_text: str, new_text: str) -> str:
    """
    Optimized diffing using unified_diff. 
    Orders of magnitude faster than ndiff for large documents.
    """
    old_lines = (old_text or "").splitlines()
    new_lines = (new_text or "").splitlines()

    # n=0 means we don't return unchanged context lines, just the raw differences
    diff = difflib.unified_diff(old_lines, new_lines, n=0, lineterm="")
    
    out_lines: List[str] = []
    for ln in diff:
        # Ignore diff metadata headers (+++, ---, @@)
        if ln.startswith("+") and not ln.startswith("+++"):
            # Strip the leading '+' and save the actual text
            added_text = ln[1:].strip()
            if added_text:
                out_lines.append(added_text)

    return "\n".join(out_lines).strip()


def _embed_delta(source_id_int: int, snapshot_id: int, snapshot_sha: str, delta_text: str) -> int:
    chunks = chunk_text(delta_text, chunk_size_chars=CHUNK_SIZE_CHARS, overlap_chars=CHUNK_OVERLAP_CHARS)
    if not chunks:
        return 0

    embs = embed_texts(chunks)
    rows: List[Dict[str, Any]] = [
        {
            "source_id": str(source_id_int),
            "snapshot_sha": str(snapshot_sha),
            "kind": "delta",
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


def main():
    sb = get_supabase_client()
    now = _utc_now_iso()

    sources = (
        sb.table("sources")
        .select("id,name,priority")
        .eq("active", True)
        .order("priority", desc=True)
        .execute()
        .data
    )

    baseline_created = 0

    for src in sources:
        src_id = int(src["id"])
        name = src["name"]

        snaps = (
            sb.table("snapshots")
            .select("id,content_hash,fetched_at")
            .eq("source_id", src_id)
            .order("fetched_at", desc=True)
            .limit(2)
            .execute()
            .data
        )

        if not snaps:
            continue

        # 1. Handle Baseline Initialization (Only 1 snapshot exists)
        if len(snaps) == 1:
            if not INIT_BASELINE_AS_CHANGE:
                continue

            if baseline_created >= BASELINE_MAX_SOURCES:
                # FIX: Use 'continue' instead of 'break' so we don't skip evaluating 
                # REAL changes in subsequent sources just because we hit the baseline cap.
                continue

            latest = snaps[0]
            latest_id = int(latest["id"])

            payload = {
                "source_id": src_id,
                "prev_snapshot_id": None, 
                "new_snapshot_id": latest_id,
                "diff_json": {
                    "type": "baseline_init",
                    "new_hash": latest["content_hash"],
                },
                "created_at": now,
                "status": "new",
            }

            try:
                sb.table("changes").upsert(payload, on_conflict="source_id,new_snapshot_id").execute()
                baseline_created += 1
                print(f"🟦 Baseline-init change created for {name}", flush=True)
            except Exception as e:
                print(f"❌ Failed to create baseline for {name}: {e}", flush=True)
            continue

        # 2. Handle Normal Content Changes (2 snapshots exist)
        latest, previous = snaps

        # If hashes match, nothing changed. Move to next source.
        if latest["content_hash"] == previous["content_hash"]:
            continue

        payload = {
            "source_id": src_id,
            "prev_snapshot_id": int(previous["id"]),
            "new_snapshot_id": int(latest["id"]),
            "diff_json": {
                "type": "content_change",
                "prev_hash": previous["content_hash"],
                "new_hash": latest["content_hash"],
            },
            "created_at": now,
            "status": "new",
        }

        try:
            sb.table("changes").upsert(payload, on_conflict="source_id,new_snapshot_id").execute()
            print(f"🚨 Change detected for {name}", flush=True)
        except Exception as e:
            print(f"❌ Failed to record change for {name}: {e}", flush=True)
            continue # If we can't record the change, don't try to embed the delta

        if not EMBED_DELTAS_ON_CHANGE:
            continue

        # 3. Extract and Embed the Delta
        try:
            old_text, _old_hash = get_snapshot_text_and_hash_by_id(int(previous["id"]))
            new_text, new_hash = get_snapshot_text_and_hash_by_id(int(latest["id"]))
            
            delta = _delta_added_text(old_text, new_text)

            if not delta:
                print(f"⏩ No actual text added for {name} (formatting/deletion only).", flush=True)
                continue

            nvec = _embed_delta(
                source_id_int=src_id,
                snapshot_id=int(latest["id"]),
                snapshot_sha=new_hash,
                delta_text=delta,
            )
            print(f"✨ Embedded {nvec} delta chunks for {name}", flush=True)

        except Exception as e:
            print(f"❌ Delta embed failed for {name}: {e}", flush=True)

    print(f"\n✅ detect_changes done. baseline_init_created={baseline_created}", flush=True)


if __name__ == "__main__":
    main()