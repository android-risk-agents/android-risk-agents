# src/detect_changes.py
from __future__ import annotations

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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _delta_added_text(old_text: str, new_text: str, max_chars: int = 12000) -> str:
    old_lines = (old_text or "").splitlines()
    new_lines = (new_text or "").splitlines()

    out_lines: List[str] = []
    for ln in difflib.ndiff(old_lines, new_lines):
        if ln.startswith("+ "):
            out_lines.append(ln[2:])

    delta = "\n".join(out_lines).strip()
    if not delta:
        return ""

    if len(delta) > max_chars:
        delta = delta[: int(max_chars * 0.8)] + "\n\n[.delta truncated.]\n\n" + delta[-int(max_chars * 0.2) :]

    return delta


def _embed_delta(source_id_int: int, snapshot_id: int, snapshot_sha: str, delta_text: str) -> int:
    chunks = chunk_text(delta_text, chunk_size_chars=CHUNK_SIZE_CHARS, overlap_chars=CHUNK_OVERLAP_CHARS)
    if not chunks:
        return 0

    embs = embed_texts(chunks)
    rows: List[Dict[str, Any]] = []
    for i, (ch, emb) in enumerate(zip(chunks, embs)):
        rows.append(
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
        )

    upsert_vector_chunks(rows)
    return len(rows)


def main():
    sb = get_supabase_client()
    now = _utc_now_iso()

    sources = (
        sb.table("sources")
        .select("id,name")
        .eq("active", True)
        .execute()
        .data
    )

    for src in sources:
        src_id = int(src["id"])
        snaps = (
            sb.table("snapshots")
            .select("id,content_hash,fetched_at")
            .eq("source_id", src_id)
            .order("fetched_at", desc=True)
            .limit(2)
            .execute()
            .data
        )

        if len(snaps) < 2:
            continue

        latest, previous = snaps

        if latest["content_hash"] == previous["content_hash"]:
            continue

        payload = {
            "source_id": src_id,
            "prev_snapshot_id": int(previous["id"]),
            "new_snapshot_id": int(latest["id"]),
            "diff_json": {
                "prev_hash": previous["content_hash"],
                "new_hash": latest["content_hash"],
            },
            "created_at": now,
            "status": "new",
        }

        sb.table("changes").upsert(payload, on_conflict="source_id,new_snapshot_id").execute()
        print(f"🚨 Change detected for {src['name']}", flush=True)

        if not EMBED_DELTAS_ON_CHANGE:
            continue

        try:
            old_text, _old_hash = get_snapshot_text_and_hash_by_id(int(previous["id"]))
            new_text, new_hash = get_snapshot_text_and_hash_by_id(int(latest["id"]))
            delta = _delta_added_text(old_text, new_text)

            if not delta:
                print(f"No delta text extracted for {src['name']} (hash changed, delta empty).", flush=True)
                continue

            nvec = _embed_delta(
                source_id_int=src_id,
                snapshot_id=int(latest["id"]),
                snapshot_sha=new_hash,
                delta_text=delta,
            )
            print(f"Embedded {nvec} delta chunks for {src['name']}", flush=True)

        except Exception as e:
            print(f"Delta embed failed for {src['name']}: {e}", flush=True)


if __name__ == "__main__":
    main()