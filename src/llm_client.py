# src/llm_client.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

from openai import OpenAI


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if (v is not None and str(v).strip() != "") else default


def get_llm_client() -> OpenAI:
    api_key = _env("LLM_API_KEY") or _env("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing LLM_API_KEY (or GROQ_API_KEY). Add it as a secret/env var.")
    base_url = _env("LLM_BASE_URL") or _env("GROQ_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def extract_json_only(s: str) -> Dict[str, Any]:
    s = (s or "").strip()

    # Handle accidental fenced blocks
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1].replace("json", "", 1).strip()

    try:
        return json.loads(s)
    except Exception:
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(s[start : end + 1])


def chat_json(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 400,
) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content or "{}"
    return extract_json_only(content)