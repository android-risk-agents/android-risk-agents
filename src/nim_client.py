from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if (v is not None and str(v).strip() != "") else default


def _debug_enabled() -> bool:
    return str(os.getenv("DEBUG_LLM", "")).strip().lower() in ("1", "true", "yes", "y")


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 2:
            inner = parts[1].strip()
            if inner.lower().startswith("json"):
                inner = inner[4:].strip()
            return inner
    return s


def _remove_illegal_control_chars(s: str) -> str:
    """
    JSON does not allow certain ASCII control chars unescaped inside strings.
    We remove the illegal ones to avoid: Invalid control character at ...
    Keep: tab(0x09), newline(0x0A), carriage return(0x0D) because they are valid
    when represented properly. If they appear unescaped, removing illegal ones
    still helps most malformed outputs.
    """
    if not s:
        return s
    # Remove: 0x00-0x08, 0x0B-0x0C, 0x0E-0x1F
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", s)


def _unwrap_singleton_list(obj: Any) -> Any:
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        return obj[0]
    return obj


def _extract_first_json_obj(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object/array from a string, using JSONDecoder.raw_decode,
    after stripping code fences and removing illegal control chars.
    """
    t = _strip_code_fences(text)
    t = _remove_illegal_control_chars(t).strip()
    if not t:
        return {}

    # Fast path: exact JSON
    try:
        obj = json.loads(t)
        obj = _unwrap_singleton_list(obj)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
        return {}
    except Exception:
        pass

    # raw_decode path: find first { or [
    start_obj = t.find("{")
    start_arr = t.find("[")
    if start_obj == -1 and start_arr == -1:
        raise ValueError("No JSON object found in model output.")

    start = start_obj if start_arr == -1 else (start_arr if start_obj == -1 else min(start_obj, start_arr))
    t2 = t[start:]

    dec = json.JSONDecoder()
    obj, _idx = dec.raw_decode(t2)
    obj = _unwrap_singleton_list(obj)

    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj[0]
    return {}


@dataclass
class NimClient:
    """
    NVIDIA NIM client. Endpoint and model are hardcoded for testing.
    Only NVIDIA_API_KEY is required as env.
    """
    base_url: str = "https://integrate.api.nvidia.com/v1"
    timeout_s: int = 60
    max_retries: int = 3

    def __post_init__(self) -> None:
        api_key = _env("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError("Missing NVIDIA_API_KEY env var / secret.")
        self.api_key = api_key
        self.debug = _debug_enabled()

    def chat_json(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 600,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calls NIM chat.completions and returns a parsed JSON dict.
        """
        rid = request_id or "req"
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_err: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
                if self.debug:
                    print(f"[NIM] status={resp.status_code} attempt={attempt} rid={rid}")

                if resp.status_code >= 400:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:500]}"
                    time.sleep(0.8 * attempt)
                    continue

                data = resp.json()
                content = (
                    (((data.get("choices") or [{}])[0]).get("message") or {}).get("content")
                    or ""
                )

                if self.debug:
                    head = _remove_illegal_control_chars(content)[:800]
                    print(f"[NIM] content_head rid={rid}: {head}")

                parsed = _extract_first_json_obj(content)

                if self.debug:
                    print(f"[NIM] parsed_keys rid={rid}: {list(parsed.keys())}")

                return parsed

            except Exception as e:
                last_err = str(e)
                time.sleep(0.8 * attempt)

        raise RuntimeError(f"NIM request failed after {self.max_retries} attempts: {last_err}")