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
    # Remove: 0x00-0x08, 0x0B-0x0C, 0x0E-0x1F
    if not s:
        return s
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", s)


def _escape_newlines_inside_strings(s: str) -> str:
    """
    Convert raw newline / carriage-return characters to escaped sequences ONLY when
    we are inside a quoted JSON string. This fixes "Invalid control character" from
    models that emit multi-line strings.
    """
    if not s:
        return s

    out: list[str] = []
    in_str = False
    esc = False

    for ch in s:
        if esc:
            out.append(ch)
            esc = False
            continue

        if ch == "\\":
            out.append(ch)
            esc = True
            continue

        if ch == '"':
            out.append(ch)
            in_str = not in_str
            continue

        if in_str:
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue

        out.append(ch)

    return "".join(out)


def _unwrap_singleton_list(obj: Any) -> Any:
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        return obj[0]
    return obj


def _extract_first_json_obj(text: str) -> Dict[str, Any]:
    t = _strip_code_fences(text)
    t = _remove_illegal_control_chars(t)
    t = t.strip()
    if not t:
        return {}

    # Try direct parse first
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

    # Fix common model issue: multiline strings
    t2 = _escape_newlines_inside_strings(t)

    # Try again
    try:
        obj = json.loads(t2)
        obj = _unwrap_singleton_list(obj)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
        return {}
    except Exception:
        pass

    # raw_decode: find first { or [
    start_obj = t2.find("{")
    start_arr = t2.find("[")
    if start_obj == -1 and start_arr == -1:
        raise ValueError("No JSON object found in model output.")

    start = start_obj if start_arr == -1 else (start_arr if start_obj == -1 else min(start_obj, start_arr))
    t3 = t2[start:]

    dec = json.JSONDecoder()
    obj, _idx = dec.raw_decode(t3)
    obj = _unwrap_singleton_list(obj)

    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj[0]
    return {}


@dataclass
class NimClient:
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
        rid = request_id or "req"
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
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
                    head = (resp.text or "")[:900]
                    head = _remove_illegal_control_chars(head)
                    if self.debug:
                        print(f"[NIM] http_error_head rid={rid}: {head}")
                    last_err = f"HTTP {resp.status_code}: {head}"
                    time.sleep(0.8 * attempt)
                    continue

                raw_text = resp.text or ""
                if self.debug:
                    print(f"[NIM] resp_text_head rid={rid}: {_remove_illegal_control_chars(raw_text[:900])}")

                clean_text = _remove_illegal_control_chars(raw_text)
                data = json.loads(clean_text)

                content = (
                    (((data.get("choices") or [{}])[0]).get("message") or {}).get("content")
                    or ""
                )

                if self.debug:
                    print(f"[NIM] content_head rid={rid}: {_remove_illegal_control_chars(content)[:900]}")

                parsed = _extract_first_json_obj(content)

                if self.debug:
                    print(f"[NIM] parsed_keys rid={rid}: {list(parsed.keys())}")

                return parsed

            except Exception as e:
                last_err = str(e)
                if self.debug:
                    print(f"[NIM] exception rid={rid}: {last_err[:400]}")
                time.sleep(0.8 * attempt)

        raise RuntimeError(f"NIM request failed after {self.max_retries} attempts: {last_err}")