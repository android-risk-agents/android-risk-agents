from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests

# -----------------------------
# Helpers
# -----------------------------


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if (v is not None and str(v).strip() != "") else default


def _debug_enabled() -> bool:
    return str(os.getenv("DEBUG_LLM", "")).strip().lower() in ("1", "true", "yes", "y")


# Precompiled regex for speed
_RE_ILLEGAL_CTRL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
_RE_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_RE_MISSING_COMMA_BEFORE_KEY = re.compile(r'(["}\]])\s*(")')


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
    return _RE_ILLEGAL_CTRL.sub("", s)


def _escape_newlines_inside_strings(s: str) -> str:
    """
    Convert raw newline / carriage-return / tab characters to escaped sequences ONLY when
    we are inside a quoted JSON string. Fixes invalid JSON from multi-line strings.
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


def _extract_json_block(s: str) -> str:
    """
    Extract a likely JSON block by trimming leading chatter and trailing chatter.
    This is not perfect, but it handles common cases where the model adds extra text.

    Strategy:
    - find first '{' or '['
    - slice from there
    - find last matching closing '}' or ']' and cut there
    """
    if not s:
        return s
    s = s.strip()

    i_obj = s.find("{")
    i_arr = s.find("[")
    if i_obj == -1 and i_arr == -1:
        return s

    start = i_obj if i_arr == -1 else (i_arr if i_obj == -1 else min(i_obj, i_arr))
    s2 = s[start:]

    if s2.startswith("{"):
        end = s2.rfind("}")
    else:
        end = s2.rfind("]")

    if end == -1:
        # Possibly truncated response, return what we have so decoder can report a useful error
        return s2

    return s2[: end + 1]


def _repair_common_json_syntax(s: str) -> str:
    """
    Repair only very common, low-risk issues:
    - trailing commas before } or ]
    - missing comma between a value-ending token and the next key quote
    """
    if not s:
        return s
    s = _RE_TRAILING_COMMA.sub(r"\1", s)
    s = _RE_MISSING_COMMA_BEFORE_KEY.sub(r"\1, \2", s)
    return s


def _json_error_context(s: str, pos: Optional[int], window: int = 250) -> str:
    if not s:
        return ""
    if pos is None:
        return s[: min(len(s), 900)]
    start = max(0, pos - window)
    end = min(len(s), pos + window)
    return s[start:end]


def _prepare_candidate_json(text: str) -> str:
    """
    Normalize candidate JSON string:
    - strip fences
    - remove illegal control chars
    - extract likely json block
    - escape newlines inside strings
    - repair common syntax issues
    """
    t = _strip_code_fences(text)
    t = _remove_illegal_control_chars(t)
    t = t.strip()
    t = _extract_json_block(t)
    t = _escape_newlines_inside_strings(t)
    t = _repair_common_json_syntax(t)
    return t.strip()


def _safe_json_loads(text: str) -> Tuple[Any, Optional[str]]:
    """
    Try json.loads with our normalization steps.
    Returns (obj, error_string). error_string is None on success.
    """
    candidate = _prepare_candidate_json(text)
    if not candidate:
        return {}, None

    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as e:
        ctx = _json_error_context(candidate, getattr(e, "pos", None))
        return None, f"{str(e)} | context: {ctx}"
    except Exception as e:
        return None, str(e)


def _extract_first_json_obj(text: str) -> Dict[str, Any]:
    """
    Extract and parse the first JSON object from model output.
    Keeps prior functionality:
    - handles dict
    - handles singleton list [ { ... } ]
    - handles list[dict] by returning first dict
    """
    obj, err = _safe_json_loads(text)
    if err is not None:
        raise json.JSONDecodeError(err, doc=text or "", pos=0)  # pos not reliable here

    obj = _unwrap_singleton_list(obj)

    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj[0]
    return {}


def _extract_content_from_nim_envelope(raw_text: str) -> str:
    """
    NIM returns an OpenAI-compatible envelope JSON. This function parses it
    robustly and extracts choices[0].message.content.
    """
    obj, err = _safe_json_loads(raw_text)
    if err is not None:
        raise json.JSONDecodeError(err, doc=raw_text or "", pos=0)

    if not isinstance(obj, dict):
        return ""

    choices = obj.get("choices") or []
    if not choices or not isinstance(choices, list):
        return ""

    first = choices[0] if isinstance(choices[0], dict) else {}
    msg = first.get("message") or {}
    if not isinstance(msg, dict):
        return ""

    content = msg.get("content") or ""
    return content if isinstance(content, str) else ""


# -----------------------------
# Client
# -----------------------------


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
        max_tokens: int = 1200,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calls NIM chat/completions and returns parsed JSON from assistant content.

        Improvements included:
        - Robust envelope parsing (no direct resp.json reliance)
        - Stronger JSON extraction, newline escaping inside quoted strings
        - Low-risk JSON repairs for near-JSON outputs
        - Debug logging includes targeted context on JSON decode failures
        - Efficient normalization with precompiled regex
        - Retry strategy keeps max_retries functionality but reduces wasted retries:
          local parse repair first, then optional model retry with temperature=0
        """
        rid = request_id or "req"
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        base_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_err: Optional[str] = None

        # We keep the original max_retries behavior, but:
        # - For each HTTP success response, we attempt local parse fixes before deciding to retry the network call.
        # - On the first parse failure, we flip temperature to 0.0 for subsequent attempts (if not already).
        payload = dict(base_payload)

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
                if self.debug:
                    print(f"[NIM] status={resp.status_code} attempt={attempt} rid={rid}")

                if resp.status_code >= 400:
                    head = _remove_illegal_control_chars((resp.text or "")[:900])
                    if self.debug:
                        print(f"[NIM] http_error_head rid={rid}: {head}")
                    last_err = f"HTTP {resp.status_code}: {head}"
                    time.sleep(0.8 * attempt)
                    continue

                raw_text = resp.text or ""
                if self.debug:
                    print(f"[NIM] resp_text_head rid={rid}: {_remove_illegal_control_chars(raw_text[:900])}")

                # Parse NIM envelope safely
                try:
                    content = _extract_content_from_nim_envelope(raw_text)
                except json.JSONDecodeError as e:
                    last_err = str(e)
                    if self.debug:
                        print(f"[NIM] envelope_json_error rid={rid}: {last_err[:400]}")
                    time.sleep(0.8 * attempt)
                    continue

                if self.debug:
                    print(f"[NIM] content_head rid={rid}: {_remove_illegal_control_chars(content)[:900]}")

                # Parse assistant content as JSON safely
                try:
                    parsed = _extract_first_json_obj(content)
                except json.JSONDecodeError as e:
                    # Log context from our normalized candidate, not just raw content head
                    if self.debug:
                        candidate = _prepare_candidate_json(content)
                        print(f"[NIM] json_error rid={rid}: {str(e)[:400]}")
                        # Provide a readable slice for debugging. If error string already contains context,
                        # this is still useful and deterministic.
                        print(f"[NIM] json_candidate_head rid={rid}: {candidate[:900]}")
                    last_err = str(e)

                    # Reduce odds of repeated malformed JSON by forcing temperature=0 on next attempts
                    if payload.get("temperature", 0.0) != 0.0:
                        payload["temperature"] = 0.0

                    time.sleep(0.8 * attempt)
                    continue

                if self.debug:
                    print(f"[NIM] parsed_keys rid={rid}: {list(parsed.keys())}")

                return parsed

            except Exception as e:
                last_err = str(e)
                if self.debug:
                    print(f"[NIM] exception rid={rid}: {last_err[:400]}")
                time.sleep(0.8 * attempt)

        raise RuntimeError(f"NIM request failed after {self.max_retries} attempts: {last_err}")