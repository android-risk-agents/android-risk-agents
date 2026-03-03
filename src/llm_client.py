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

# Safe fixes for common LLM JSON glitches:
_RE_COMMA_COLON_BEFORE_QUOTE = re.compile(r",\s*:\s*(\")")  # ,: "key"
_RE_COMMA_COLON_BEFORE_CLOSE = re.compile(r",\s*:\s*([}\]])")  # ,: } or ,: ]
_RE_DANGLING_COMMA_COLON_EOF = re.compile(r",\s*:\s*$")  # ,: <EOF>
_RE_DANGLING_QUOTE_COMMA_COLON_CLOSE = re.compile(r'"\s*,\s*:\s*([}\]])')  # "...",: } or ]


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


def _extract_between_markers(
    text: str, start: str = "<<<JSON>>>", end: str = "<<<ENDJSON>>>"
) -> str:
    """
    Extract payload between markers.
    Tolerant to missing start or end marker, which happens with some NIM generations.
    """
    if not text:
        return text

    t = (text or "").strip()
    i = t.find(start)
    j = t.rfind(end)

    # Normal case: both markers exist
    if i != -1 and j != -1 and j > i:
        return t[i + len(start) : j].strip()

    # END exists but START doesn't: strip END and continue
    if i == -1 and j != -1:
        return t[:j].strip()

    # START exists but END doesn't: take everything after START
    if i != -1 and j == -1:
        return t[i + len(start) :].strip()

    return t


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


def _extract_json_object_block(s: str) -> str:
    """
    Extract a likely JSON OBJECT block (not array):
    - find first '{'
    - slice from there
    - find last '}' and cut there
    """
    if not s:
        return s
    s = s.strip()

    i = s.find("{")
    if i == -1:
        return s  # let downstream raise

    s2 = s[i:]
    end = s2.rfind("}")
    if end == -1:
        return s2  # possibly truncated
    return s2[: end + 1]


def _repair_common_json_syntax(s: str) -> str:
    """
    Repair only very common, low-risk issues LLMs emit.
    These fixes do not invent content, they only remove illegal punctuation patterns.
    """
    if not s:
        return s

    # Special: "...",: }  -> "..." }
    s = _RE_DANGLING_QUOTE_COMMA_COLON_CLOSE.sub(r'"\1', s)

    # Remove trailing commas: ", }" or ", ]"
    s = _RE_TRAILING_COMMA.sub(r"\1", s)

    # Fix stray comma-colon glitch: ,: "key"
    s = _RE_COMMA_COLON_BEFORE_QUOTE.sub(r", \1", s)

    # Fix stray comma-colon before closing: ,: } or ,: ]
    s = _RE_COMMA_COLON_BEFORE_CLOSE.sub(r"\1", s)

    # Fix stray comma-colon at end of string: ,: <EOF>
    s = _RE_DANGLING_COMMA_COLON_EOF.sub("", s)

    # Insert missing comma between a value-ending token and the next key quote
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


def _looks_truncated_or_non_object(candidate: str) -> bool:
    if not candidate:
        return True
    c = candidate.strip()
    if not c.startswith("{"):
        return True
    if c.count("{") > c.count("}"):
        return True
    if not c.endswith("}"):
        return True
    return False


def _prepare_candidate_json(text: str) -> str:
    """
    Normalize candidate JSON string.
    """
    t = _extract_between_markers(text)
    t = _strip_code_fences(t)
    t = _remove_illegal_control_chars(t)
    t = (t or "").strip()
    t = _extract_json_object_block(t)
    t = _escape_newlines_inside_strings(t)
    t = _repair_common_json_syntax(t)
    t = _repair_common_json_syntax(t)  # second deterministic pass
    return (t or "").strip()


def _safe_json_loads(text: str) -> Tuple[Any, Optional[str]]:
    candidate = _prepare_candidate_json(text)
    if not candidate:
        return {}, None

    if _looks_truncated_or_non_object(candidate):
        return None, "Truncated or non-object JSON detected"

    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as e:
        ctx = _json_error_context(candidate, getattr(e, "pos", None))
        return None, f"{str(e)} | context: {ctx}"
    except Exception as e:
        return None, str(e)


def _extract_first_json_obj(text: str) -> Dict[str, Any]:
    obj, err = _safe_json_loads(text)
    if err is not None:
        raise json.JSONDecodeError(err, doc=text or "", pos=0)

    obj = _unwrap_singleton_list(obj)

    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj[0]
    return {}


def _extract_content_from_nim_envelope(raw_text: str) -> str:
    """
    NIM returns an OpenAI-compatible envelope JSON.
    Extract choices[0].message.content.
    """
    cleaned = _remove_illegal_control_chars(raw_text or "")
    data = json.loads(cleaned)
    content = (
        (((data.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
    )
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
        """
        rid = request_id or "req"

        # ✅ FIX: normalize base_url so both forms work:
        # - https://integrate.api.nvidia.com
        # - https://integrate.api.nvidia.com/v1
        base = (self.base_url or "").rstrip("/")
        if base.endswith("/v1"):
            url = f"{base}/chat/completions"
        else:
            url = f"{base}/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": f"{system}\n\n{user}"}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_err: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                if self.debug:
                    print(f"[NIM] url={url}")

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

                # Envelope parse
                try:
                    content = _extract_content_from_nim_envelope(raw_text)
                except Exception as e:
                    last_err = str(e)
                    if self.debug:
                        print(f"[NIM] envelope_error rid={rid}: {last_err[:400]}")
                    time.sleep(0.8 * attempt)
                    continue

                if self.debug:
                    print(f"[NIM] content_head rid={rid}: {_remove_illegal_control_chars(content)[:900]}")

                # Content parse
                try:
                    parsed = _extract_first_json_obj(content)
                except json.JSONDecodeError as e:
                    last_err = str(e)
                    if self.debug:
                        candidate = _prepare_candidate_json(content)
                        print(f"[NIM] json_error rid={rid}: {last_err[:400]}")
                        print(f"[NIM] json_candidate_head rid={rid}: {candidate[:900]}")

                    # Force deterministic output on subsequent tries
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


# ✅ UPDATED: respects LLM_BASE_URL (you will set it from secrets.NVIDIA_LLM_URL in Actions)
def get_llm_client() -> NimClient:
    base_url = os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    return NimClient(base_url=base_url)


def chat_json(
    client: NimClient,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 400,
) -> Dict[str, Any]:
    return client.chat_json(
        model=model,
        system=system,
        user=user,
        temperature=temperature,
        max_tokens=max_tokens,
    )