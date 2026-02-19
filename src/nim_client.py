from __future__ import annotations

import os
import json
import time
import random
from json import JSONDecoder
from typing import Dict, Any, Optional

import requests


class NimClient:
    """
    NIM OpenAI-compatible chat/completions client with hardened JSON extraction
    + GitHub Actions-visible debug logging.

    Only secret required: NVIDIA_API_KEY
    Endpoint + model are hardcoded in code (as discussed).
    """

    def __init__(self):
        self.api_key = os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing NVIDIA_API_KEY")

        # Hardcoded NIM endpoint
        self.chat_url = "https://integrate.api.nvidia.com/v1/chat/completions"

        self.timeout_s = int(os.getenv("NIM_TIMEOUT_S", "90"))
        self.retries = int(os.getenv("NIM_RETRIES", "3"))

        self.debug = os.getenv("DEBUG_LLM", "").strip() in ("1", "true", "TRUE", "yes", "YES")

    def chat_json(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 400,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err: Optional[Exception] = None

        for attempt in range(1, self.retries + 1):
            try:
                resp = requests.post(
                    self.chat_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_s,
                )

                if self.debug:
                    rid = f" rid={request_id}" if request_id else ""
                    print(f"[NIM] status={resp.status_code} attempt={attempt}{rid}")

                if resp.status_code != 200:
                    txt = resp.text or ""
                    if self.debug:
                        print(f"[NIM] error_body_head: {txt[:1200]}")
                    raise RuntimeError(f"NIM error {resp.status_code}: {txt[:1200]}")

                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                ) or ""

                if self.debug:
                    rid = f" rid={request_id}" if request_id else ""
                    print(f"[NIM] content_head{rid}: {content[:1200]}")

                parsed = self._extract_json_only(content)

                if self.debug:
                    rid = f" rid={request_id}" if request_id else ""
                    keys = list(parsed.keys()) if isinstance(parsed, dict) else []
                    print(f"[NIM] parsed_keys{rid}: {keys}")

                return parsed

            except Exception as e:
                last_err = e
                if self.debug:
                    rid = f" rid={request_id}" if request_id else ""
                    print(f"[NIM] exception{rid}: {type(e).__name__}: {str(e)[:300]}")

                if attempt >= self.retries:
                    break

                sleep_s = min(10.0, 1.5 * (2 ** (attempt - 1)))
                sleep_s *= (0.85 + random.random() * 0.3)
                time.sleep(sleep_s)

        raise RuntimeError(f"NIM request failed after {self.retries} attempts: {last_err}") from last_err

    @staticmethod
    def _strip_fences(text: str) -> str:
        t = (text or "").strip()
        if "```" not in t:
            return t

        parts = t.split("```")
        for p in parts:
            p2 = p.strip()
            if "{" in p2 and "}" in p2:
                if p2.lower().startswith("json"):
                    p2 = p2[4:].strip()
                return p2

        return t.replace("```json", "").replace("```", "").strip()

    @classmethod
    def _extract_json_only(cls, text: str) -> Dict[str, Any]:
        """
        Robustly parse the FIRST JSON object from the model output.
        Avoids 'Extra data' by using JSONDecoder.raw_decode.
        """
        t = cls._strip_fences(text)

        try:
            obj = json.loads(t)
            if not isinstance(obj, dict):
                raise ValueError("Expected top-level JSON object")
            return obj
        except Exception:
            pass

        start = t.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in model output: {t[:300]}")

        decoder = JSONDecoder()
        obj, _ = decoder.raw_decode(t[start:])

        if not isinstance(obj, dict):
            raise ValueError("Expected top-level JSON object")

        return obj