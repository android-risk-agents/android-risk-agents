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
    Minimal NIM client that returns JSON dicts.
    Hardcoded endpoint by default, only secret needed is NVIDIA_API_KEY.
    """

    def __init__(self):
        self.api_key = os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing NVIDIA_API_KEY")

        # Hardcode NIM chat endpoint here (as discussed)
        self.chat_url = "https://integrate.api.nvidia.com/v1/chat/completions"

        # Optional tuning knobs (can stay as defaults)
        self.timeout_s = int(os.getenv("NIM_TIMEOUT_S", "90"))
        self.retries = int(os.getenv("NIM_RETRIES", "3"))

    def chat_json(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 400,
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

                if resp.status_code != 200:
                    raise RuntimeError(f"NIM error {resp.status_code}: {resp.text[:1200]}")

                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )

                return self._extract_json_only(content)

            except Exception as e:
                last_err = e
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
        Parses the FIRST JSON object found (avoids JSONDecodeError: Extra data).
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