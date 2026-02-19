from __future__ import annotations

import os
import json
import requests
from typing import Dict, Any, List


class NimClient:
    def __init__(self):
        self.api_key = os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing NVIDIA_API_KEY")

        self.base_url = "https://integrate.api.nvidia.com/v1/chat/completions"

    def chat_json(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> Dict[str, Any]:

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        resp = requests.post(self.base_url, json=payload, headers=headers, timeout=90)

        if resp.status_code != 200:
            raise RuntimeError(f"NIM error {resp.status_code}: {resp.text}")

        data = resp.json()

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        return self._extract_json_only(content)

    @staticmethod
    def _extract_json_only(text: str) -> Dict[str, Any]:
        text = (text or "").strip()

        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1].replace("json", "", 1).strip()

        try:
            return json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1:
                raise
            return json.loads(text[start:end + 1])