from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


@dataclass
class LocalGenerationResult:
    answer: Optional[str]
    status: str
    error: Optional[str]
    model: str
    prompt_chars: int


class LocalLLMService:
    """Local-only text generation using an Ollama-compatible HTTP endpoint."""

    def __init__(self) -> None:
        self.enabled = os.getenv("LOCAL_LLM_ENABLED", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self.base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
        self.model = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b-instruct").strip()
        self.timeout_seconds = float(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "45"))
        self.temperature = float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.1"))
        self.top_p = float(os.getenv("LOCAL_LLM_TOP_P", "0.9"))
        self.max_tokens = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "220"))

    def _request_json(
        self,
        *,
        path: str,
        payload: dict,
        timeout: float,
    ) -> dict:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return json.loads(body)

    def generate(self, prompt: str, timeout_seconds: Optional[float] = None) -> LocalGenerationResult:
        if not self.enabled:
            return LocalGenerationResult(
                answer=None,
                status="disabled",
                error=None,
                model=self.model,
                prompt_chars=len(prompt),
            )

        payload = {
            "model": self.model,
            "prompt": str(prompt),
            "stream": False,
            "options": {
                "temperature": float(self.temperature),
                "top_p": float(self.top_p),
                "num_predict": int(self.max_tokens),
            },
        }
        timeout = float(timeout_seconds) if timeout_seconds is not None else float(self.timeout_seconds)
        try:
            parsed = self._request_json(path="/api/generate", payload=payload, timeout=timeout)
            answer = str(parsed.get("response") or "").strip()
            if not answer:
                # Some Ollama models expose usable content on /api/chat even when /api/generate
                # returns an empty "response" field.
                chat_payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": str(prompt)}],
                    "stream": False,
                    "options": payload["options"],
                }
                parsed = self._request_json(path="/api/chat", payload=chat_payload, timeout=timeout)
                message = parsed.get("message") if isinstance(parsed, dict) else None
                if isinstance(message, dict):
                    answer = str(message.get("content") or "").strip()
            if not answer:
                return LocalGenerationResult(
                    answer=None,
                    status="empty_response",
                    error=None,
                    model=self.model,
                    prompt_chars=len(prompt),
                )
            return LocalGenerationResult(
                answer=answer,
                status="ok",
                error=None,
                model=self.model,
                prompt_chars=len(prompt),
            )
        except urllib.error.URLError as e:
            return LocalGenerationResult(
                answer=None,
                status="unavailable",
                error=f"{type(e).__name__}: {e}",
                model=self.model,
                prompt_chars=len(prompt),
            )
        except Exception as e:
            return LocalGenerationResult(
                answer=None,
                status="error",
                error=f"{type(e).__name__}: {e}",
                model=self.model,
                prompt_chars=len(prompt),
            )
