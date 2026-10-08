"""Ollama local provider (http://127.0.0.1:11434). Только локальный сервер."""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from app.config import AppConfig
from app.core.models import ModelInfo
from app.llm.base import LLMProvider, LLMUnavailableError

log = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.base_url = cfg.llm.ollama_url.rstrip("/")
        self.model = cfg.llm.ollama_model

    def health_check(self) -> tuple[bool, str]:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=8.0)
            if r.status_code != 200:
                return False, f"HTTP {r.status_code}"
            models = [m.get("name", "") for m in r.json().get("models", [])]
            if models and self.model not in models:
                return False, f"Модель '{self.model}' не найдена. Доступны: {', '.join(models)}"
            return True, f"Модель: {self.model}"
        except Exception as e:
            return False, f"Ollama недоступен: {type(e).__name__}"

    def complete(self, messages: list[dict], *, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.cfg.llm.temperature if temperature is None else temperature,
                "num_predict": max_tokens or self.cfg.llm.max_tokens,
            },
        }
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=payload,
                           timeout=self.cfg.llm.request_timeout_s)
        except httpx.HTTPError as e:
            raise LLMUnavailableError(
                f"Ollama недоступен ({self.base_url}): {e}. Запустите `ollama serve`."
            ) from e
        if r.status_code != 200:
            raise LLMUnavailableError(f"Ollama HTTP {r.status_code}: {r.text[:200]}")
        return r.json().get("message", {}).get("content", "")

    def model_info(self) -> ModelInfo:
        return ModelInfo(backend=self.name, model=self.model, version="")
