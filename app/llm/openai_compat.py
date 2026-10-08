"""OpenAI-compatible LOCAL API provider (LM Studio, llama-server и т.п.).

Наследует подход model/lmstudio.py из Grafix: chat/completions, авто-выбор
загруженной модели, таймауты. Строго localhost/LAN — внешние облачные API
запрещены (ТЗ §95).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import httpx

from app.config import AppConfig
from app.core.models import ModelInfo
from app.llm.base import LLMProvider, LLMUnavailableError

log = logging.getLogger(__name__)


class OpenAICompatProvider(LLMProvider):
    name = "openai_compat"

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.base_url = cfg.llm.openai_compat_base_url.rstrip("/")
        self.model = cfg.llm.openai_compat_model
        if not _is_local(self.base_url):
            log.warning("openai_compat base_url не является локальным адресом: %s", self.base_url)

    def health_check(self) -> tuple[bool, str]:
        try:
            r = httpx.get(f"{self.base_url}/models", timeout=8.0)
            if r.status_code == 200:
                data = r.json().get("data", [])
                if not self.model and data:
                    self.model = data[0].get("id", "")
                return True, f"Модель: {self.model or 'по умолчанию'}"
            return False, f"HTTP {r.status_code}"
        except Exception as e:
            return False, f"Сервер недоступен: {type(e).__name__}"

    def complete(self, messages: list[dict], *, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        payload = {
            "messages": messages,
            "temperature": self.cfg.llm.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.cfg.llm.max_tokens,
        }
        if self.model:
            payload["model"] = self.model
        try:
            r = httpx.post(f"{self.base_url}/chat/completions", json=payload,
                           timeout=self.cfg.llm.request_timeout_s)
        except httpx.HTTPError as e:
            raise LLMUnavailableError(
                f"Локальная LLM ({self.name}) недоступна: {e}. "
                "Проверьте, что запущен LM Studio / llama-server."
            ) from e
        if r.status_code != 200:
            raise LLMUnavailableError(f"LLM API HTTP {r.status_code}: {r.text[:200]}")
        try:
            return r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise LLMUnavailableError("Некорректный ответ LLM API") from e

    def model_info(self) -> ModelInfo:
        return ModelInfo(backend=self.name, model=self.model or "(auto)", version="")


def _is_local(url: str) -> bool:
    try:
        host = httpx.URL(url).host or ""
    except Exception:
        return False
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"} or host.startswith("127.")
