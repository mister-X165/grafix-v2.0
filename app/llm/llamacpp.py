"""llama.cpp provider — основной backend (ТЗ §5).

Два режима (оба полностью локальные):
1. ``llama-server`` (рекомендуется): HTTP-эндпоинт с OpenAI-compatible API —
   используется автоматически, если сервер отвечает на llama_server_url.
2. Python-биндинги ``llama-cpp-python``: прямая загрузка GGUF-модели
   (cfg.llm.model_path). Опциональная зависимость; при отсутствии — понятное
   сообщение и предложение запустить llama-server (§51).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

import httpx

from app.config import AppConfig
from app.core.models import ModelInfo
from app.llm.base import LLMProvider, LLMUnavailableError

log = logging.getLogger(__name__)


class LlamaCppProvider(LLMProvider):
    name = "llama.cpp"

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.server_url = cfg.llm.llama_server_url.rstrip("/")
        self.model_path = cfg.llm.model_path
        self._llm: Any = None
        self._lock = threading.Lock()
        self._mode: str = ""  # server | bindings | ""

    # ------------------------------------------------------------- modes ---
    def _server_alive(self) -> bool:
        try:
            r = httpx.get(f"{self.server_url}/health", timeout=4.0)
            if r.status_code == 200:
                return True
            r = httpx.get(f"{self.server_url}/v1/models", timeout=4.0)
            return r.status_code == 200
        except Exception:
            return False

    def _load_bindings(self) -> bool:
        if self._llm is not None:
            return True
        try:
            from llama_cpp import Llama  # опциональная зависимость
        except ImportError:
            return False
        p = Path(self.model_path)
        if not p.is_absolute():
            p = self.cfg.root / p
        if not p.exists():
            log.error("GGUF model not found: %s", p)
            return False
        with self._lock:
            try:
                self._llm = Llama(model_path=str(p), n_ctx=8192, verbose=False)
            except Exception as e:
                log.error("llama.cpp model load failed: %s", e)
                return False
        return True

    # ------------------------------------------------------------ API -----
    def health_check(self) -> tuple[bool, str]:
        if self._server_alive():
            self._mode = "server"
            return True, f"llama-server на {self.server_url}"
        if self.model_path and self._load_bindings():
            self._mode = "bindings"
            return True, f"GGUF загружен: {Path(self.model_path).name}"
        return False, (
            "llama.cpp недоступен: llama-server не отвечает, а модель GGUF не "
            "загружена. Запустите llama-server или укажите llm.model_path в "
            "config/config.yaml (и установите llama-cpp-python)."
        )

    def complete(self, messages: list[dict], *, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        if not self._mode:
            ok, msg = self.health_check()
            if not ok:
                raise LLMUnavailableError(msg)
        temp = self.cfg.llm.temperature if temperature is None else temperature
        mt = max_tokens or self.cfg.llm.max_tokens
        if self._mode == "server":
            try:
                r = httpx.post(f"{self.server_url}/v1/chat/completions",
                               json={"messages": messages, "temperature": temp,
                                     "max_tokens": mt},
                               timeout=self.cfg.llm.request_timeout_s)
            except httpx.HTTPError as e:
                raise LLMUnavailableError(f"llama-server недоступен: {e}") from e
            if r.status_code != 200:
                raise LLMUnavailableError(f"llama-server HTTP {r.status_code}: {r.text[:200]}")
            return r.json()["choices"][0]["message"]["content"]
        # bindings
        assert self._llm is not None
        try:
            res = self._llm.create_chat_completion(messages=messages, temperature=temp,
                                                   max_tokens=mt)
            return res["choices"][0]["message"]["content"]
        except LLMUnavailableError:
            raise
        except Exception as e:
            raise LLMUnavailableError(f"Ошибка инференса llama.cpp: {e}") from e

    def model_info(self) -> ModelInfo:
        model = Path(self.model_path).name if self.model_path else "llama-server"
        return ModelInfo(backend=self.name, model=model, version=self._mode)
