"""LLM abstraction layer (ТЗ §5, §41–§56).

`LLMProvider` — интерфейс; реализации: llama.cpp (основной), Ollama,
OpenAI-compatible local API (LM Studio). Облачные LLM API запрещены (§95).

`structured_generate` — цикл parse → validate (Pydantic) → repair → retry
с ограниченным числом попыток (§55). Ответы LLM никогда не принимаются на веру.
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import AppConfig, get_config
from app.core.models import ModelInfo

log = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)


class LLMUnavailableError(RuntimeError):
    """LLM недоступна — понятное сообщение для UI (§51)."""


class StructuredOutputError(RuntimeError):
    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


# ----------------------------------------------------------- JSON utils -----

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}|\[[\s\S]*\]")


def extract_json(text: str) -> Any:
    """Достать первый JSON-объект/массив из ответа LLM (терпимо к markdown)."""
    t = (text or "").strip()
    # убрать ```json ... ```
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"```\s*$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK_RE.search(t)
    if not m:
        raise ValueError("В ответе нет JSON")
    frag = m.group(0)
    try:
        return json.loads(frag)
    except json.JSONDecodeError:
        # попытка починить распространённые огрехи: висячие запятые, обрезанный хвост
        fixed = re.sub(r",\s*([}\]])", r"\1", frag)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return json.loads(_balance(fixed))


def _balance(s: str) -> str:
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    out = s
    if in_str:
        out += '"'
    for ch in reversed(stack):
        out += "}" if ch == "{" else "]"
    return out


# ------------------------------------------------------------- provider -----

class LLMProvider(ABC):
    """Интерфейс локальной LLM."""

    name: str = "base"

    @abstractmethod
    def health_check(self) -> tuple[bool, str]:
        """(ready, сообщение). Проверяется при старте и перед анализом."""

    @abstractmethod
    def complete(self, messages: list[dict], *, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        """Один completion-вызов. Бросает LLMUnavailableError при проблемах."""

    @abstractmethod
    def model_info(self) -> ModelInfo: ...

    # ---------------------------------------- structured generation --------
    def structured_generate(self, messages: list[dict], schema: Type[TModel],
                            *, retries: Optional[int] = None,
                            max_tokens: Optional[int] = None) -> TModel:
        """parse → validate → repair → retry (§55, §56).

        При провале валидации отправляем модели ошибку и просим исправить JSON.
        """
        cfg = get_config()
        attempts = (cfg.llm.max_retries if retries is None else retries) + 1
        last_raw = ""
        last_err = ""
        msgs = list(messages)
        for attempt in range(attempts):
            try:
                raw = self.complete(msgs, max_tokens=max_tokens, temperature=0.0)
            except LLMUnavailableError:
                raise
            except Exception as e:  # сеть таймаут и т.п. — retry
                last_err = f"Ошибка генерации: {e}"
                log.warning("llm attempt %d failed: %s", attempt + 1, e)
                continue
            last_raw = raw
            try:
                data = extract_json(raw)
                if isinstance(data, list):
                    data = {list_key_for(schema): data}
                return schema.model_validate(data)
            except (ValueError, ValidationError, KeyError) as e:
                last_err = str(e)
                log.warning("invalid structured output (attempt %d): %s", attempt + 1,
                            last_err[:400])
                msgs = messages + [
                    {"role": "assistant", "content": raw[:2000]},
                    {"role": "user", "content":
                        f"Ответ не прошёл валидацию: {last_err[:800]}\n"
                        "Верни ИСПРАВЛЕННЫЙ JSON, соответствующий схеме, без пояснений."},
                ]
        raise StructuredOutputError(
            f"LLM не смогла выдать валидный JSON после {attempts} попыток: {last_err[:300]}",
            raw=last_raw,
        )


def list_key_for(schema: Type[BaseModel]) -> str:
    """Соглашение: списочные ответы оборачиваются в поле с именем списка."""
    fields = list(schema.model_fields)
    return fields[0] if fields else "items"


# ---------------------------------------------------------------- factory ---

def create_provider(cfg: Optional[AppConfig] = None) -> LLMProvider:
    cfg = cfg or get_config()
    backend = cfg.llm.backend
    if backend == "ollama":
        from app.llm.ollama import OllamaProvider

        return OllamaProvider(cfg)
    if backend == "openai_compat":
        from app.llm.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(cfg)
    from app.llm.llamacpp import LlamaCppProvider

    return LlamaCppProvider(cfg)
