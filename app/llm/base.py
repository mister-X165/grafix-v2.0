"""LLM abstraction layer (ТЗ §5, §41–§56).

`LLMProvider` — интерфейс; реализации: llama.cpp (основной), Ollama,
OpenAI-compatible local API (LM Studio). Облачные LLM API запрещены (§95).

`structured_generate` — цикл parse → validate (Pydantic) → repair → retry
с ограниченным числом попыток (§55). Ответы LLM никогда не принимаются на веру.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
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


# ---------------------------------------------------------------- provider ---

class CancellationRequested(Exception):
    """LLM-вызов прерван пользователем (отмена анализа)."""


def _has_running_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


async def to_thread_complete(provider: LLMProvider, messages: list[dict], *,
                             max_tokens: Optional[int] = None,
                             temperature: Optional[float] = None,
                             stop_event: Optional[threading.Event] = None) -> str:
    """Выполнить синхронный `complete` вне event loop.

    Ключевой момент стабильности: сетевые вызовы к локальной LLM могут занимать
    десятки секунд (request_timeout_s по умолчанию — 300 с). Если выполнять их
    прямо в coroutine, они сериализуют весь конвейер и блокируют обработку
    событий (UI «зависает», прогресс не обновляется, отмена не работает).
    В пуле потоков такие вызовы перекрываются и не мешают loops/таймаутам.
    """
    loop = asyncio.get_running_loop()

    def _call() -> str:
        if stop_event is not None and stop_event.is_set():
            raise CancellationRequested("Анализ остановлен пользователем.")
        return provider.complete(messages, max_tokens=max_tokens, temperature=temperature)

    if stop_event is None:
        return await loop.run_in_executor(None, _call)

    # Отменяемый вариант: ждём future с короткими чашками, чтобы среагировать
    # на cancel() в течение ~0.25 с, не дожидаясь конца долгого HTTP-запроса.
    fut = loop.run_in_executor(None, _call)
    while True:
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=0.25)
        except asyncio.TimeoutError:
            if stop_event.is_set():
                fut.cancel()
                raise CancellationRequested("Анализ остановлен пользователем.") from None
        except CancellationRequested:
            raise
        except Exception:
            raise


async def to_thread_health_check(provider: LLMProvider,
                                 stop_event: Optional[threading.Event] = None
                                 ) -> tuple[bool, str]:
    """health_check тоже сетевой (может ждать таймаут) — выносим в executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, provider.health_check)


async def _generic_structured_async(provider, messages: list[dict],
                                    schema: "Type[TModel]", *,
                                    retries: Optional[int] = None,
                                    max_tokens: Optional[int] = None):
    """Async-путь structured_generate для duck-typed провайдеров.

    Провайдеры без собственного async-реализации (тестовые заглушки,
    сторонние адаптеры) исполняются здесь: их синхронный `complete()`
    выносится в executor через to_thread_complete, поэтому event loop
    не блокируется, вызовы перекрываются и отмена работает по stop_event.
    """
    cfg = get_config()
    attempts = (cfg.llm.max_retries if retries is None else retries) + 1
    last_raw = ""
    last_err = ""
    msgs = list(messages)
    stop = getattr(provider, "stop_event", None)
    for attempt in range(attempts):
        if stop is not None and stop.is_set():
            raise CancellationRequested("Анализ остановлен пользователем.")
        try:
            raw = await to_thread_complete(provider, msgs,
                                           max_tokens=max_tokens,
                                           temperature=0.0,
                                           stop_event=stop)
        except CancellationRequested:
            raise
        except LLMUnavailableError:
            raise
        except Exception as e:  # сеть/таймаут — retry
            last_err = f"Ошибка генерации: {e}"
            log.warning("generic llm attempt %d failed: %s", attempt + 1, e)
            continue
        last_raw = raw
        try:
            data = extract_json(raw)
            if isinstance(data, list):
                data = {list_key_for(schema): data}
            return schema.model_validate(data)
        except (ValueError, ValidationError, KeyError) as e:
            last_err = str(e)
            log.warning("generic invalid structured output (attempt %d): %s",
                        attempt + 1, last_err[:400])
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


class LLMProvider(ABC):
    """Интерфейс локальной LLM."""

    name: str = "base"
    #: поток отмены анализа (устанавливается pipeline из UI-воркера)
    stop_event: Optional[threading.Event] = None

    @abstractmethod
    def health_check(self) -> tuple[bool, str]:
        """(ready, сообщение). Проверяется при старте и перед анализом."""

    @abstractmethod
    def complete(self, messages: list[dict], *, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        """Один completion-вызов. Бросает LLMUnavailableError при проблемах."""

    @abstractmethod
    def model_info(self) -> ModelInfo: ...

    def check_cancelled(self) -> None:
        """Бросает CancellationRequested, если пользователь нажал «Отменить»."""
        if self.stop_event is not None and self.stop_event.is_set():
            raise CancellationRequested("Анализ остановлен пользователем.")

    # ---------------------------------------- structured generation --------
    async def structured_generate_async(self, messages: list[dict], schema: Type[TModel],
                                        *, retries: Optional[int] = None,
                                        max_tokens: Optional[int] = None) -> TModel:
        """Асинхронная версия parse → validate → repair → retry (§55, §56).

        Вызовы модели выполняются вне event loop (см. to_thread_complete),
        поэтому несколько claim'ов обрабатываются параллельно и таймауты
        конвейера продолжают работать корректно.

        Реализация универсальна (generic-путь): провайдер может быть как
        полноценным LLMProvider, так и duck-typed объектом с синхронными
        complete()/health_check() — блокирующий вызов выносится в executor,
        event loop не зависает, вызовы перекрываются, отмена работает.
        """
        return await _generic_structured_async(self, messages, schema,
                                               retries=retries,
                                               max_tokens=max_tokens)

    def _async_raw(self, messages: list[dict], *,
                   max_tokens: Optional[int] = None) -> "Any":
        """Точка расширения: собственный async-транспорт провайдера.

        По умолчанию возвращает coroutine с синхронным complete() в executor;
        провайдеры с нативным async-клиентом могут переопределить этот метод.
        """
        return to_thread_complete(self, messages, max_tokens=max_tokens,
                                  temperature=0.0, stop_event=self.stop_event)

    def structured_generate(self, messages: list[dict], schema: Type[TModel],
                            *, retries: Optional[int] = None,
                            max_tokens: Optional[int] = None) -> TModel:
        """parse → validate → repair → retry (§55, §56).

        При провале валидации отправляем модели ошибку и просим исправить JSON.
        Синхронная версия — для CLI и тестов; внутри event loop автоматически
        используется асинхронный путь через executor (см. structured_generate_async).
        """
        cfg = get_config()
        attempts = (cfg.llm.max_retries if retries is None else retries) + 1
        last_raw = ""
        last_err = ""
        msgs = list(messages)
        for attempt in range(attempts):
            self.check_cancelled()
            try:
                if _has_running_loop():
                    # мы внутри работающего event loop: сам блокирующий вызов
                    # недопустим — но синхронный API здесь оставлен только как
                    # совместимость; реальный pipeline использует async-вариант.
                    raise RuntimeError(
                        "structured_generate() вызван внутри event loop; "
                        "используйте structured_generate_async()")
                raw = self.complete(msgs, max_tokens=max_tokens, temperature=0.0)
            except CancellationRequested:
                raise
            except RuntimeError:
                raise
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
