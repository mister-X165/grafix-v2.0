"""LM Studio client (OpenAI-compatible) for Grafix triple extraction + text QA."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from model.triples import Triple

DEFAULT_BASE = os.environ.get("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
DEFAULT_MODEL = os.environ.get("LM_STUDIO_MODEL", "")  # empty = auto first loaded
DEFAULT_TIMEOUT = float(os.environ.get("LM_STUDIO_TIMEOUT", "300"))
DEFAULT_MAX_TOKENS = int(os.environ.get("LM_STUDIO_MAX_TOKENS", "500"))

SYSTEM_PROMPT = """Ты — модуль извлечения графа знаний для программы Grafix.

ЗАДАЧА:
Прочитай пользовательский текст и извлеки все явные факты в виде троек (субъект, связь, объект).

ФОРМАТ ОТВЕТА (строго):
Верни один JSON-объект и ничего больше. Без markdown, без ```, без пояснений до/после JSON.
{"triples":[{"subject":"Имя","relation":"тип_связи","object":"Имя_или_значение"}]}

ПРАВИЛА:
1. Только факты, которые прямо следуют из текста. Не додумывай.
2. subject и object — канонические имена (именительный падеж), без лишних слов.
3. relation — короткий snake_case на русском или латинице: работает_в, основал, знает, отец, находится_в, связан_с.
4. Если фактов нет: {"triples":[]}
5. Не пиши анализ текста прозой — только JSON.
"""

ENTITY_COMMENT_SYSTEM = """Ты пишешь краткий комментарий об сущности графа знаний для программы Grafix.
Опирайся на текст документа, список связей сущности и (если есть) результаты веб-поиска.
Стиль: 2–4 предложения, по делу, на русском. Без списков и без заголовков.
Пиши СРАЗУ готовый текст комментария — только его, без префиксов вроде «Комментарий:».
Не выдумывай факты сверх документа, связей и сниппетов. Если используешь веб — кратко укажи источник.
"""

ENTITY_COMMENT_SYSTEM_DOC_ONLY = """Ты пишешь краткий комментарий об сущности графа знаний для программы Grafix.
Опирайся ТОЛЬКО на текст документа и список связей сущности.
Стиль: 2–4 предложения, по делу, на русском. Без списков, без заголовков.
Пиши СРАЗУ готовый текст комментария — только его, без префиксов вроде «Комментарий:».
Не выдумывай факты.
"""


QA_SYSTEM_PROMPT = """Ты — помощник Grafix. Отвечай на вопрос по приведённому тексту документа.
Если даны результаты веб-поиска — можешь опираться и на них; кратко ссылайся на источники.
Отвечай кратко на русском. Не выдумывай факты сверх документа и сниппетов.
Если ответа нет ни в тексте, ни в поиске — так и скажи.
"""

QA_SYSTEM_PROMPT_DOC_ONLY = """Ты — помощник Grafix. Отвечай на вопрос ТОЛЬКО по приведённому тексту документа.
Отвечай кратко на русском. Если в тексте нет ответа — так и скажи.
Не выдумывай факты вне текста.
"""


class LMStudioExtractor:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ):
        self.base_url = (base_url or DEFAULT_BASE).rstrip("/")
        self.model = model if model is not None else DEFAULT_MODEL
        self.timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
        self._resolved_model: str | None = None
        self._available: bool | None = None

    @property
    def ready(self) -> bool:
        if self._available is None:
            self._available = self.ping()
        return bool(self._available and self.resolved_model)

    @property
    def resolved_model(self) -> str | None:
        if self._resolved_model:
            return self._resolved_model
        if self.model:
            self._resolved_model = self.model
            return self._resolved_model
        models = self.list_models()
        if models:
            self._resolved_model = models[0]
        return self._resolved_model

    def ping(self) -> bool:
        try:
            self._get_json("/models")
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            data = self._get_json("/models")
        except Exception:
            return []
        items = data.get("data") or []
        return [m.get("id") for m in items if m.get("id")]

    def refresh(self) -> None:
        self._available = None
        self._resolved_model = None

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Low-level chat; returns {ok, raw, error, model}."""
        self.refresh()
        if not self.ping():
            self._available = False
            return {"ok": False, "raw": "", "error": "LM Studio недоступен", "model": None}
        self._available = True
        model = self.resolved_model
        if not model:
            return {"ok": False, "raw": "", "error": "Нет загруженной модели", "model": None}

        payload: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        try:
            data = self._post_json("/chat/completions", payload)
        except Exception as e:
            self._available = False
            err = str(e)
            if "timed out" in err.lower() or "timeout" in err.lower():
                err = (
                    f"Таймаут LM Studio ({int(self.timeout)} с). "
                    "Модель не успела ответить — укороти текст, выключи веб-поиск "
                    "или увеличь LM_STUDIO_TIMEOUT."
                )
            return {"ok": False, "raw": "", "error": err, "model": model}

        content = _message_content(data)
        if not (content or "").strip():
            # Still "ok" HTTP-wise, but empty body — surface as soft failure upstream
            return {
                "ok": True,
                "raw": "",
                "error": None,
                "model": model,
                "response": data,
                "empty": True,
            }
        return {"ok": True, "raw": content, "error": None, "model": model, "response": data}

    def extract(self, text: str, few_shot: list[dict] | None = None) -> list[Triple]:
        result = self.extract_debug(text, few_shot=few_shot)
        return result["triples"]

    def extract_debug(self, text: str, few_shot: list[dict] | None = None) -> dict[str, Any]:
        """Extract triples and return debug payload for UI."""
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex in few_shot or []:
            ex_text = (ex.get("text") or "").strip()
            triples = ex.get("triples") or []
            if not ex_text or not triples:
                continue
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Извлеки тройки из текста ниже. Ответ — только JSON.\n\n"
                        f"Текст:\n{ex_text}"
                    ),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps({"triples": triples}, ensure_ascii=False),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": (
                    "Извлеки тройки из текста ниже. Ответ — только JSON вида "
                    '{"triples":[{"subject":"...","relation":"...","object":"..."}]}.\n\n'
                    f"Текст:\n{text.strip()}"
                ),
            }
        )

        chat = self.chat(messages, temperature=0.05, max_tokens=DEFAULT_MAX_TOKENS)
        raw = chat.get("raw") or ""
        triples = parse_triples_flexible(raw) if chat.get("ok") else []
        return {
            "triples": triples,
            "raw_response": raw,
            "parsed_json": [
                {"subject": t.subject, "relation": t.relation, "object": t.object} for t in triples
            ],
            "error": chat.get("error"),
            "model": chat.get("model"),
            "ok": bool(chat.get("ok")),
            "prompt_user": messages[-1]["content"],
        }

    def answer_about_text(
        self,
        text: str,
        question: str,
        web_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        from graph.websearch import format_results_for_prompt

        use_web = bool(web_results)
        system = QA_SYSTEM_PROMPT if use_web else QA_SYSTEM_PROMPT_DOC_ONLY
        doc = _clip_text((text or "").strip(), 6000)
        parts = [f"Текст документа:\n{doc or '(пусто)'}"]
        if use_web:
            parts.append(
                "Результаты веб-поиска:\n"
                + format_results_for_prompt((web_results or [])[:4])
            )
        parts.append(f"Вопрос:\n{(question or '').strip()}\n\nОтвет:")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        chat = self.chat(messages, temperature=0.2, max_tokens=DEFAULT_MAX_TOKENS)
        return {
            "answer": (chat.get("raw") or "").strip() or (chat.get("error") or "Нет ответа"),
            "raw_response": chat.get("raw") or "",
            "error": chat.get("error"),
            "model": chat.get("model"),
            "ok": bool(chat.get("ok")),
            "source": "lmstudio_text_qa",
            "prompt_user": messages[-1]["content"],
        }

    def generate_entity_comment(
        self,
        text: str,
        entity: str,
        relations: list[dict[str, Any]],
        web_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        from graph.websearch import format_results_for_prompt

        lines: list[str] = []
        for r in relations:
            direction = r.get("direction")
            rel = r.get("relation") or "?"
            neighbor = r.get("neighbor") or "?"
            if direction == "in":
                lines.append(f"{neighbor} —[{rel}]→ {entity}")
            else:
                lines.append(f"{entity} —[{rel}]→ {neighbor}")
        rel_block = "\n".join(lines) if lines else "(связей нет)"
        use_web = bool(web_results)
        system = ENTITY_COMMENT_SYSTEM if use_web else ENTITY_COMMENT_SYSTEM_DOC_ONLY
        doc = _excerpt_around_entity(text or "", entity, max_chars=2800)
        parts = [
            f"Сущность: {entity}",
            f"Связи:\n{rel_block}",
            f"Фрагмент документа:\n{doc or '(пусто)'}",
        ]
        if use_web:
            parts.append(
                "Результаты веб-поиска:\n"
                + format_results_for_prompt((web_results or [])[:3])
            )
        parts.append("Комментарий:")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        chat = self.chat(messages, temperature=0.35, max_tokens=DEFAULT_MAX_TOKENS)
        raw = (chat.get("raw") or "").strip()
        # Strip common wrappers / labels if model echoes the prompt cue
        comment = raw
        for prefix in ("Комментарий:", "Comment:", "Ответ:"):
            if comment.lower().startswith(prefix.lower()):
                comment = comment[len(prefix) :].strip()
        if chat.get("ok") and not comment:
            # Keep raw dump in error for Debug drawer
            return {
                "comment": "",
                "raw_response": raw or json.dumps(chat.get("response") or {}, ensure_ascii=False)[:2000],
                "error": (
                    "Модель ответила, но текст комментария пуст "
                    "(часто уходит в reasoning). Открой Debug-лог."
                ),
                "model": chat.get("model"),
                "ok": False,
                "prompt_user": messages[-1]["content"],
            }
        return {
            "comment": comment or (chat.get("error") or ""),
            "raw_response": raw or (chat.get("raw") or ""),
            "error": chat.get("error"),
            "model": chat.get("model"),
            "ok": bool(chat.get("ok") and comment),
            "prompt_user": messages[-1]["content"],
        }

    def _get_json(self, path: str) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            headers={"Accept": "application/json"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=min(self.timeout, 10)) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post_json(self, path: str, payload: dict) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LM Studio HTTP {e.code}: {detail}") from e


def _clip_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _excerpt_around_entity(text: str, entity: str, max_chars: int = 2800) -> str:
    """Prefer a window around the entity mention; fall back to head clip."""
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    key = (entity or "").strip().lower()
    if not key:
        return _clip_text(text, max_chars)
    lower = text.lower()
    idx = lower.find(key)
    if idx < 0:
        return _clip_text(text, max_chars)
    start = max(0, idx - max_chars // 3)
    end = min(len(text), start + max_chars)
    chunk = text[start:end].strip()
    if start > 0:
        chunk = "…" + chunk
    if end < len(text):
        chunk = chunk + "…"
    return chunk


def _message_content(data: dict) -> str:
    """Pull assistant text from varied LM Studio / OpenAI-compatible shapes."""
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError):
        return ""

    msg = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(msg, dict):
        msg = {}

    def _as_text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts: list[str] = []
            for p in value:
                if isinstance(p, str) and p.strip():
                    parts.append(p.strip())
                elif isinstance(p, dict):
                    t = p.get("text") or p.get("content") or ""
                    if isinstance(t, str) and t.strip():
                        parts.append(t.strip())
            return "\n".join(parts).strip()
        return ""

    # Prefer visible answer content; fall back to reasoning (Gemma / thinking models)
    for key in ("content", "text", "reasoning_content", "reasoning"):
        text = _as_text(msg.get(key))
        if text:
            return text

    # Legacy completions-style
    text = _as_text(choice.get("text") if isinstance(choice, dict) else None)
    if text:
        return text

    return ""


def parse_triples_flexible(content: str) -> list[Triple]:
    """Parse Gemma/LM outputs in several common shapes."""
    if not content or not content.strip():
        return []

    text = content.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    if fence:
        text = fence.group(1).strip()

    # Try direct JSON / embedded JSON object or array
    candidates = _json_candidates(text)
    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            # Fix trailing commas
            try:
                data = json.loads(re.sub(r",\s*([}\]])", r"\1", cand))
            except json.JSONDecodeError:
                continue
        triples = _triples_from_data(data)
        if triples:
            return triples

    # Arrow / dash lines: A — rel → B
    lined = _parse_arrow_lines(content)
    if lined:
        return lined

    return _parse_loose_lines(content)


def _json_candidates(text: str) -> list[str]:
    out: list[str] = []
    # whole text
    out.append(text)
    # first object
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        out.append(text[start : end + 1])
    # first array
    a0 = text.find("[")
    a1 = text.rfind("]")
    if a0 >= 0 and a1 > a0:
        out.append(text[a0 : a1 + 1])
    return out


def _triples_from_data(data: Any) -> list[Triple]:
    raw_list: list = []
    if isinstance(data, dict):
        for key in ("triples", "relations", "edges", "links", "граф", "тройки", "связи"):
            if isinstance(data.get(key), list):
                raw_list = data[key]
                break
        if not raw_list and all(k in data for k in ("subject", "object")):
            raw_list = [data]
    elif isinstance(data, list):
        raw_list = data

    out: list[Triple] = []
    for item in raw_list:
        t = _triple_from_item(item)
        if t:
            out.append(t)
    return out


def _triple_from_item(item: Any) -> Triple | None:
    if isinstance(item, str):
        # "A|rel|B" or "A — rel → B"
        if "|" in item:
            parts = [p.strip() for p in item.split("|")]
            if len(parts) >= 3 and all(parts[:3]):
                return Triple(parts[0], parts[1].replace(" ", "_"), parts[2])
        m = re.match(r"(.+?)\s*[—–-]\s*(.+?)\s*[→>]\s*(.+)$", item.strip())
        if m:
            return Triple(m.group(1).strip(), m.group(2).strip().replace(" ", "_"), m.group(3).strip())
        return None

    if not isinstance(item, dict):
        return None

    s = _pick(item, ("subject", "subj", "source", "from", "head", "субъект", "источник"))
    r = _pick(item, ("relation", "rel", "predicate", "type", "label", "edge", "связь", "отношение", "предикат"))
    o = _pick(item, ("object", "obj", "target", "to", "tail", "объект", "цель"))

    # Nested {"subject":{"name":...}}
    s = _as_name(s)
    o = _as_name(o)
    r = str(r or "").strip().replace(" ", "_")
    if s and r and o:
        return Triple(s, r, o)
    return None


def _pick(item: dict, keys: tuple[str, ...]) -> Any:
    lower = {str(k).lower(): v for k, v in item.items()}
    for k in keys:
        if k in item:
            return item[k]
        if k.lower() in lower:
            return lower[k.lower()]
    return None


def _as_name(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, dict):
        for k in ("name", "label", "id", "text", "title"):
            if val.get(k):
                return str(val[k]).strip()
        return ""
    return str(val).strip()


def _parse_arrow_lines(content: str) -> list[Triple]:
    out: list[Triple] = []
    for line in content.splitlines():
        line = line.strip().strip("-•*").strip()
        m = re.match(r"(.+?)\s*[—–-]\s*(.+?)\s*[→>]\s*(.+)$", line)
        if not m:
            continue
        s, r, o = m.group(1).strip(), m.group(2).strip().replace(" ", "_"), m.group(3).strip()
        if s and r and o:
            out.append(Triple(s, r, o))
    return out


def parse_triples_json(content: str) -> list[Triple]:
    return parse_triples_flexible(content)


def _parse_loose_lines(content: str) -> list[Triple]:
    out: list[Triple] = []
    for line in content.splitlines():
        line = line.strip().strip("-•*")
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3 and all(parts[:3]):
            out.append(Triple(parts[0], parts[1].replace(" ", "_"), parts[2]))
    return out
