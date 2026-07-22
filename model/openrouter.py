"""OpenRouter client for Grafix — DeepSeek 3.2 and other remote models."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from model.lmstudio import (
    DEFAULT_MAX_TOKENS,
    DEEPSEEK_SYSTEM_PROMPT,
    ENTITY_COMMENT_SYSTEM,
    ENTITY_COMMENT_SYSTEM_DOC_ONLY,
    ENTITY_COMMENTS_BATCH_SYSTEM,
    QA_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT_DOC_ONLY,
    SYSTEM_PROMPT,
    _clip_text,
    _excerpt_around_entity,
    _message_content,
    parse_comments_flexible,
    parse_extract_bundle,
    parse_triples_flexible,
)
from model.triples import Triple

DEFAULT_BASE = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v3.2")
DEFAULT_MODEL_V4 = os.environ.get("OPENROUTER_MODEL_V4", "deepseek/deepseek-v4-pro")
DEFAULT_TIMEOUT = float(os.environ.get("OPENROUTER_TIMEOUT", "300"))
# Reasoning eats into max_tokens — keep headroom for the final answer
DEFAULT_MAX_TOKENS_OR = int(os.environ.get("OPENROUTER_MAX_TOKENS", "4000"))
# Extract + thinking: high effort can burn most of the budget on reasoning alone
DEFAULT_EXTRACT_MAX_TOKENS = int(
    os.environ.get("OPENROUTER_EXTRACT_MAX_TOKENS", "32000")
)
DEFAULT_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
APP_TITLE = os.environ.get("OPENROUTER_APP_TITLE", "Grafix")
APP_URL = os.environ.get("OPENROUTER_APP_URL", "https://github.com/komandantemerk/grafix")

# DeepSeek 3.2 / V4 Pro: thinking mode via OpenRouter `reasoning`
_REASONING_OFF = {"0", "false", "no", "off"}
REASONING_ENABLED = (
    os.environ.get("OPENROUTER_REASONING", "1").strip().lower() not in _REASONING_OFF
)
REASONING_EFFORT = (
    os.environ.get("OPENROUTER_REASONING_EFFORT", "high").strip().lower() or "high"
)
# Extract needs room for a large JSON; medium leaves more budget for the answer
EXTRACT_REASONING_EFFORT = (
    os.environ.get("OPENROUTER_EXTRACT_REASONING_EFFORT", "medium").strip().lower()
    or "medium"
)
# UI engine id → OpenRouter model slug
OPENROUTER_MODELS = {
    "deepseek": DEFAULT_MODEL,
    "deepseek-v4": DEFAULT_MODEL_V4,
}
DEEPSEEK_ENGINES = frozenset(OPENROUTER_MODELS)


def is_deepseek_engine(engine: str | None) -> bool:
    return (engine or "").strip().lower() in DEEPSEEK_ENGINES


def model_for_engine(engine: str | None) -> str:
    e = (engine or "deepseek").strip().lower()
    return OPENROUTER_MODELS.get(e, DEFAULT_MODEL)


def openrouter_for_engine(engine: str | None = "deepseek") -> "OpenRouterExtractor":
    return OpenRouterExtractor(model=model_for_engine(engine))


class OpenRouterExtractor:
    """OpenAI-compatible chat via OpenRouter (DeepSeek family by default)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ):
        self.api_key = (api_key if api_key is not None else DEFAULT_API_KEY).strip()
        self.base_url = (base_url or DEFAULT_BASE).rstrip("/")
        self.model = model if model is not None else DEFAULT_MODEL
        self.timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
        self._available: bool | None = None

    @property
    def ready(self) -> bool:
        if self._available is None:
            self._available = self.ping()
        return bool(self._available and self.api_key and self.model)

    @property
    def resolved_model(self) -> str | None:
        return self.model if self.api_key else None

    def refresh(self) -> None:
        self.api_key = os.environ.get("OPENROUTER_API_KEY", self.api_key).strip()
        self._available = None

    def ping(self) -> bool:
        if not self.api_key:
            return False
        try:
            self._get_json("/models")
            return True
        except Exception:
            # Key may still work for chat even if /models is restricted
            return bool(self.api_key)

    def list_models(self) -> list[str]:
        if not self.api_key:
            return []
        try:
            data = self._get_json("/models")
        except Exception:
            return [self.model] if self.model else []
        items = data.get("data") or []
        return [m.get("id") for m in items if m.get("id")]

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int | None = None,
        reasoning: bool | None = None,
        reasoning_effort: str | None = None,
        reasoning_exclude: bool | None = None,
    ) -> dict[str, Any]:
        self.refresh()
        if not self.api_key:
            return {
                "ok": False,
                "raw": "",
                "error": "Нет OPENROUTER_API_KEY — задай ключ в окружении",
                "model": None,
            }
        model = self.model
        if not model:
            return {"ok": False, "raw": "", "error": "Не задана модель OpenRouter", "model": None}

        payload: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        else:
            payload["max_tokens"] = DEFAULT_MAX_TOKENS_OR

        use_reasoning = REASONING_ENABLED if reasoning is None else bool(reasoning)
        effort_used = None
        if use_reasoning:
            effort_used = (reasoning_effort or REASONING_EFFORT).strip().lower() or "high"
            reasoning_cfg: dict[str, Any] = {
                "enabled": True,
                "effort": effort_used,
            }
            if reasoning_exclude is True:
                reasoning_cfg["exclude"] = True
            payload["reasoning"] = reasoning_cfg
        else:
            # Explicitly disable thinking (V3.2 / V4 otherwise may still think)
            payload["reasoning"] = {"enabled": False, "effort": "none"}

        try:
            data = self._post_json("/chat/completions", payload)
        except Exception as e:
            self._available = False
            err = str(e)
            if "timed out" in err.lower() or "timeout" in err.lower():
                err = (
                    f"Таймаут OpenRouter ({int(self.timeout)} с). "
                    "Увеличь OPENROUTER_TIMEOUT или сократи текст."
                )
            return {"ok": False, "raw": "", "error": err, "model": model}

        self._available = True
        content = _message_content(data)
        reasoning_text = _message_reasoning(data)
        if not (content or "").strip() and (reasoning_text or "").strip():
            maybe = reasoning_text.strip()
            if "{" in maybe and "triples" in maybe:
                content = maybe
        finish = None
        try:
            finish = data["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError, AttributeError):
            finish = None
        err = None
        if finish == "length":
            err = (
                "Ответ обрезан по max_tokens — увеличь OPENROUTER_EXTRACT_MAX_TOKENS "
                f"(сейчас запрос: {payload.get('max_tokens')}) или сократи текст / "
                "выключи Reasoning / понизь OPENROUTER_EXTRACT_REASONING_EFFORT. "
                "Парсер попробует спасти уже готовые тройки."
            )
        return {
            "ok": True,
            "raw": content,
            "reasoning": reasoning_text if use_reasoning else "",
            "error": err,
            "model": model,
            "response": data,
            "finish_reason": finish,
            "empty": not bool((content or "").strip()),
            "reasoning_enabled": use_reasoning,
            "reasoning_effort": effort_used,
            "max_tokens": payload.get("max_tokens"),
        }

    def extract(self, text: str, few_shot: list[dict] | None = None, *, reasoning: bool | None = None) -> list[Triple]:
        return self.extract_debug(text, few_shot=few_shot, reasoning=reasoning)["triples"]

    def extract_debug(
        self,
        text: str,
        few_shot: list[dict] | None = None,
        *,
        reasoning: bool | None = None,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = [{"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT}]
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
                    "Извлеки граф: entities + triples (explicit/hidden/false с evidence) "
                    "+ contradictions. Ответ — только полный JSON вида "
                    '{"entities":[...],"triples":[...],"contradictions":[...]}. '
                    "Обязательно закрой массивы и объект. Рассуждения в ответ не пиши.\n\n"
                    f"Текст:\n{text.strip()}"
                ),
            }
        )
        chat = self.chat(
            messages,
            temperature=0.05,
            max_tokens=DEFAULT_EXTRACT_MAX_TOKENS,
            reasoning=reasoning,
            reasoning_effort=EXTRACT_REASONING_EFFORT if reasoning is not False else None,
        )
        raw = chat.get("raw") or ""
        bundle = parse_extract_bundle(raw) if chat.get("ok") else {
            "triples": [],
            "entities": [],
            "contradictions": [],
        }
        triples = bundle["triples"]
        err = chat.get("error")
        if chat.get("ok") and not triples and raw.strip():
            err = (err + "; " if err else "") + "JSON не разобран — смотри сырой ответ"
        return {
            "triples": triples,
            "entities": bundle.get("entities") or [],
            "contradictions": bundle.get("contradictions") or [],
            "raw_response": raw,
            "parsed_json": [t.as_dict() for t in triples],
            "error": err,
            "model": chat.get("model"),
            "ok": bool(chat.get("ok")),
            "finish_reason": chat.get("finish_reason"),
            "prompt_user": messages[-1]["content"],
            "reasoning": chat.get("reasoning") or "",
            "reasoning_effort": chat.get("reasoning_effort"),
            "reasoning_enabled": chat.get("reasoning_enabled"),
        }

    def batch_entity_comments(
        self,
        text: str,
        triples: list[Triple] | list[dict[str, Any]],
        max_entities: int = 35,
        *,
        reasoning: bool | None = None,
    ) -> dict[str, Any]:
        """After graph extract: one comment per unique entity (strict coverage)."""
        entities: list[str] = []
        seen: set[str] = set()
        triple_lines: list[str] = []
        for item in triples or []:
            if isinstance(item, Triple):
                s, r, o = item.subject, item.relation, item.object
            else:
                s = str((item or {}).get("subject") or "").strip()
                r = str((item or {}).get("relation") or "").strip()
                o = str((item or {}).get("object") or "").strip()
            if s and o:
                kind = "explicit"
                if isinstance(item, Triple):
                    kind = getattr(item, "kind", "explicit") or "explicit"
                else:
                    kind = str((item or {}).get("kind") or "explicit")
                tag = {"hidden": "скрытая", "false": "ложная"}.get(kind, "явная")
                triple_lines.append(f"{s} —[{r}|{tag}]→ {o}")
            for name in (s, o):
                if name and name not in seen:
                    seen.add(name)
                    entities.append(name)
        entities = entities[: max(1, max_entities)]
        if not entities:
            return {
                "comments": {},
                "ok": True,
                "error": None,
                "raw_response": "",
                "model": self.model,
                "prompt_user": "",
                "missing": [],
            }

        doc = _clip_text((text or "").strip(), 5000)
        entity_list = "\n".join(f"- {e}" for e in entities)
        rel_block = "\n".join(triple_lines[:80]) if triple_lines else "(нет)"
        user = (
            "Граф уже построен. Теперь ОБЯЗАТЕЛЬНО напиши комментарий для каждой сущности.\n\n"
            f"Сущности (все должны быть в comments):\n{entity_list}\n\n"
            f"Тройки графа:\n{rel_block}\n\n"
            f"Текст документа:\n{doc or '(пусто)'}\n\n"
            'Ответ — только JSON: {"comments":{"Имя":"комментарий",...}}'
        )
        messages = [
            {"role": "system", "content": ENTITY_COMMENTS_BATCH_SYSTEM},
            {"role": "user", "content": user},
        ]
        chat = self.chat(
            messages,
            temperature=0.25,
            max_tokens=DEFAULT_EXTRACT_MAX_TOKENS,
            reasoning=reasoning,
        )
        raw = chat.get("raw") or ""
        parsed = parse_comments_flexible(raw) if chat.get("ok") else {}

        # Map to exact entity names (case-insensitive soft match)
        lower_map = {e.lower(): e for e in entities}
        comments: dict[str, str] = {}
        for key, val in parsed.items():
            canon = lower_map.get(key.lower())
            if canon:
                comments[canon] = val
            elif key in entities:
                comments[key] = val

        missing = [e for e in entities if e not in comments or not comments[e].strip()]
        err = chat.get("error")
        if chat.get("ok") and missing:
            # Second pass only for missing entities
            miss_list = "\n".join(f"- {e}" for e in missing)
            user2 = (
                "Допиши комментарии ТОЛЬКО для сущностей без комментария.\n\n"
                f"Сущности:\n{miss_list}\n\n"
                f"Тройки графа:\n{rel_block}\n\n"
                f"Текст документа:\n{doc or '(пусто)'}\n\n"
                'Ответ — только JSON: {"comments":{"Имя":"комментарий",...}}'
            )
            chat2 = self.chat(
                [
                    {"role": "system", "content": ENTITY_COMMENTS_BATCH_SYSTEM},
                    {"role": "user", "content": user2},
                ],
                temperature=0.25,
                max_tokens=DEFAULT_EXTRACT_MAX_TOKENS,
                reasoning=reasoning,
            )
            if chat2.get("ok"):
                raw2 = chat2.get("raw") or ""
                parsed2 = parse_comments_flexible(raw2)
                for key, val in parsed2.items():
                    canon = lower_map.get(key.lower()) or (key if key in entities else None)
                    if canon and val.strip() and canon not in comments:
                        comments[canon] = val.strip()
                raw = (raw + "\n\n---\n\n" + raw2).strip()
                if chat2.get("error"):
                    err = (err + "; " if err else "") + str(chat2["error"])
            missing = [e for e in entities if e not in comments or not comments[e].strip()]

        if missing:
            err = (
                (err + "; " if err else "")
                + f"нет комментариев для {len(missing)} сущностей: "
                + ", ".join(missing[:8])
                + ("…" if len(missing) > 8 else "")
            )

        return {
            "comments": comments,
            "ok": bool(chat.get("ok") and comments),
            "error": err,
            "raw_response": raw,
            "model": chat.get("model"),
            "prompt_user": user,
            "missing": missing,
            "entities": entities,
        }

    def answer_about_text(
        self,
        text: str,
        question: str,
        web_results: list[dict[str, Any]] | None = None,
        *,
        reasoning: bool | None = None,
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
        chat = self.chat(
            messages,
            temperature=0.2,
            max_tokens=DEFAULT_MAX_TOKENS_OR,
            reasoning=reasoning,
        )
        return {
            "answer": (chat.get("raw") or "").strip() or (chat.get("error") or "Нет ответа"),
            "raw_response": chat.get("raw") or "",
            "error": chat.get("error"),
            "model": chat.get("model"),
            "ok": bool(chat.get("ok")),
            "source": "openrouter_text_qa",
            "prompt_user": messages[-1]["content"],
        }

    def generate_entity_comment(
        self,
        text: str,
        entity: str,
        relations: list[dict[str, Any]],
        web_results: list[dict[str, Any]] | None = None,
        *,
        reasoning: bool | None = None,
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
        chat = self.chat(
            messages,
            temperature=0.35,
            max_tokens=DEFAULT_MAX_TOKENS_OR,
            reasoning=reasoning,
        )
        raw = (chat.get("raw") or "").strip()
        comment = raw
        for prefix in ("Комментарий:", "Comment:", "Ответ:"):
            if comment.lower().startswith(prefix.lower()):
                comment = comment[len(prefix) :].strip()
        if chat.get("ok") and not comment:
            return {
                "comment": "",
                "raw_response": raw,
                "error": "DeepSeek вернул пустой комментарий",
                "model": chat.get("model"),
                "ok": False,
                "prompt_user": messages[-1]["content"],
            }
        return {
            "comment": comment or (chat.get("error") or ""),
            "raw_response": raw,
            "error": chat.get("error"),
            "model": chat.get("model"),
            "ok": bool(chat.get("ok") and comment),
            "prompt_user": messages[-1]["content"],
        }

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "HTTP-Referer": APP_URL,
            "X-Title": APP_TITLE,
        }

    def _get_json(self, path: str) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            headers=self._headers(),
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=min(self.timeout, 20)) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post_json(self, path: str, payload: dict) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {e.code}: {detail}") from e


def _message_reasoning(data: dict[str, Any]) -> str:
    """Extract thinking / reasoning text from OpenRouter response (if present)."""
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError):
        return ""
    msg = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(msg, dict):
        return ""

    def _as_text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts: list[str] = []
            for p in value:
                if isinstance(p, str) and p.strip():
                    parts.append(p.strip())
                elif isinstance(p, dict):
                    t = p.get("text") or p.get("content") or p.get("summary") or ""
                    if isinstance(t, str) and t.strip():
                        parts.append(t.strip())
            return "\n".join(parts).strip()
        return ""

    for key in ("reasoning", "reasoning_content"):
        text = _as_text(msg.get(key))
        if text:
            return text

    details = msg.get("reasoning_details")
    if isinstance(details, list) and details:
        chunks: list[str] = []
        for item in details:
            if isinstance(item, dict):
                t = item.get("text") or item.get("content") or item.get("summary") or ""
                if isinstance(t, str) and t.strip():
                    chunks.append(t.strip())
            elif isinstance(item, str) and item.strip():
                chunks.append(item.strip())
        if chunks:
            return "\n".join(chunks)
    return ""
