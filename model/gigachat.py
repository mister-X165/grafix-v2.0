"""GigaChat (Sber) client for Grafix — OAuth + OpenAI-compatible chat."""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from model.locale import language_instruction, normalize_language
from model.lmstudio import (
    DEFAULT_MAX_TOKENS,
    GIGACHAT_SYSTEM_PROMPT,
    GRAPH_BRIDGE_SYSTEM,
    ENTITY_COMMENT_SYSTEM,
    ENTITY_COMMENT_SYSTEM_DOC_ONLY,
    ENTITY_COMMENTS_BATCH_SYSTEM,
    QA_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT_DOC_ONLY,
    _clip_text,
    _excerpt_around_entity,
    _message_content,
    parse_comments_flexible,
    parse_extract_bundle,
    parse_triples_flexible,
)
from model.openrouter import parse_bridge_json
from model.triples import Triple


def _load_dotenv() -> None:
    path = Path(__file__).resolve().parents[1] / ".env"
    if not path.is_file():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if not key:
                continue
            if key not in os.environ or not str(os.environ.get(key) or "").strip():
                os.environ[key] = val
    except OSError:
        pass


_load_dotenv()

DEFAULT_BASE = os.environ.get("GIGACHAT_BASE_URL", "https://api.giga.chat/v1").rstrip("/")
DEFAULT_OAUTH = os.environ.get(
    "GIGACHAT_OAUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
).strip()
DEFAULT_SCOPE = os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip()
DEFAULT_MODEL = os.environ.get("GIGACHAT_MODEL", "GigaChat-3-Ultra").strip() or "GigaChat-3-Ultra"
DEFAULT_TIMEOUT = float(os.environ.get("GIGACHAT_TIMEOUT", "300"))
DEFAULT_EXTRACT_MAX_TOKENS = int(os.environ.get("GIGACHAT_EXTRACT_MAX_TOKENS", "16000"))
DEFAULT_MAX_TOKENS_GC = int(os.environ.get("GIGACHAT_MAX_TOKENS", "4000"))
# Reasoning (GigaChat Ultra / Max): model_options.reasoning.effort
_REASONING_OFF = {"0", "false", "no", "off", "none"}
REASONING_ENABLED = (
    os.environ.get("GIGACHAT_REASONING", "1").strip().lower() not in _REASONING_OFF
)
REASONING_EFFORT = (
    os.environ.get("GIGACHAT_REASONING_EFFORT", "medium").strip().lower() or "medium"
)
# Sber TLS often needs CA bundle or verify off for ngw; default verify on, opt-out via env
_VERIFY_OFF = {"0", "false", "no", "off"}
VERIFY_SSL = os.environ.get("GIGACHAT_VERIFY_SSL", "1").strip().lower() not in _VERIFY_OFF

GIGACHAT_ENGINES = frozenset({"gigachat"})


def is_gigachat_engine(engine: str | None) -> bool:
    return (engine or "").strip().lower() in GIGACHAT_ENGINES


def _env_credentials() -> str:
    return os.environ.get("GIGACHAT_CREDENTIALS", "").strip()


def _ssl_context() -> ssl.SSLContext:
    if VERIFY_SSL:
        return ssl.create_default_context()
    return ssl._create_unverified_context()


class GigaChatExtractor:
    """GigaChat via OAuth access token + /v1/chat/completions."""

    def __init__(
        self,
        credentials: str | None = None,
        base_url: str | None = None,
        oauth_url: str | None = None,
        scope: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ):
        self.credentials = (
            credentials.strip() if credentials is not None else _env_credentials()
        )
        self.base_url = (base_url or os.environ.get("GIGACHAT_BASE_URL", DEFAULT_BASE)).rstrip("/")
        self.oauth_url = (oauth_url or os.environ.get("GIGACHAT_OAUTH_URL", DEFAULT_OAUTH)).strip()
        self.scope = (
            scope
            or os.environ.get("GIGACHAT_SCOPE", DEFAULT_SCOPE)
        ).strip() or "GIGACHAT_API_PERS"
        self.model = (
            model
            if model is not None
            else (os.environ.get("GIGACHAT_MODEL", DEFAULT_MODEL).strip() or "GigaChat-3-Ultra")
        )
        self.timeout = timeout if timeout is not None else float(
            os.environ.get("GIGACHAT_TIMEOUT", str(DEFAULT_TIMEOUT))
        )
        self._available: bool | None = None
        self._access_token: str | None = None
        self._token_expires_at: float = 0.0

    @property
    def ready(self) -> bool:
        if self._available is None:
            self._available = self.ping()
        return bool(self._available and self.credentials and self.model)

    @property
    def resolved_model(self) -> str | None:
        return self.model if self.credentials else None

    def refresh(self) -> None:
        _load_dotenv()
        self.credentials = _env_credentials() or self.credentials
        self.scope = os.environ.get("GIGACHAT_SCOPE", self.scope).strip() or self.scope
        self.model = os.environ.get("GIGACHAT_MODEL", self.model).strip() or self.model
        self.base_url = os.environ.get("GIGACHAT_BASE_URL", self.base_url).rstrip("/")
        self.oauth_url = os.environ.get("GIGACHAT_OAUTH_URL", self.oauth_url).strip()
        self._available = None

    def ping(self) -> bool:
        if not self.credentials:
            return False
        try:
            token = self._ensure_token()
            if not token:
                return False
            self._get_json("/models")
            return True
        except Exception:
            return bool(self.credentials)

    def list_models(self) -> list[str]:
        if not self.credentials:
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
        *,
        reasoning: bool | None = None,
        reasoning_effort: str | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        self.refresh()
        if not self.credentials:
            return {
                "ok": False,
                "raw": "",
                "error": "Нет GIGACHAT_CREDENTIALS — задай Auth key в .env",
                "model": None,
            }
        model = self.model
        if not model:
            return {"ok": False, "raw": "", "error": "Не задана модель GigaChat", "model": None}

        payload: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "stream": False,
        }
        payload["max_tokens"] = (
            max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS_GC
        )

        use_reasoning = REASONING_ENABLED if reasoning is None else bool(reasoning)
        effort_used = None
        if use_reasoning:
            effort_used = (reasoning_effort or REASONING_EFFORT).strip().lower() or "medium"
            payload["model_options"] = {
                "reasoning": {
                    "effort": effort_used,
                }
            }

        try:
            data = self._post_json("/chat/completions", payload)
        except Exception as e:
            self._available = False
            err = str(e)
            if "timed out" in err.lower() or "timeout" in err.lower():
                err = (
                    f"Таймаут GigaChat ({int(self.timeout)} с). "
                    "Увеличь GIGACHAT_TIMEOUT или сократи текст."
                )
            return {
                "ok": False,
                "raw": "",
                "error": err,
                "model": model,
                "reasoning_enabled": use_reasoning,
                "reasoning_effort": effort_used,
            }

        self._available = True
        content = _message_content(data)
        reasoning_text = _gigachat_reasoning(data)
        if not (content or "").strip() and (reasoning_text or "").strip():
            maybe = reasoning_text.strip()
            if "{" in maybe and ("triples" in maybe or "entities" in maybe):
                content = maybe
        finish = None
        try:
            finish = data["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError, AttributeError):
            finish = None
        err = None
        if finish == "length":
            err = (
                "Ответ обрезан по max_tokens — увеличь GIGACHAT_EXTRACT_MAX_TOKENS "
                f"(сейчас: {payload.get('max_tokens')}) или сократи текст."
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

    def extract(self, text: str, few_shot: list[dict] | None = None, **kwargs: Any) -> list[Triple]:
        return self.extract_debug(text, few_shot=few_shot, **kwargs)["triples"]

    def extract_debug(
        self,
        text: str,
        few_shot: list[dict] | None = None,
        *,
        reasoning: bool | None = None,
        language: str | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        lang = normalize_language(language)
        lang_block = language_instruction(lang)
        system = GIGACHAT_SYSTEM_PROMPT + "\n\n" + lang_block
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        for ex in few_shot or []:
            ex_text = (ex.get("text") or "").strip()
            triples = ex.get("triples") or []
            if not ex_text or not triples:
                continue
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Извлеки граф: entities + triples (explicit/hidden/false с evidence) "
                        "+ contradictions. Ответ — только полный JSON.\n\n"
                        f"{lang_block}\n\n"
                        f"Текст:\n{ex_text}"
                    ),
                }
            )
            payload = []
            for t in triples:
                if isinstance(t, Triple):
                    payload.append(t.as_dict())
                elif isinstance(t, dict):
                    payload.append(t)
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"entities": [], "triples": payload, "contradictions": []},
                        ensure_ascii=False,
                    ),
                }
            )
        body = (text or "").strip()
        messages.append(
            {
                "role": "user",
                "content": (
                    "Извлеки граф: entities + triples.\n"
                    "ОБЯЗАТЕЛЬНО: (1) explicit, (2) hidden по транзитивности/следствиям с evidence+confidence, "
                    "(3) false при любом опровержении в тексте, (4) contradictions.\n"
                    "Нельзя вернуть только explicit, если в тексте есть цепочки A→B→C или отрицания.\n"
                    "Ответ — только полный JSON вида "
                    '{"entities":[...],"triples":[...],"contradictions":[...]}. '
                    "Закрой массивы. Рассуждения в ответ не пиши.\n\n"
                    f"{lang_block}\n\n"
                    f"Текст:\n{body}"
                ),
            }
        )
        chat = self.chat(
            messages,
            temperature=0.05,
            max_tokens=DEFAULT_EXTRACT_MAX_TOKENS,
            reasoning=reasoning,
        )
        raw = chat.get("raw") or ""
        bundle = (
            parse_extract_bundle(raw)
            if chat.get("ok") and raw.strip()
            else {"triples": [], "entities": [], "contradictions": []}
        )
        triples = bundle.get("triples") or []
        if chat.get("ok") and raw.strip() and not triples:
            triples = parse_triples_flexible(raw)
        err = chat.get("error")
        if chat.get("ok") and not triples and raw.strip():
            err = (err + "; " if err else "") + "JSON троек не разобран — смотри сырой ответ"
        n_hidden = sum(1 for t in triples if (getattr(t, "kind", None) or "explicit") == "hidden")
        n_false = sum(1 for t in triples if (getattr(t, "kind", None) or "") == "false")
        # One retry nudge if model returned only explicit on a non-trivial text
        if (
            chat.get("ok")
            and triples
            and n_hidden == 0
            and n_false == 0
            and len(body) > 200
        ):
            retry_user = (
                "В прошлом ответе почти наверняка не хватает hidden/false.\n"
                "Перепиши JSON заново: добавь ВСЕ обоснованные hidden (транзитивность, следствия) "
                "и false при опровержениях. Сохрани корректные explicit.\n"
                "Только JSON.\n\n"
                f"{lang_block}\n\n"
                f"Текст:\n{body}"
            )
            chat2 = self.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": retry_user},
                ],
                temperature=0.1,
                max_tokens=DEFAULT_EXTRACT_MAX_TOKENS,
                reasoning=reasoning,
            )
            raw2 = chat2.get("raw") or ""
            if chat2.get("ok") and raw2.strip():
                bundle2 = parse_extract_bundle(raw2)
                triples2 = bundle2.get("triples") or parse_triples_flexible(raw2)
                h2 = sum(
                    1
                    for t in triples2
                    if (getattr(t, "kind", None) or "explicit") == "hidden"
                )
                f2 = sum(1 for t in triples2 if (getattr(t, "kind", None) or "") == "false")
                if triples2 and (h2 + f2 > n_hidden + n_false or len(triples2) >= len(triples)):
                    triples = triples2
                    bundle = bundle2
                    raw = (raw + "\n\n--- retry ---\n\n" + raw2).strip()
                    if chat2.get("error"):
                        err = (err + "; " if err else "") + str(chat2["error"])
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
            "prompt_system": system,
            "language": lang,
            "reasoning": chat.get("reasoning") or "",
            "reasoning_effort": chat.get("reasoning_effort"),
            "reasoning_enabled": chat.get("reasoning_enabled"),
        }

    def bridge_texts(
        self,
        base_text: str,
        new_text: str,
        existing_triples: list[Triple] | list[dict[str, Any]] | None = None,
        *,
        reasoning: bool | None = None,
    ) -> dict[str, Any]:
        """GigaChat: extract new facts + bridge edges between base graph and new text."""
        triple_lines: list[str] = []
        for item in (existing_triples or [])[:80]:
            if isinstance(item, Triple):
                s, r, o = item.subject, item.relation, item.object
                kind = getattr(item, "kind", "explicit") or "explicit"
            else:
                s = str((item or {}).get("subject") or "").strip()
                r = str((item or {}).get("relation") or "").strip()
                o = str((item or {}).get("object") or "").strip()
                kind = str((item or {}).get("kind") or "explicit")
            if s and o:
                triple_lines.append(f"{s} —[{r}|{kind}]→ {o}")
        base = _clip_text((base_text or "").strip(), 4500)
        fresh = _clip_text((new_text or "").strip(), 4500)
        user = (
            "Склей базовый граф с новым текстом.\n\n"
            f"Тройки базового графа:\n"
            + ("\n".join(triple_lines) if triple_lines else "(пусто)")
            + "\n\n"
            f"БАЗОВЫЙ текст:\n{base or '(пусто)'}\n\n"
            f"НОВЫЙ текст:\n{fresh or '(пусто)'}\n\n"
            'Ответ — только JSON: {"new_triples":[...],"bridge_triples":[...]}'
        )
        messages = [
            {"role": "system", "content": GRAPH_BRIDGE_SYSTEM},
            {"role": "user", "content": user},
        ]
        chat = self.chat(
            messages,
            temperature=0.05,
            max_tokens=DEFAULT_EXTRACT_MAX_TOKENS,
            reasoning=reasoning,
        )
        raw = chat.get("raw") or ""
        new_triples: list[Triple] = []
        bridge_triples: list[Triple] = []
        if chat.get("ok") and raw.strip():
            parsed = parse_bridge_json(raw)
            new_triples = parsed["new_triples"]
            bridge_triples = parsed["bridge_triples"]
        err = chat.get("error")
        if chat.get("ok") and not new_triples and not bridge_triples and raw.strip():
            fallback = parse_triples_flexible(raw)
            if fallback:
                new_triples = fallback
            else:
                err = (err + "; " if err else "") + "JSON склейки не разобран"
        return {
            "new_triples": new_triples,
            "bridge_triples": bridge_triples,
            "raw_response": raw,
            "parsed_json": {
                "new_triples": [t.as_dict() for t in new_triples],
                "bridge_triples": [t.as_dict() for t in bridge_triples],
            },
            "error": err,
            "model": chat.get("model"),
            "ok": bool(chat.get("ok")),
            "finish_reason": chat.get("finish_reason"),
            "prompt_user": user,
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
        language: str | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        entities: list[str] = []
        seen: set[str] = set()
        triple_lines: list[str] = []
        for item in triples or []:
            if isinstance(item, Triple):
                s, r, o = item.subject, item.relation, item.object
                kind = getattr(item, "kind", "explicit") or "explicit"
            else:
                s = str((item or {}).get("subject") or "").strip()
                r = str((item or {}).get("relation") or "").strip()
                o = str((item or {}).get("object") or "").strip()
                kind = str((item or {}).get("kind") or "explicit")
            if s and o:
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
                "entities": [],
            }

        lang_block = language_instruction(language)
        doc = _clip_text((text or "").strip(), 5000)
        entity_list = "\n".join(f"- {e}" for e in entities)
        rel_block = "\n".join(triple_lines[:80]) if triple_lines else "(нет)"
        user = (
            "Граф уже построен. Теперь ОБЯЗАТЕЛЬНО напиши комментарий для каждой сущности.\n\n"
            f"{lang_block}\n\n"
            f"Сущности (все должны быть в comments):\n{entity_list}\n\n"
            f"Тройки графа:\n{rel_block}\n\n"
            f"Текст документа:\n{doc or '(пусто)'}\n\n"
            'Ответ — только JSON: {"comments":{"Имя":"комментарий",...}}'
        )
        chat = self.chat(
            [
                {"role": "system", "content": ENTITY_COMMENTS_BATCH_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.25,
            max_tokens=DEFAULT_EXTRACT_MAX_TOKENS,
        )
        raw = chat.get("raw") or ""
        parsed = parse_comments_flexible(raw) if chat.get("ok") else {}
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
        **_kwargs: Any,
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
            max_tokens=DEFAULT_MAX_TOKENS_GC,
            reasoning=reasoning,
        )
        return {
            "answer": (chat.get("raw") or "").strip() or (chat.get("error") or "Нет ответа"),
            "raw_response": chat.get("raw") or "",
            "error": chat.get("error"),
            "model": chat.get("model"),
            "ok": bool(chat.get("ok")),
            "source": "gigachat_text_qa",
            "prompt_user": messages[-1]["content"],
        }

    def generate_entity_comment(
        self,
        text: str,
        entity: str,
        relations: list[dict[str, Any]],
        web_results: list[dict[str, Any]] | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        from graph.websearch import format_results_for_prompt

        use_web = bool(web_results)
        system = ENTITY_COMMENT_SYSTEM if use_web else ENTITY_COMMENT_SYSTEM_DOC_ONLY
        name = (entity or "").strip()
        rel_lines = []
        for r in relations or []:
            direction = r.get("direction") or ""
            rel = r.get("relation") or ""
            neighbor = r.get("neighbor") or ""
            if neighbor:
                arrow = "→" if direction == "out" else "←"
                rel_lines.append(f"{arrow} [{rel}] {neighbor}")
        excerpt = _excerpt_around_entity(text or "", name, max_chars=2800)
        parts = [
            f"Сущность: {name}",
            "Связи:\n" + ("\n".join(rel_lines[:40]) if rel_lines else "(нет)"),
            f"Фрагмент документа:\n{excerpt or '(пусто)'}",
        ]
        if use_web:
            parts.append(
                "Веб-поиск:\n" + format_results_for_prompt((web_results or [])[:3])
            )
        parts.append("Напиши комментарий:")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        chat = self.chat(messages, temperature=0.25, max_tokens=DEFAULT_MAX_TOKENS)
        comment = (chat.get("raw") or "").strip()
        return {
            "comment": comment,
            "ok": bool(chat.get("ok") and comment),
            "error": chat.get("error"),
            "raw_response": chat.get("raw") or "",
            "model": chat.get("model"),
            "prompt_user": messages[-1]["content"],
        }

    def _ensure_token(self) -> str:
        now = time.time()
        if self._access_token and now < self._token_expires_at - 60:
            return self._access_token
        token, expires_in = self._fetch_access_token()
        self._access_token = token
        self._token_expires_at = now + max(60, int(expires_in or 1800))
        return token

    def _fetch_access_token(self) -> tuple[str, int]:
        if not self.credentials:
            raise RuntimeError("GIGACHAT_CREDENTIALS пуст")
        body = ("scope=" + self.scope).encode("utf-8")
        req = urllib.request.Request(
            self.oauth_url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "RqUID": str(uuid.uuid4()),
                "Authorization": f"Basic {self.credentials}",
                "User-Agent": "Grafix",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=min(60.0, self.timeout), context=_ssl_context()) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise RuntimeError(f"GigaChat OAuth HTTP {e.code}: {detail or e.reason}") from e
        except Exception as e:
            # Common on Windows without Sber CA — retry once unverified if verify was on
            if VERIFY_SSL:
                try:
                    with urllib.request.urlopen(
                        req,
                        timeout=min(60.0, self.timeout),
                        context=ssl._create_unverified_context(),
                    ) as resp:
                        raw = resp.read().decode("utf-8", errors="replace")
                except Exception as e2:
                    raise RuntimeError(f"GigaChat OAuth: {e}; retry: {e2}") from e2
            else:
                raise RuntimeError(f"GigaChat OAuth: {e}") from e

        data = json.loads(raw)
        token = (
            data.get("access_token")
            or data.get("tok")
            or data.get("accessToken")
            or ""
        ).strip()
        if not token:
            raise RuntimeError(f"GigaChat OAuth: нет access_token в ответе: {raw[:200]}")
        expires_in = data.get("expires_at") or data.get("expires_in") or 1800
        try:
            expires_in = int(expires_in)
            # Some responses return unix timestamp in expires_at
            if expires_in > 10_000_000:
                expires_in = max(60, expires_in - int(time.time()))
        except (TypeError, ValueError):
            expires_in = 1800
        return token, expires_in

    def _auth_headers(self) -> dict[str, str]:
        token = self._ensure_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Grafix",
        }

    def _get_json(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, headers=self._auth_headers(), method="GET")
        return self._read_json(req)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers=self._auth_headers(), method="POST"
        )
        return self._read_json(req)

    def _read_json(self, req: urllib.request.Request) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=_ssl_context()) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace") if e.fp else ""
            if e.code == 401:
                # Force token refresh once
                self._access_token = None
                self._token_expires_at = 0
                if "Authorization" in (req.headers or {}):
                    try:
                        req.add_header("Authorization", f"Bearer {self._ensure_token()}")
                    except Exception:
                        pass
                    else:
                        with urllib.request.urlopen(
                            req, timeout=self.timeout, context=_ssl_context()
                        ) as resp2:
                            raw = resp2.read().decode("utf-8", errors="replace")
                            return json.loads(raw) if raw.strip() else {}
            raise RuntimeError(f"GigaChat HTTP {e.code}: {detail or e.reason}") from e
        except Exception as e:
            if VERIFY_SSL:
                try:
                    with urllib.request.urlopen(
                        req, timeout=self.timeout, context=ssl._create_unverified_context()
                    ) as resp:
                        raw = resp.read().decode("utf-8", errors="replace")
                except Exception as e2:
                    raise RuntimeError(f"GigaChat: {e}; retry: {e2}") from e2
            else:
                raise RuntimeError(f"GigaChat: {e}") from e
        return json.loads(raw) if raw.strip() else {}


def _gigachat_reasoning(data: dict[str, Any] | None) -> str:
    """Pull reasoning / thinking text from a GigaChat chat.completions response."""
    if not isinstance(data, dict):
        return ""
    try:
        msg = (data.get("choices") or [{}])[0].get("message") or {}
    except (IndexError, AttributeError, TypeError):
        return ""
    if not isinstance(msg, dict):
        return ""
    for key in ("reasoning_content", "reasoning", "thinking"):
        val = msg.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""
