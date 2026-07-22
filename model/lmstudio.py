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
Прочитай текст и извлеки явные факты как тройки (субъект, связь, объект) для построения графа.

ФОРМАТ ОТВЕТА (строго):
Один JSON-объект и ничего больше. Без markdown, без ```, без текста до/после JSON.
{"triples":[{"subject":"Имя","relation":"тип_связи","object":"Имя"}]}

ПРАВИЛА:
1. Только факты из текста. Не додумывай.
2. subject и object — короткие канонические имена (организации, модели, люди, проекты, холдинги).
   Не делай объектом длинную фразу («выпуск дешевых моделей») — лучше отдельная сущность или пропусти.
3. relation — один короткий snake_case: работает_в, основал, производит, входит_в, создан_в, связан_с, торговая_марка.
4. Не больше 30 троек. Если фактов больше — только самые важные для графа связей.
5. JSON обязан быть полным: закрой все кавычки, скобки и массив triples.
6. Если фактов нет: {"triples":[]}
7. Не пиши анализ прозой — только JSON.
"""

# DeepSeek: explicit + hidden + false + contradictions (reasoning-aware extract)
DEEPSEEK_SYSTEM_PROMPT = """Ты — модуль извлечения графа знаний для программы Grafix (движок DeepSeek).

Вход: длинный связный текст. Он может содержать неочевидные факты и связи, сущности-дистракторы (упомянуты, но реально ни с чем не связаны) и намеренные противоречия (чаще временные). Твоя работа — извлечь достоверный граф, вскрыть скрытые связи и НЕ попасться на ловушки.

═══ ПОРЯДОК РАБОТЫ (рассуждай про себя; в ответ вывод рассуждений не пиши) ═══
1. Инвентаризация. Выпиши все сущности. Слей синонимы, местоимения, должности и описания к одному канону («он», «директор», «компания» → конкретное имя). Сущность без единой обоснованной связи с остальными помечай как дистрактор; связей ей НЕ выдумывай. Сомневаешься — не дистрактор.
2. Явные факты. Извлеки только то, что прямо утверждается.
3. Временная шкала. Восстанови порядок событий и даты, сверь на согласованность.
4. Скрытые связи. Выведи то, что логически следует из фактов текста (паттерны ниже).
5. Опровержения и противоречия. Отметь связи, которые текст объявляет ошибочными, и пары фактов, которые несовместимы.

═══ СКРЫТАЯ СВЯЗЬ (kind:"hidden") ═══
Связь, НЕ названная прямо, но однозначно выводимая ТОЛЬКО из фактов текста. Допустимые паттерны:
- транзитивность: A→B и B→C ⟹ A→C;
- общий узел: A и B оба связаны с C ⟹ возможная связь A–B;
- разрешение личности: два разных упоминания = одна сущность;
- следствие: из утверждённого факта неизбежно вытекает другой;
- причинная цепочка, собранная из нескольких предложений.
У каждой hidden-связи ОБЯЗАТЕЛЬНО поле evidence — какие именно факты текста её порождают. Нет вывода из текста — нет связи.

═══ АНТИ-ГАЛЛЮЦИНАЦИЯ (жёстко) ═══
- НИКОГДА не используй знания «извне» — только сам текст.
- Соседство/совместное упоминание — это НЕ связь. Близость в тексте ничего не доказывает.
- Сомневаешься, реальна ли связь → hidden с низким confidence либо опусти.
- Не «чини» противоречия, выбирая правдоподобный вариант. Твоя задача — зафиксировать конфликт, а не разрешить его.
- Дистракторам связей ради полноты не добавляй.

═══ ПРОТИВОРЕЧИЯ (робастность) ═══
Если два факта не могут быть истинны одновременно (несовместимый порядок или даты для одного события/сущности) — вынеси их в contradictions. Спорный факт при этом НЕ выводи как обычную explicit-тройку (иначе граф утверждает ложное ребро). Остальные, непротиворечивые факты извлекай как обычно.

═══ ФОРМАТ ОТВЕТА (строго) ═══
Ровно один JSON-объект и ничего кроме него. Без markdown, без обратных кавычек, без текста до/после. Двойные кавычки, без висячих запятых, без комментариев.

{
  "entities":[
    {"name":"Канон","type":"person|org|place|event|concept|other","aliases":["…"],"distractor":false}
  ],
  "triples":[
    {"subject":"Канон","relation":"snake_case","object":"Канон","kind":"explicit"},
    {"subject":"Канон","relation":"snake_case","object":"Канон","kind":"hidden","evidence":"из [факт1] и [факт2] следует …","confidence":0.85},
    {"subject":"Канон","relation":"snake_case","object":"Канон","kind":"false","evidence":"текст утверждает обратное: …"}
  ],
  "contradictions":[
    {"type":"temporal|factual","statements":["утверждение A","утверждение B"],"entities":["…"],"note":"почему несовместимы","confidence":0.9}
  ]
}

═══ ПРАВИЛА ПОЛЕЙ ═══
1. subject/object — короткие канонические имена ИЗ entities. Никаких местоимений и описаний.
2. relation — один короткий snake_case; направление subject→object держи единообразно; переиспользуй уже введённые типы связей.
3. kind обязателен: explicit / hidden / false. evidence обязателен у hidden и false. confidence (0.0–1.0) — у hidden и contradictions.
4. Для false передавай суть отвергаемой связи (например: causes, связан_с), а сам факт опровержения — в evidence.
5. Приоритет — самые обоснованные выводы. Список слабыми догадками не добивай. Дубли не повторяй.
6. Лимиты (тюнингуемые): explicit — без жёсткого потолка; hidden — до 12; false — до 10. Качество важнее количества.
7. Если извлекать нечего: {"entities":[],"triples":[],"contradictions":[]}.
8. Только JSON. Никакого анализа прозой.

Микропример формата (схематично):
{"entities":[{"name":"Орлов","type":"person","aliases":["директор"],"distractor":false},{"name":"Аркада","type":"org","aliases":[],"distractor":false},{"name":"Актив-X","type":"concept","aliases":[],"distractor":false}],"triples":[{"subject":"Орлов","relation":"руководит","object":"Аркада","kind":"explicit"},{"subject":"Орлов","relation":"контролирует","object":"Актив-X","kind":"hidden","evidence":"из [Орлов руководит Аркадой] и [Аркада владеет Активом-X] следует контроль","confidence":0.7}],"contradictions":[]}
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

ENTITY_COMMENTS_BATCH_SYSTEM = """Ты — модуль комментариев графа знаний Grafix.

КОНТЕКСТ: граф уже построен (список сущностей и троек дан ниже).
ЗАДАЧА: строго для КАЖДОЙ сущности из списка напиши краткий комментарий по тексту документа и её связям.

ФОРМАТ (строго):
Один JSON-объект и ничего больше. Без markdown, без ```.
{"comments":{"Имя сущности":"текст комментария"}}

ПРАВИЛА:
1. Ключ в comments — точное имя сущности из списка (без переименований).
2. Комментарий: 1–3 предложения на русском, по делу. Без списков и заголовков.
3. Только факты из документа и связей. Не додумывай.
4. Обязательно закрой JSON. Не пропускай ни одну сущность из списка.
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
            # Fix trailing commas / truncated closing brackets
            repaired = _repair_truncated_json(cand)
            try:
                data = json.loads(re.sub(r",\s*([}\]])", r"\1", repaired))
            except json.JSONDecodeError:
                continue
        triples = _triples_from_data(data)
        if triples:
            return triples

    # Truncated model output: salvage complete triple objects via regex
    salvaged = _salvage_triple_objects(text)
    if salvaged:
        return salvaged

    # Arrow / dash lines: A — rel → B
    lined = _parse_arrow_lines(content)
    if lined:
        return lined

    return _parse_loose_lines(content)


def _repair_truncated_json(text: str) -> str:
    """Best-effort close of cut-off {"triples":[... JSON."""
    s = text.strip()
    if not s:
        return s
    # Drop trailing incomplete key/value fragment after last complete object
    last_obj = s.rfind("}")
    if last_obj >= 0 and ("[" in s or "{" in s):
        s = s[: last_obj + 1]
    # Balance brackets
    opens_curly = s.count("{") - s.count("}")
    opens_square = s.count("[") - s.count("]")
    if opens_curly < 0 or opens_square < 0:
        return text.strip()
    # Remove dangling comma before close
    s = re.sub(r",\s*$", "", s)
    s += "]" * max(0, opens_square) + "}" * max(0, opens_curly)
    return s


def _salvage_triple_objects(text: str) -> list[Triple]:
    """Pull complete {subject,relation,object} dicts from truncated JSON."""
    out: list[Triple] = []
    seen: set[tuple[str, str, str]] = set()
    # Allow any key order inside a shallow object
    obj_re = re.compile(r"\{([^{}]+)\}")
    for block in obj_re.finditer(text):
        body = block.group(0)
        if "subject" not in body.lower() or "object" not in body.lower():
            continue
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            try:
                data = json.loads(body.replace("'", '"'))
            except json.JSONDecodeError:
                continue
        t = _triple_from_item(data)
        if not t:
            continue
        key = (t.subject, t.relation, t.object)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def parse_comments_flexible(content: str) -> dict[str, str]:
    """Parse {"comments":{...}} or a flat name→text map from model output."""
    if not content or not content.strip():
        return {}

    text = content.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    if fence:
        text = fence.group(1).strip()

    candidates = _json_candidates(text)
    for cand in candidates:
        data = None
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            repaired = _repair_truncated_json(cand)
            try:
                data = json.loads(re.sub(r",\s*([}\]])", r"\1", repaired))
            except json.JSONDecodeError:
                continue
        comments = _comments_from_data(data)
        if comments:
            return comments

    return _salvage_comment_pairs(text)


def _comments_from_data(data: Any) -> dict[str, str]:
    raw: Any = None
    if isinstance(data, dict):
        for key in ("comments", "entity_comments", "комментарии", "notes"):
            if isinstance(data.get(key), dict):
                raw = data[key]
                break
        if raw is None:
            if "triples" in data:
                return {}
            raw = data
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        name = str(k or "").strip()
        if not name or name.lower() in {"triples", "comments", "relations"}:
            continue
        if isinstance(v, dict):
            v = v.get("comment") or v.get("text") or v.get("value") or ""
        comment = str(v or "").strip()
        if name and comment:
            out[name] = comment
    return out


def _salvage_comment_pairs(text: str) -> dict[str, str]:
    """Best-effort \"Name\": \"comment...\" pairs from truncated JSON."""
    out: dict[str, str] = {}
    pair_re = re.compile(r'"((?:\\.|[^"\\])+)"\s*:\s*"((?:\\.|[^"\\])*)"')
    skip = {"triples", "comments", "subject", "relation", "object", "relations"}
    for m in pair_re.finditer(text):
        key = m.group(1).encode("utf-8").decode("unicode_escape", errors="ignore").strip()
        val = m.group(2).encode("utf-8").decode("unicode_escape", errors="ignore").strip()
        if not key or key.lower() in skip:
            continue
        if len(val) < 12:
            continue
        out[key] = val
    return out


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
        from model.triples import normalize_edge_kind

        kind_raw = _pick(
            item,
            ("kind", "edge_kind", "link_kind", "link_type", "класс", "вид", "тип_ребра"),
        )
        evidence = str(
            _pick(item, ("evidence", "reason", "rationale", "обоснование", "доказательство"))
            or ""
        ).strip()
        conf_raw = _pick(item, ("confidence", "score", "prob", "уверенность"))
        confidence: float | None = None
        if conf_raw is not None and conf_raw != "":
            try:
                confidence = float(conf_raw)
                if confidence < 0:
                    confidence = 0.0
                elif confidence > 1:
                    confidence = 1.0
            except (TypeError, ValueError):
                confidence = None
        return Triple(
            s,
            r,
            o,
            kind=normalize_edge_kind(kind_raw),
            evidence=evidence,
            confidence=confidence,
        )
    return None


def parse_extract_bundle(content: str) -> dict[str, Any]:
    """Parse DeepSeek extract JSON: triples + entities + contradictions."""
    triples = parse_triples_flexible(content)
    entities: list[dict[str, Any]] = []
    contradictions: list[dict[str, Any]] = []
    if not content or not content.strip():
        return {"triples": triples, "entities": entities, "contradictions": contradictions}

    text = content.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    if fence:
        text = fence.group(1).strip()

    data = None
    for cand in _json_candidates(text):
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            repaired = _repair_truncated_json(cand)
            try:
                data = json.loads(re.sub(r",\s*([}\]])", r"\1", repaired))
            except json.JSONDecodeError:
                continue
        if isinstance(data, dict):
            break
        data = None

    if isinstance(data, dict):
        entities = _entities_from_data(data.get("entities"))
        contradictions = _contradictions_from_data(data.get("contradictions"))
    return {"triples": triples, "entities": entities, "contradictions": contradictions}


def _entities_from_data(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = _as_name(_pick(item, ("name", "entity", "label", "id", "имя")))
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        etype = str(_pick(item, ("type", "kind", "category", "тип")) or "other").strip().lower()
        if etype not in {"person", "org", "place", "event", "concept", "other"}:
            etype = "other"
        aliases_raw = _pick(item, ("aliases", "aka", "синонимы")) or []
        aliases: list[str] = []
        if isinstance(aliases_raw, list):
            for a in aliases_raw:
                s = str(a or "").strip()
                if s and s.lower() != name.lower() and s not in aliases:
                    aliases.append(s)
        distractor = bool(_pick(item, ("distractor", "is_distractor", "дистрактор")))
        out.append(
            {
                "name": name,
                "type": etype,
                "aliases": aliases[:12],
                "distractor": distractor,
            }
        )
    return out


def _contradictions_from_data(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        ctype = str(_pick(item, ("type", "kind", "тип")) or "factual").strip().lower()
        if ctype not in {"temporal", "factual"}:
            ctype = "factual"
        statements_raw = _pick(item, ("statements", "claims", "утверждения")) or []
        statements: list[str] = []
        if isinstance(statements_raw, list):
            for s in statements_raw:
                t = str(s or "").strip()
                if t:
                    statements.append(t)
        ents_raw = _pick(item, ("entities", "names", "сущности")) or []
        entities: list[str] = []
        if isinstance(ents_raw, list):
            for e in ents_raw:
                n = str(e or "").strip()
                if n and n not in entities:
                    entities.append(n)
        note = str(_pick(item, ("note", "reason", "explanation", "пояснение")) or "").strip()
        conf_raw = _pick(item, ("confidence", "score"))
        confidence: float | None = None
        if conf_raw is not None and conf_raw != "":
            try:
                confidence = float(conf_raw)
                confidence = max(0.0, min(1.0, confidence))
            except (TypeError, ValueError):
                confidence = None
        if not statements and not note:
            continue
        row: dict[str, Any] = {
            "type": ctype,
            "statements": statements[:6],
            "entities": entities[:12],
            "note": note,
        }
        if confidence is not None:
            row["confidence"] = confidence
        out.append(row)
    return out


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
