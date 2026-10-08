# Архитектура: локальная система интеллектуального фактчекинга медицинской информации

## 1. Анализ исходного репозитория

Репозиторий — приложение **Grafix**: анализ текста → извлечение сущностей/связей (LM Studio / OpenRouter / MicroGPT) → граф Neo4j → FastAPI + Vite.

Полезные наработки, которые переиспользуются как архитектурные образцы (не код напрямую):

| Наработка Grafix | Применение в новом проекте |
|---|---|
| `model/lmstudio.py` — клиент OpenAI-compatible локального API, строгие JSON-промпты, анти-галлюцинационные контракты | Слой `llm/openai_compat.py` (LM Studio backend) + принципы structured output |
| `model/infer.py`, `microgpt.py` — локальный инференс, controlled dependency loading | Паттерн для llama.cpp provider и graceful degradation |
| `graph/websearch.py` — DuckDuckGo без API-ключа, «никогда не бросает» | Провайдер `web` в `research/providers.py` |
| `graph/history.py` — SQLite-история, JSON-колонки, миграции через PRAGMA table_info | Слой `storage/db.py` |
| Строгие системные промпты с правилами «только из текста, не додумывай» | Основа prompt-библиотеки с защитой от инъекций |

Что НЕ переносится: Neo4j, облачные API (OpenRouter/GigaChat) — по ТЗ запрещены внешние LLM API; фронтенд Vite — UI делается на PySide6.

## 2. Ключевые противоречия ТЗ и их разрешение

1. **«локальное приложение» vs «интернет для поиска»** → локальной является LLM и хранение данных; сетевой доступ разрешён только к search/source API по белому списку режимов. Это фиксируется в security model.
2. **«PySide6 GUI» vs «headless-серверное окружение разработки»** → GUI изолирован в `app/ui/`; ядро (`core/`) не импортирует Qt и полностью тестируется без дисплея. При отсутствии PySide6 запускается CLI-режим (`run.py --cli`).
3. **«llama.cpp основной backend» vs «тестируемость без модели»** → `LLMProvider` — интерфейс; в рантайме доступны llama.cpp / Ollama / OpenAI-compatible; mock допускается только в тестах (ТЗ п.76).
4. **«RESEARCH MODE — широкий интернет» vs SSRF/resource limits** → любой исходящий запрос проходит через `security.urlguard` (в т.ч. результаты поиска перед fetch).
5. **«LLM интерпретирует» vs «детерминированный verdict»** → `verdict/engine.py` — чистая детерминированная функция от evidence; LLM влияет только через извлечённые структурированные данные (направление, сила, релевантность), прошедшие валидацию.

## 3. Компонентная схема (pipeline из §2 ТЗ)

```
UI (PySide6) ──► PipelineOrchestrator (async, этапы+прогресс)
                    │
   acquisition ─────┤ urlguard(SSRF) → httpx fetch (limits) → trafilatura/bs4/PyMuPDF
   claims ──────────┤ ClaimExtractor → ClaimDecomposer → classification/importance (LLM+Pydantic)
   planning ────────┤ ResearchPlanner (queries: neutral/scientific/primary/review/guideline/contradiction)
   search ──────────┤ SearchProvider(s): pubmed, europepmc, crossref, web(DDG), trusted-domain
   sources ─────────┤ dedup(canonical/DOI/PMID) → SourceEvaluator (6 scores) → independence/graph
   evidence ────────┤ EvidenceExtractor (quote must be substring of retrieved text)
   adversarial ─────┤ ContradictionAnalyst (обязателен в RESEARCH MODE)
   verdict ─────────┤ VerdictEngine (детерминированный) + confidence calc
   report ──────────┤ ReportGenerator (LLM summary, ссылки только из реальных источников)
                    │
   storage ─────────┴ SQLite: projects/documents/sources/claims/evidence/search_queries/
                               verdicts/reports/audit_events + cache (url+content hash)
```

## 4. Модульный граф

```
run.py, cli.py
app/config.py            typed config (YAML + .env), единый источник истины о модели/лимитах/доменах
app/logging_setup.py     file+console logging
app/core/models.py       Pydantic-схемы: Claim, Source, Evidence, Verdict, Report, SearchResult
app/core/pipeline.py     оркестратор, события прогресса
app/security/urlguard.py валидация URL, SSRF, redirect policy, лимиты
app/content/fetcher.py   async httpx fetch + size/timeout limits + PDF
app/content/extractor.py trafilatura/bs4 извлечение, metadata, canonical
app/llm/base.py          LLMProvider ABC, structured_generate (parse→validate→repair→retry)
app/llm/llamacpp.py      llama.cpp (python bindings или llama-server)
app/llm/ollama.py        Ollama HTTP API
app/llm/openai_compat.py LM Studio / любой local OpenAI-compat
app/prompts.py           загрузка prompts/*.txt, обёртка untrusted content
app/claims/extractor.py  извлечение/декомпозиция/классификация claims
app/research/planner.py  генерация поисковых запросов
app/research/providers.py SearchProvider + PubMed/EuropePMC/Crossref/Web
app/research/dedup.py    дедупликация, canonicalization
app/sources/evaluator.py authority/methodology/recency/relevance/independence/transparency
app/sources/graph.py     original vs derivative (упрощённый source graph)
app/evidence/extractor.py извлечение evidence (strict quote verification)
app/evidence/contradiction.py adversarial pass
app/verdict/engine.py    детерминированный расчёт verdict/confidence
app/report/generator.py  итоговый отчёт + quality gate (§102)
app/storage/db.py        SQLite + audit trail + cache
app/ui/main_window.py    PySide6 (главный экран, прогресс, отчёт, детали claim)
app/ui/history_window.py история анализов
```

## 5. Принципы

- LLM ≠ источник истины; каждый вывод привязан к evidence с верифицируемой цитатой.
- Внешний контент — UNTRUSTED DATA, оборачивается в разметку `<untrusted_content>`; инструкции из него не исполняются.
- Ошибка одного источника не останавливает pipeline (graceful degradation, снижается confidence).
- Никаких фиктивных данных: если API недоступен — фиксируется в отчёте и audit.
