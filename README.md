# Локальная система интеллектуального фактчекинга медицинской информации

Windows-приложение (Python 3.10+, PySide6) для evidence-based проверки достоверности
медицинской информации в интернете с **полностью локальной LLM** (llama.cpp / Ollama /
LM Studio). Облачные AI API не используются (§95 ТЗ): приватность — URL и содержимое
материалов не покидают ваш компьютер.

## Как это работает (не «URL → LLM → правда/ложь»)

```
URL → загрузка материала → извлечение текста → выделение и декомпозиция claims
    → план исследования → поиск источников (PubMed / Europe PMC / Crossref / web)
    → оценка качества источников (6 параметров, тиры доказательств)
    → извлечение evidence (подтверждения + опровержения, adversarial search)
    → проверка противоречий и цитат → детерминированный verdict-engine
    → структурированный отчёт (SQLite + audit trail + кэш)
```

LLM анализирует, классифицирует и извлекает — но **не является источником истины**:
каждый вывод привязан к реально полученному источнику; итоговые verdict и confidence
рассчитываются детерминированным Python-кодом.

## Требования

- Windows 10/11 x64 (работает также на Linux/macOS)
- Python 3.10–3.12
- Локальный LLM-бэкенд (один из):
  - **llama.cpp** (`llama-server` или `llama-cpp-python`) + GGUF-модель Qwen 7B/8B
    (рекомендуется, основной backend);
  - **Ollama** (`ollama pull qwen2.5:7b`, сервер на `127.0.0.1:11434`);
  - **LM Studio** (Local Server, OpenAI-compatible API на `127.0.0.1:1234`).

Без LLM приложение тоже запускается: анализ деградирует до детерминированного
режима с явным предупреждением в отчёте (graceful degradation).

## Установка

```bat
cd c:\factcheck
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Опциональные зависимости (можно доустановить позже):

| Пакет | Что даёт без него | Fallback |
|---|---|---|
| `trafilatura` | качественное извлечение текста статей | BeautifulSoup |
| `PyMuPDF` | чтение PDF-материалов | сообщение «PDF недоступен» |
| `llama-cpp-python` | прямой инференс GGUF | запустите `llama-server` |
| `python-dotenv` | файл `.env` | переменные окружения ОС |

### llama.cpp (основной backend)

1. Скачайте бинарники: <https://github.com/ggml-org/llama.cpp/releases> (win-cuda/cpu).
2. Запустите сервер с GGUF-моделью, например:

```bat
llama-server -m models\qwen2.5-7b-instruct-q4_k_m.gguf --port 8080 -c 8192
```

Модель (Qwen2.5-7B-Instruct, квант Q4_K_M) — любая Qwen 7B/8B GGUF; путь задаётся
в `config/config.yaml`, в коде модель не зашита.

### Ollama

```bat
winget install Ollama.Ollama
ollama pull qwen2.5:7b
```

### LM Studio

Загрузите модель, включите Local Server (порт 1234), backend = `openai_compat`.

## Конфигурация

Единый configuration layer: `config/config.yaml` (+ секреты через `.env`, см.
`.env.example`). Модель, бэкенды, лимиты (размер загрузки, редиректы, таймауты,
число страниц/запросов), доверенные домены CONTROLLED MODE, тиры источников —
только там.

**Файл конфигурации не является обязательным для запуска.** Формат конфигурации —
YAML (`config/config.yaml`), а не `config.json` — файл `config.json` проекту не
нужен. Если `config/config.yaml` отсутствует, приложение автоматически
использует встроенные безопасные значения по умолчанию (backend `llama.cpp` на
`http://127.0.0.1:8080`, стандартные лимиты ресурсов и т. д.) и корректно
запускается. Чтобы изменить путь к модели, backend или лимиты — создайте файл
`config/config.yaml` (образец уже лежит в репозитории) или укажите свой путь
через переменную окружения `FACTCHECK_CONFIG=/path/to/my.yaml`.

Файл `.env` также опционален: он нужен только для ключей внешних научных API
(`NCBI_API_KEY`, `CROSSREF_MAILTO`). Без него всё работает по публичным
эндпоинтам с пониженными rate limit.

Смена backend:

```yaml
llm:
  backend: llama.cpp   # llama.cpp | ollama | openai_compat
```

API-ключи поисковых источников (опционально, в `.env`):
`NCBI_API_KEY`, `EuropePMC` работает без ключа, `CROSSREF_MAILTO` — contact для
priority-доступа. Без ключей всё работает по публичным эндпоинтам.

## Запуск

```bat
python run.py            :: графический интерфейс
run.bat                  :: то же самое на Windows
python run.py --cli "https://example.org/article" --mode research [--json out.json]
```

### Запуск через Visual Studio Code (рекомендуется)

В репозиторий включена готовая конфигурация `.vscode/` (settings, launch, tasks).

1. Откройте папку проекта в VS Code: `File → Open Folder` (в пути желательно без
   кириллицы и пробелов).
2. Создайте виртуальное окружение и установите зависимости — через палитру задач:
   `Terminal → Run Task…` → **«Создать venv (.venv)»**, затем
   **«Установить зависимости (pip install -r requirements.txt)»**.
   Либо вручную в терминале PowerShell:
   ```powershell
   py -3 -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```
   Если активация блокируется политикой выполнения:
   `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
3. Выберите интерпретатор: `Ctrl+Shift+P` → **Python: Select Interpreter** →
   `.venv\Scripts\python.exe` (VS Code подхватит его автоматически благодаря
   `python.defaultInterpreterPath` в `.vscode/settings.json`).
4. Запуск:
   - **F5** → конфигурация **«Фактчекер: GUI»** (или **«Фактчекер: CLI»**, где URL
     запрашивается при запуске);
   - либо `Ctrl+Shift+B` — задача **«Запустить GUI»**;
   - тесты: вкладка Testing или задача **«Тесты: pytest»**.

> Ошибка `GUI недоступен (No module named 'PySide6')` означает, что VS Code
> использует системный Python вместо `.venv`, либо зависимости не установлены.
> Проверьте имя интерпретатора в правом нижнем углу окна VS Code (должно быть
> `.venv`) и повторите шаг 2.

GUI: поле URL → режим (Контролируемый / Исследовательский) → кнопка «ПРОВЕРИТЬ» →
прогресс по этапам → итоговый экран (общий вердикт, сводка, список claims с
«Подробнее», кликабельные источники). Внутренние prompts / chain-of-thought /
стектрейсы пользователю не показываются — только в лог (`data/logs/factcheck.log`,
`logging.level: DEBUG` для разработчика).

## Режимы

- **CONTROLLED** — доказательства берутся только из доверенных категорий
  (WHO, CDC, FDA, EMA, NICE, PubMed, Cochrane, Europe PMC, Crossref, госорганы,
  признанные организации, научные издательства). Список — в конфигурации.
- **RESEARCH** — расширенный поиск: система сама формирует запросы (нейтральные,
  научные, первичные исследования, обзорные, guideline, контр-аргументация),
  открывает страницы, обязана выполнить adversarial search (попытку опровергнуть
  предварительный вердикт) и независимо оценить каждый источник.

## Хранение данных и приватность

SQLite: `data/factcheck.db` (projects, documents, sources, claims, evidence,
search_queries, verdicts, reports, audit_events) + кэш загруженных документов по
(URL, content hash). История анализа, отчёты, логи — локально. Ничего не
отправляется в сторонние AI-сервисы.

## Безопасность

- SSRF-защита: блокируются localhost/private IP/file:// и metadata-endpoint'ы,
  редиректы проверяются по каждому шагу и ограничены;
- весь веб-контент размечается как UNTRUSTED DATA и подставляется в промпт строго
  после системных инструкций (защита от prompt injection);
- JavaScript не исполняется, приоритет httpx → HTML-parser; лимиты размера,
  страниц, запросов и общий timeout анализа;
- ошибка любого источника/провайдера не останавливает pipeline.

## Тесты

```bat
pip install -r requirements-dev.txt
pytest                # unit + integration + security (сеть и LLM мокнируются)
python -m compileall .
ruff check app tests
```

Тесты НЕ требуют интернета и реальной модели. Мок — только в тестах; в рабочем
режиме фиктивных источников/verdict нет: при недоступности API в отчёт попадают
«Ограничения», а недоказуемые claims получают формальный
`INSUFFICIENT_EVIDENCE` с указанием причины.

## Ограничения системы

Это **система проверки информации**, а не диагностика и не персональный врач:
диагнозы не ставятся, лечение не назначается. Результаты носят справочный
характер и не заменяют консультацию медицинского специалиста.

## Частые проблемы

| Симптом | Решение |
|---|---|
| «LLM недоступна… детерминированный режим» | запустите llama-server/Ollama/LM Studio; проверьте `llm.backend` и URL в config.yaml |
| «Модель 'qwen2.5:7b' не найдена» | `ollama pull qwen2.5:7b` или измените `llm.ollama_model` |
| GUI не открывается | `pip install PySide6`; либо `python run.py --cli URL` |
| PDF не читается | `pip install PyMuPDF` |
| Мало источников / много INSUFFICIENT_EVIDENCE | включён CONTROLLED-режим или rate-limit сети; попробуйте RESEARCH, увеличьте `research.min_sources_per_claim` |
| Сайт требует JS | текст не извлечён — это ожидаемое поведение (безопасность); попробуйте другой материал |

## Структура проекта

```
app/            core/models.py, config.py, pipeline.py, cli.py, dependencies.py
  llm/          base (structured generate+repair), llamacpp, ollama, openai_compat
  security/     urlguard (SSRF), sanitization (prompt injection)
  content/      fetcher (httpx, лимиты, кэш), extractor (trafilatura/bs4/PDF)
  claims/       extraction + decomposition + classification + importance
  research/     planner, providers (PubMed/EuropePMC/Crossref/web), dedup
  sources/      evaluator (6 параметров качества), graph (независимость)
  evidence/     extractor (цитаты+верификация), contradiction analyst
  verdict/      детерминированный движок (verdict+confidence)
  report/       генератор отчёта
  storage/      SQLite + audit + cache
  ui/           PySide6 главное окно
prompts/        тексты system-промптов по ролям (вне кода)
config/         config.yaml
docs/           architecture.md, data_model.md, security_model.md
tests/          pytest suite (unit/integration/security)
data/, reports/ локальные данные приложения (не коммитятся)
run.py, run.bat точки входа
```
