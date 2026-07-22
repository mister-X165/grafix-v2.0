# Grafix

Анализ текста → сущности и связи (**Gemma / LM Studio**, **DeepSeek 3.2 / OpenRouter**, MicroGPT) → граф Neo4j → **FastAPI** + **Vite**.

Бренд-материалы: `media/` (гайдбук, логотипы, иконки, шрифты, key visual).  
В UI подключены копии в `frontend/public/brand/`.

## Быстрый старт

### 1) LM Studio (Gemma)

1. Открой LM Studio, загрузи **Gemma**.
2. Включи **Local Server** (OpenAI-compatible), порт **1234**.
3. Проверка: http://127.0.0.1:8000/api/health → блок `lm_studio.ready: true`.

Опционально:

```
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_MODEL=
LM_STUDIO_TIMEOUT=300
LM_STUDIO_MAX_TOKENS=500
```

`LM_STUDIO_MODEL` пустой = берётся первая загруженная модель.

### 2) DeepSeek 3.2 (OpenRouter)

В UI выбери модель **DeepSeek 3.2**. Ключ — через окружение или файл `.env` в корне проекта:

```
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=deepseek/deepseek-v3.2
OPENROUTER_MODEL_V4=deepseek/deepseek-v4-pro
OPENROUTER_TIMEOUT=300
OPENROUTER_MAX_TOKENS=4000
OPENROUTER_EXTRACT_MAX_TOKENS=32000
```

В UI: **DeepSeek 3.2** (`deepseek/deepseek-v3.2`) и **DeepSeek V4 Pro** (`deepseek/deepseek-v4-pro`).

Для обеих моделей включён **reasoning** (`OPENROUTER_REASONING=1`). Для QA effort по умолчанию `high`; для извлечения графа — `medium` (`OPENROUTER_EXTRACT_REASONING_EFFORT`), чтобы JSON не обрезался thinking’ом. Отключить: `OPENROUTER_REASONING=0`. Для V4 Pro можно `OPENROUTER_REASONING_EFFORT=xhigh`.

Если в Debug снова «обрезан по max_tokens» — подними `OPENROUTER_EXTRACT_MAX_TOKENS` или поставь `OPENROUTER_EXTRACT_REASONING_EFFORT=low`.

Проверка: `/api/health` → `openrouter.ready: true` и `has_api_key: true`.

Модель по умолчанию: `deepseek/deepseek-v3.2` (OpenRouter).

### Backend (FastAPI)

```bash
cd c:\grafix
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m app
```

API: http://127.0.0.1:8000  
Docs: http://127.0.0.1:8000/docs

### Frontend (Vite)

```bash
cd c:\grafix\frontend
npm.cmd run dev
```

UI: http://127.0.0.1:5173

В промпт можно загрузить `.txt` / `.docx` / `.doc` / `.rtf` / `.pdf` (кнопка или drag-and-drop).  
Старый `.doc` читается через Microsoft Word (если установлен) или `antiword`/`catdoc`.

Галочка **«Искать в интернете»** у вопроса и у комментария сущности подмешивает сниппеты DuckDuckGo в промпт Gemma (без API-ключа).

Приоритет извлечения: **LM Studio → датасет → MicroGPT → эвристики**.

### История графов (SQLite)

Каждый анализ автоматически сохраняется в `data/grafix_history.db`.  
В UI: блок **История** (сворачивается) — вкладки прошлых итераций, кнопка **Сохранить**.

```
GET    /api/graphs
GET    /api/graphs/{id}
POST   /api/graphs
PATCH  /api/graphs/{id}
DELETE /api/graphs/{id}
```

Опционально путь к БД: `GRAFIX_HISTORY_DB=c:\grafix\data\grafix_history.db`

Регистрации пользователей пока нет — общая локальная история на машине.

### Neo4j (опционально)

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=grafix
```

### Датасет STELLAR → обучение Gemma

Исходники: `STELLAR-Complete-Clarity-main/.../texts/Alexey/`  
Пары: проза (`graph_text_3/5/7…`) + разметка (`graph_text_4/6/8…`).

```bash
cd c:\grafix
python import_stellar.py
```

Готово к LoRA / SFT:

- `data/gemma_sft.jsonl` — chat-формат (system/user/assistant) для Unsloth / Axolotl / HF TRL  
- `data/stellar_alexey.jsonl` — полные тексты + тройки  
- `data/stellar_alexey_chunks.jsonl` — короткие примеры  

LM Studio сам не обучает: тренируешь снаружи, потом грузишь адаптер/модель обратно в LM Studio.
