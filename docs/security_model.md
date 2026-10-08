# Security model

## 1. Угрозы и контрмеры

| Угроза | Контрмера | Модуль |
|---|---|---|
| SSRF (localhost, private IP, file://, cloud metadata 169.254.169.254) | Строгая валидация схемы (только http/https), разрешение DNS и проверка всех IP против private/loopback/link-local/reserved диапазонов, повторная проверка после каждого redirect, лимит redirect'ов | `app/security/urlguard.py` |
| Prompt injection из веб-контента | Весь внешний контент маркируется как UNTRUSTED DATA и оборачивается `<untrusted_content id=...>`; системный промпт запрещает исполнять инструкции из данных; иерархия SYSTEM → APP → USER → UNTRUSTED (§29); текст не попадает в system-роль | `app/prompts.py`, все LLM-задачи |
| Ресурсное истощение (огромные страницы, бесконечные редиректы) | max_download_size (streaming-abort), request_timeout, total_analysis_timeout, max_pages/max_requests на анализ, max_pdf_pages, обрезка контекста | `config`, `content/fetcher.py`, `core/pipeline.py` |
| XSS/выполнение JS при извлечении | Обычный HTTP-fetch без исполнения JS; Playwright — только по явному включению, изолированный контекст; ничего не исполняется и не устанавливается автоматически | `content/fetcher.py` |
| Галлюцинации LLM (вымышленные DOI/PMID/URL/цитаты) | URL берутся только из реально полученных источников; цитата evidence проверяется как подстрока полученного текста (`quote_verified`); DOI/PMID сверяются с ответом API; quality gate перед отчётом (§102) | `evidence/extractor.py`, `report/generator.py` |
| Вредоносный HTML/PDF | Парсинг только текстового извлечения; бинарные данные не интерпретируются; битый PDF → graceful error; size limits до парсинга | `content/extractor.py` |
| Злоупотребление INSUFFICIENT_EVIDENCE | Метрика insufficient_evidence_rate + тесты на adversarial set | `metrics.py`, tests |
| Утечка данных в облако | Локальная LLM по умолчанию; внешние запросы только к search/source API; никаких телеметрии/внешних LLM API | `llm/*`, `research/providers.py` |

## 2. Модель доверия

- Домены из `config.trusted_domains` (CONTROLLED MODE): WHO/CDC/FDA/EMA/NICE/PubMed/Cochrane/EuropePMC/Crossref и др. — заданы в `config/config.yaml`, а не в коде.
- Любой URL из результатов поиска перед загрузкой проходит ту же SSRF-валидацию, что и пользовательский ввод.
- Коммерческий интерес/реклама помечаются в модели источника и выводятся в отчёте.

## 3. Границы ответственности UI

Приложение — система проверки информации, не диагностическая. Дисклеймер присутствует в каждом отчёте и на главном экране (§70).
