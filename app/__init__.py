"""FactCheck Local — локальная система проверки медицинской достоверности.

Пакет полностью автономен: никаких облачных LLM API (ТЗ §95) и внешних
сервисов, кроме открытых научных источников (PubMed / Europe PMC / Crossref /
веб-поиск). Оркестрация конвейера — в ``app.pipeline``; точка входа —
``python -m app`` (CLI) либо GUI при установленном PySide6.
"""

from __future__ import annotations

__version__ = "1.0.0"
