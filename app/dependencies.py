"""Проверка обязательных и опциональных зависимостей (ТЗ §51, §113).

Отсутствие обязательных → понятное сообщение и выход. Отсутствие опциональных → функция
отключается с явным указанием пользователю, приложение продолжает работу.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass

OPTIONAL = {
    "trafilatura": "качественное извлечение текста статей (fallback: BeautifulSoup)",
    "pymupdf": "чтение PDF-материалов",
    "llama_cpp": "прямой инференс llama.cpp (fallback: llama-server HTTP)",
    "PySide6": "графический интерфейс (fallback: режим --cli)",
    "dotenv": "загрузка .env (fallback: переменные окружения ОС)",
}

REQUIRED = ["httpx", "yaml", "pydantic", "bs4", "lxml"]


@dataclass
class DependencyStatus:
    missing_required: list[str]
    missing_optional: dict[str, str]   # пакет → что отключается

    @property
    def ok(self) -> bool:
        return not self.missing_required

    def human_report(self) -> str:
        lines = []
        for m in self.missing_required:
            lines.append(f"ОБЯЗАТЕЛЬНАЯ зависимость отсутствует: {m}")
        for m, what in self.missing_optional.items():
            lines.append(f"Не установлено {m} — отключено: {what}")
        return "\n".join(lines)


def check_dependencies() -> DependencyStatus:
    missing_req: list[str] = []
    missing_opt: dict[str, str] = {}
    for mod in REQUIRED:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing_req.append(mod)
    for mod, what in OPTIONAL.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            missing_opt[mod] = what
    return DependencyStatus(missing_req, missing_opt)


def ensure_or_exit() -> None:
    st = check_dependencies()
    if not st.ok:
        print("Невозможно запустить приложение:\n" + st.human_report(), file=sys.stderr)
        print("Установите зависимости: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)
