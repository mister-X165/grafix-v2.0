"""Точка запуска приложения (ТЗ §94).

    python run.py            — графический интерфейс (PySide6)
    python run.py --cli URL  — анализ без GUI (подробности: python run.py --help)

Все настройки берутся из единого configuration layer (config/config.yaml + .env),
модель и backend в этом файле НЕ зашиты (§6, §95: локальная LLM по умолчанию).
"""

from __future__ import annotations

import sys


def main() -> int:
    args = sys.argv[1:]
    if "--cli" in args:
        args = [a for a in args if a != "--cli"]
        from app.cli import main as cli_main
        return cli_main(args)
    # GUI по умолчанию; при отсутствии PySide6 — понятное сообщение и подсказка (§51)
    try:
        from app.ui.main_window import run_gui
    except ImportError as e:
        print(f"GUI недоступен ({e}).\n"
              "Установите зависимости:  pip install -r requirements.txt\n"
              "Или запустите CLI:       python run.py --cli <URL> [--mode controlled|research]",
              file=sys.stderr)
        return 1
    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
