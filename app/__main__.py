"""Точка входа: CLI по умолчанию; GUI — при явном ``--gui`` и PySide6 (ТЗ §1, §113).

    python -m app                      -- справка по CLI
    python -m app URL [--mode ...]     -- CLI-анализ
    python -m app --gui                -- графический интерфейс (нужен PySide6)
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if "--gui" in args:
        try:
            from app.ui.main_window import run_gui  # PySide6-интерфейс (§10)
        except ImportError as e:
            print(f"GUI недоступен ({e}).\nУстановите PySide6: pip install PySide6\n"
                  "Или используйте CLI: python -m app <URL>", file=sys.stderr)
            return 1
        return run_gui()

    from app.cli import main as cli_main

    return cli_main([a for a in args if a != "--cli"])


if __name__ == "__main__":
    sys.exit(main())
