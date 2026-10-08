"""Логирование в файл + консоль (ТЗ §52)."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import AppConfig, get_config


def setup_logging(cfg: AppConfig | None = None) -> logging.Logger:
    cfg = cfg or get_config()
    level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    log_path = Path(cfg.logging.file)
    if not log_path.is_absolute():
        log_path = cfg.root / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(max(level, logging.INFO))
    root.addHandler(sh)
    return logging.getLogger("factcheck")
