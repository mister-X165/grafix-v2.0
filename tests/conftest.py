"""Общие фикстуры pytest (ТЗ §72).

Изоляция от реального интернета и файловой системы приложения:
каждый тест получает временный data_dir / БД; сеть в тестах не используется
(реальные HTTP-вызовы делаются только через явные mock/transport).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def tmp_cfg(tmp_path, monkeypatch):
    """AppConfig с изолированными путями; секреты и сеть отключены."""
    from app.config import load_config, reset_config_cache

    cfg = load_config(ROOT / "config" / "config.yaml")
    cfg.storage.data_dir = str(tmp_path / "data")
    cfg.logging.file = str(tmp_path / "logs" / "test.log")
    cfg.llm.backend = "ollama"          # ни один провайдер в юнит-тестах не поднимается
    cfg.research.web_search_enabled = False
    reset_config_cache()
    monkeypatch.setattr("app.config.get_config", lambda: cfg)
    yield cfg
    reset_config_cache()


@pytest.fixture()
def db(tmp_cfg):
    from app.storage.db import Database

    d = Database(Path(tmp_cfg.storage.data_dir) / "test.db")
    yield d
    d.close()


class FakeLLM:
    """Заглушка LLM ТОЛЬКО для тестов (§76: mocks разрешены только в тестах)."""

    def __init__(self, response: str = "") -> None:
        self.response = response
        self.calls: list[list[dict]] = []

    def health_check(self):
        return True, "fake"

    def model_info(self):
        from app.core.models import ModelInfo
        return ModelInfo(backend="test", model="FakeLLM", version="0")

    def complete(self, messages, *, max_tokens=None, temperature=None):
        self.calls.append(messages)
        return self.response

    def structured_generate(self, messages, schema, **kw):
        import json
        self.calls.append(messages)
        data = json.loads(self.response)
        if isinstance(data, dict) and hasattr(schema, "model_validate"):
            return schema.model_validate(data)
        if isinstance(data, list) and hasattr(schema, "model_validate"):
            key = getattr(schema, "__test_list_key__", None) or "items"
            return schema.model_validate({key: data})
        return schema.model_validate(data)


@pytest.fixture()
def fake_llm():
    return FakeLLM()
