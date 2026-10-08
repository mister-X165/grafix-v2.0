"""Typed configuration layer.

Единый источник истины о модели, бэкенде LLM, лимитах ресурсов и доверенных
доменах (см. ТЗ §5–§7, §32, §95). Секреты берутся из .env / переменных
окружения, NEVER из кода.

Использование::

    from app.config import get_config
    cfg = get_config()          # загружается один раз, кэшируется
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "config.yaml"


class LLMConfig(BaseModel):
    """Локальная LLM: llama.cpp (основной), Ollama, OpenAI-compatible (LM Studio)."""

    backend: Literal["llama.cpp", "ollama", "openai_compat"] = "llama.cpp"
    model_path: str = ""
    llama_server_url: str = "http://127.0.0.1:8080"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b"
    openai_compat_base_url: str = "http://127.0.0.1:1234/v1"
    openai_compat_model: str = ""
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, ge=64, le=32768)
    request_timeout_s: float = Field(300.0, gt=0)
    max_retries: int = Field(2, ge=0, le=5)


class LimitsConfig(BaseModel):
    """Resource limits (§32). Все значения конфигурируемы."""

    max_download_size_bytes: int = Field(10 * 1024 * 1024, ge=1024)
    max_pages_per_analysis: int = Field(25, ge=1)
    max_redirects: int = Field(5, ge=0, le=10)
    max_requests_per_analysis: int = Field(120, ge=1)
    request_timeout_s: float = Field(30.0, gt=0)
    total_analysis_timeout_s: float = Field(1800.0, gt=0)
    max_pdf_pages: int = Field(40, ge=1)
    max_context_chars: int = Field(24000, ge=1000)
    search_concurrency: int = Field(4, ge=1, le=16)
    fetch_concurrency: int = Field(3, ge=1, le=16)


class ResearchConfig(BaseModel):
    mode_default: Literal["controlled", "research"] = "research"
    min_sources_per_claim: int = Field(3, ge=1)
    adversarial_search: bool = True
    web_search_enabled: bool = True
    pubmed_enabled: bool = True
    europepmc_enabled: bool = True
    crossref_enabled: bool = True


class SourcesConfig(BaseModel):
    trusted_domains: dict[str, list[str]] = Field(default_factory=dict)
    tier_domains: dict[int, list[str]] = Field(default_factory=dict)
    news_domains: list[str] = Field(default_factory=list)
    low_quality_signals: list[str] = Field(default_factory=list)

    @field_validator("tier_domains", mode="before")
    @classmethod
    def _keys_to_int(cls, v: object) -> object:
        if isinstance(v, dict):
            return {int(k): list(val) for k, val in v.items()}
        return v

    def all_trusted_domains(self) -> set[str]:
        out: set[str] = set()
        for domains in self.trusted_domains.values():
            out.update(d.lower().strip() for d in domains if d.strip())
        return out


class StorageConfig(BaseModel):
    data_dir: str = "data"
    db_name: str = "factcheck.db"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "data/logs/factcheck.log"


class UIConfig(BaseModel):
    debug_mode: bool = False


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    config_path: Path = DEFAULT_CONFIG_PATH

    # --- derived paths -----------------------------------------------------
    @property
    def root(self) -> Path:
        return ROOT

    @property
    def data_dir(self) -> Path:
        p = ROOT / self.storage.data_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.storage.db_name

    @property
    def prompts_dir(self) -> Path:
        return ROOT / "prompts"

    @property
    def reports_dir(self) -> Path:
        p = ROOT / "reports"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # --- secrets via env (never hardcoded) ---------------------------------
    def api_key(self, name: str) -> str:
        """Ключ внешнего search API (NCBI/Crossref). Пустая строка, если не задан."""
        return os.environ.get(name, "").strip()


def load_dotenv(path: Path | None = None) -> None:
    """Минимальный загрузчик .env (без внешней зависимости)."""
    try:
        from dotenv import load_dotenv as _ld  # опционально

        _ld(path or ROOT / ".env")
        return
    except ImportError:
        pass
    p = path or ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def load_config(path: Path | None = None) -> AppConfig:
    """Загрузить YAML-конфигурацию поверх значений по умолчанию."""
    load_dotenv()
    cfg_path = Path(path) if path else Path(os.environ.get("FACTCHECK_CONFIG", DEFAULT_CONFIG_PATH))
    data: dict = {}
    if cfg_path.exists():
        loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data = loaded
    data["config_path"] = cfg_path
    return AppConfig(**data)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return load_config()


def reset_config_cache() -> None:
    """Для тестов: сбросить кэш конфигурации.

    Устойчиво к monkeypatch-замене get_config на обычную lambda (в этом случае
    кэша нет и сбрасывать нечего).
    """
    fn = globals().get("get_config")
    clear = getattr(fn, "cache_clear", None)
    if callable(clear):
        clear()
