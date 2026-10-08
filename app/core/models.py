"""Pydantic-модели данных pipeline (ТЗ §20, §25, §55–§60, data_model.md).

Все ответы LLM валидируются этими схемами; числовые поля ограничены 0..1,
enum'ы строго фиксированы (§56)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------- enums -----

class AnalysisMode(str, Enum):
    CONTROLLED = "CONTROLLED"   # только доверенные категории источников (§7.1)
    RESEARCH = "RESEARCH"       # расширенный поиск с оценкой каждого источника (§7.2)


class ClaimType(str, Enum):
    MEDICAL_FACT = "MEDICAL_FACT"
    STATISTICAL_CLAIM = "STATISTICAL_CLAIM"
    CAUSAL_CLAIM = "CAUSAL_CLAIM"
    SAFETY_CLAIM = "SAFETY_CLAIM"
    EFFICACY_CLAIM = "EFFICACY_CLAIM"
    DIAGNOSTIC_CLAIM = "DIAGNOSTIC_CLAIM"
    TREATMENT_CLAIM = "TREATMENT_CLAIM"
    PREVENTION_CLAIM = "PREVENTION_CLAIM"
    SCIENTIFIC_CLAIM = "SCIENTIFIC_CLAIM"
    ADVERTISEMENT = "ADVERTISEMENT"
    OPINION = "OPINION"
    PREDICTION = "PREDICTION"
    QUOTE = "QUOTE"
    ANALOGY = "ANALOGY"


#: типы claims, требующие обязательной проверки
CHECKABLE_CLAIM_TYPES = {
    ClaimType.MEDICAL_FACT, ClaimType.STATISTICAL_CLAIM, ClaimType.CAUSAL_CLAIM,
    ClaimType.SAFETY_CLAIM, ClaimType.EFFICACY_CLAIM, ClaimType.DIAGNOSTIC_CLAIM,
    ClaimType.TREATMENT_CLAIM, ClaimType.PREVENTION_CLAIM, ClaimType.SCIENTIFIC_CLAIM,
}

#: веса типов для итоговой оценки статьи (§69)
IMPORTANCE_WEIGHT: dict[ClaimType, float] = {
    ClaimType.SAFETY_CLAIM: 1.0,
    ClaimType.TREATMENT_CLAIM: 0.95,
    ClaimType.EFFICACY_CLAIM: 0.95,
    ClaimType.DIAGNOSTIC_CLAIM: 0.9,
    ClaimType.PREVENTION_CLAIM: 0.9,
    ClaimType.MEDICAL_FACT: 0.85,
    ClaimType.CAUSAL_CLAIM: 0.85,
    ClaimType.STATISTICAL_CLAIM: 0.8,
    ClaimType.SCIENTIFIC_CLAIM: 0.7,
    ClaimType.ADVERTISEMENT: 0.6,
    ClaimType.PREDICTION: 0.5,
    ClaimType.QUOTE: 0.4,
    ClaimType.ANALOGY: 0.3,
    ClaimType.OPINION: 0.2,
}


class EvidenceDirection(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    NEUTRAL = "NEUTRAL"


class VerdictLabel(str, Enum):
    VERIFIED = "VERIFIED"
    LIKELY_TRUE = "LIKELY_TRUE"
    PARTIALLY_TRUE = "PARTIALLY_TRUE"
    MISLEADING = "MISLEADING"
    LIKELY_FALSE = "LIKELY_FALSE"
    FALSE = "FALSE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ContentNature(str, Enum):
    INFORMATION = "INFORMATION"
    ADVERTISEMENT = "ADVERTISEMENT"
    NATIVE_ADVERTISEMENT = "NATIVE_ADVERTISEMENT"
    SPONSORED_CONTENT = "SPONSORED_CONTENT"
    AFFILIATE_CONTENT = "AFFILIATE_CONTENT"


class QueryPurpose(str, Enum):
    NEUTRAL = "neutral"
    SCIENTIFIC = "scientific"
    PRIMARY = "primary_study"
    REVIEW = "systematic_review"
    GUIDELINE = "guideline"
    CONTRADICTION = "contradiction"


# ------------------------------------------------------------- models -------

class Claim(BaseModel):
    """Одно проверяемое утверждение (§12, §57)."""

    id: str = Field(default_factory=lambda: new_id("claim"))
    document_id: str = ""
    text: str
    normalized_text: str = ""
    claim_type: ClaimType = ClaimType.MEDICAL_FACT
    importance: float = Field(0.5, ge=0.0, le=1.0)
    check_required: bool = True
    context: str = ""
    statistical_flags: list[str] = Field(default_factory=list)   # §39
    hedging_observed: bool = False                               # §40 (в исходном тексте)

    @field_validator("text")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("claim text must not be empty")
        return v


class SourceMeta(BaseModel):
    """Структурированное представление источника (§20, §23)."""

    id: str = Field(default_factory=lambda: new_id("source"))
    url: str
    title: str = ""
    author: str = ""
    organization: str = ""
    publication_date: str = ""
    retrieved_at: str = Field(default_factory=_utcnow)
    source_type: str = ""            # journal_article / guideline / agency_page / news / blog ...
    study_type: str = ""             # rct / cohort / meta-analysis / review / in_vitro / animal ...
    doi: str = ""
    pmid: str = ""
    tier: int = Field(8, ge=1, le=8)  # иерархия доказательств §21
    funding: str = ""
    conflicts_of_interest: str = ""
    commercial_interest: bool = False
    content_nature: ContentNature = ContentNature.INFORMATION
    trusted_category: str = ""       # ключ из config.sources.trusted_domains, если домен доверенный
    original_source_url: str = ""    # §24: для репостов — указание на первоисточник
    is_derivative: bool = False
    independent_cluster_key: str = ""
    fetched: bool = False            # реально ли получен текст (§27)
    fetch_error: str = ""
    body_text: str = ""              # извлечённый текст (для evidence extraction)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceScores(BaseModel):
    """Шесть исходных параметров + итог (§22). Всё 0..1."""

    source_id: str
    authority: float = Field(0.5, ge=0.0, le=1.0)
    methodology: float = Field(0.5, ge=0.0, le=1.0)
    recency: float = Field(0.5, ge=0.0, le=1.0)
    relevance: float = Field(0.5, ge=0.0, le=1.0)
    independence: float = Field(0.5, ge=0.0, le=1.0)
    transparency: float = Field(0.5, ge=0.0, le=1.0)
    overall: float = Field(0.5, ge=0.0, le=1.0)


class Evidence(BaseModel):
    """Фрагмент источника, поддерживающий/опровергающий claim (§25, §26, §58)."""

    id: str = Field(default_factory=lambda: new_id("evidence"))
    claim_id: str
    source_id: str
    text: str                       # дословная цитата из полученного текста
    evidence_type: str = "quote"    # quote / statistic / statement
    direction: EvidenceDirection = EvidenceDirection.NEUTRAL
    strength: float = Field(0.5, ge=0.0, le=1.0)
    relevance: float = Field(0.5, ge=0.0, le=1.0)
    quote_verified: bool = False    # цитата найдена в реально полученном тексте (§27)
    causal_note: str = ""           # §38 correlation vs causation


class SearchResult(BaseModel):
    """Нормализованный результат поиска (§66)."""

    title: str = ""
    url: str
    snippet: str = ""
    provider: str = ""
    publication_date: str = ""
    doi: str = ""
    pmid: str = ""
    source_type: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchQuery(BaseModel):
    id: str = Field(default_factory=lambda: new_id("query"))
    project_id: str = ""
    claim_id: str = ""
    provider: str = ""
    query: str
    purpose: QueryPurpose = QueryPurpose.NEUTRAL
    result_count: int = 0
    error: str = ""


class Verdict(BaseModel):
    """Итог по одному claim, рассчитанный детерминированным движком (§59, §101)."""

    claim_id: str
    label: VerdictLabel
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    explanation: str = ""
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    insufficient_reason: str = ""                 # обязателен для INSUFFICIENT_EVIDENCE (§33)
    quality_gate_passed: bool = True
    details: dict[str, float] = Field(default_factory=dict)  # diag: support/contradiction/coverage...


class ModelInfo(BaseModel):
    backend: str = ""
    model: str = ""
    version: str = ""


class Report(BaseModel):
    """Финальный отчёт (§60, §42)."""

    project_id: str = ""
    input_url: str = ""
    mode: AnalysisMode = AnalysisMode.RESEARCH
    overall_verdict: VerdictLabel = VerdictLabel.INSUFFICIENT_EVIDENCE
    overall_confidence: float = Field(0.0, ge=0.0, le=1.0)
    summary: str = ""
    claims: list[Claim] = Field(default_factory=list)
    verdicts: list[Verdict] = Field(default_factory=list)
    sources: list[SourceMeta] = Field(default_factory=list)
    scores: list[SourceScores] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)     # §104
    methodology: str = ""
    generated_at: str = Field(default_factory=_utcnow)
    model_info: ModelInfo = Field(default_factory=ModelInfo)
    counts: dict[str, int] = Field(default_factory=dict)      # сводка по вердиктам

    def verdict_counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for v in self.verdicts:
            c[v.label.value] = c.get(v.label.value, 0) + 1
        return c


class Document(BaseModel):
    """Полученный и извлечённый материал пользователя."""

    id: str = Field(default_factory=lambda: new_id("doc"))
    url: str = ""
    title: str = ""
    author: str = ""
    published: str = ""
    canonical_url: str = ""
    text: str = ""
    content_hash: str = ""
    content_nature: ContentNature = ContentNature.INFORMATION
    language: str = ""
    retrieved_at: str = Field(default_factory=_utcnow)
    references: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Project(BaseModel):
    id: str = Field(default_factory=lambda: new_id("proj"))
    input_url: str = ""
    mode: AnalysisMode = AnalysisMode.RESEARCH
    created_at: str = Field(default_factory=_utcnow)
    status: str = "NEW"   # NEW | RUNNING | DONE | FAILED
    model_info: ModelInfo = Field(default_factory=ModelInfo)


class AuditEvent(BaseModel):
    ts: str = Field(default_factory=_utcnow)
    project_id: str = ""
    stage: str = ""
    event: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)


class PipelineStage(str, Enum):
    """Этапы для progress UI (§11)."""

    ACQUIRE = "Материал получен"
    EXTRACT = "Текст извлечён"
    CLAIMS = "Claims выделены"
    PLAN = "Формируется исследовательский план"
    SEARCH = "Поиск научных источников"
    EVIDENCE = "Проверка доказательств"
    CONTRADICTION = "Поиск контраргументов"
    VERDICT = "Расчёт итоговой оценки"
    REPORT = "Формирование отчёта"
