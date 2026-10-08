"""Claim engine (ТЗ §12–§15, §38–§40).

LLM извлекает и декомпозирует claims; детерминированный Python-код
дополнительно проверяет статистические маркеры, hedging и переносит
importance в допустимый диапазон. Ответ LLM не принимается на веру (§56).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from pydantic import BaseModel, Field

from app.config import AppConfig, get_config
from app.core.models import (CHECKABLE_CLAIM_TYPES, Claim, ClaimType, Document)
from app.llm.base import LLMProvider, StructuredOutputError
from app.prompts import build_messages, contains_injection, get_system_prompt

log = logging.getLogger(__name__)


class _LLMClaim(BaseModel):
    text: str
    normalized_text: str = ""
    claim_type: ClaimType = ClaimType.MEDICAL_FACT
    importance: float = Field(0.5, ge=0.0, le=1.0)
    check_required: bool = True
    context: str = ""
    statistical_flags: list[str] = Field(default_factory=list)
    hedging_observed: bool = False


class _ClaimsList(BaseModel):
    claims: list[_LLMClaim] = Field(default_factory=list)


# ------------------------------------------------ deterministic heuristics --

_REL_RISK_RE = re.compile(r"(на \d+\s*%|by \d+%|снижает риск.*\d+|\d+[- ]?\d*\s*%\s*(сниж|повыш|risk|reduc))", re.I)
_HEDGE_TERMS = ("может", "возможно", "предполага", "вероятно", "may", "might", "suggests",
                "associated", "связан с", "ассоциирован")
_CAUSAL_TERMS = ("вызывает", "приводит к", "cause", "causes", "do causes", "ведёт к", "provoc")
_STRONG_TERMS = ("полностью безопасен", "доказано", "учёные доказали", "гарантированно",
                 "вылечивает", "100%", "навсегда", "proven", "cures")
_ABSOLUTE_RISK_HINT = re.compile(r"(абсолютн|absolute risk)", re.I)


def heuristic_flags(text: str) -> tuple[list[str], bool]:
    """Детерминированные маркеры §39/§40 поверх ответа LLM."""
    flags: list[str] = []
    t = text.lower()
    if _REL_RISK_RE.search(t) and not _ABSOLUTE_RISK_HINT.search(t):
        flags.append("relative_vs_absolute_risk")
    if re.search(r"\d+\s*(пациент|patient|мышей|mice|rat)", t) and re.search(r"люд|human", t):
        pass  # mixed — оставляем LLM
    if any(w in t for w in ("в исследовании на мышах", "in mice", "на животных", "in vitro")) \
            and re.search(r"(люд|человек|patient|human)", t):
        flags.append("extrapolation")
    hedged = any(w in t for w in _HEDGE_TERMS)
    overstated = any(w in t for w in _STRONG_TERMS) or any(w in t for w in _CAUSAL_TERMS)
    hedging = hedged and overstated
    return flags, hedging


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


class ClaimExtractor:
    """ContentAnalyst + ClaimExtractor + ClaimDecomposer (одна LLM-роль за раз)."""

    def __init__(self, llm: LLMProvider, cfg: Optional[AppConfig] = None) -> None:
        self.llm = llm
        self.cfg = cfg or get_config()

    def extract(self, doc: Document) -> list[Claim]:
        system = get_system_prompt("claim_extractor", self.cfg.prompts_dir)
        text = doc.text[: self.cfg.limits.max_context_chars]
        if contains_injection(doc.text):
            log.warning("prompt injection markers detected in document %s (kept as data)", doc.url)
        messages = build_messages(
            system,
            f"Извлеки claims из материала «{doc.title}». Ответь строго JSON по инструкции.",
            {"document": text},
        )
        try:
            result = self.llm.structured_generate(messages, _ClaimsList)
            claims = self._to_claims(result.claims, doc)
            if claims:
                return claims
            log.info("LLM returned no claims; falling back to heuristic extraction")
        except Exception as e:  # graceful degradation (§50, §76): любая ошибка LLM — fallback
            log.error("claim extraction failed: %s — using heuristic fallback", e)
        return heuristic_claims(doc)

    def _to_claims(self, items: list[_LLMClaim], doc: Document) -> list[Claim]:
        out: list[Claim] = []
        seen: set[str] = set()
        doc_norm = normalize(doc.text)
        for it in items:
            norm = normalize(it.text)
            if not norm or norm in seen or len(it.text) < 12:
                continue
            seen.add(norm)
            # анти-галлюцинация: fragment claim должен присутствовать в документе
            core = " ".join(norm.split()[:6])
            if core and core not in doc_norm:
                log.debug("claim not found verbatim in document, keeping with lowered importance: %r", it.text[:60])
            hflags, hhedging = heuristic_flags(it.text)
            flags = sorted(set(it.statistical_flags) | set(hflags))
            ctype = it.claim_type
            importance = it.importance
            if ctype in CHECKABLE_CLAIM_TYPES and importance < 0.5:
                importance = max(importance, 0.5)   # медицинские факты важны по умолчанию
            out.append(Claim(
                document_id=doc.id, text=it.text.strip(),
                normalized_text=(it.normalized_text.strip() or it.text.strip()),
                claim_type=ctype, importance=round(min(1.0, max(0.0, importance)), 2),
                check_required=bool(ctype in CHECKABLE_CLAIM_TYPES and it.check_required),
                context=it.context.strip()[:500], statistical_flags=flags,
                hedging_observed=bool(it.hedging_observed or hhedging),
            ))
        out.sort(key=lambda c: (-c.importance, -c.check_required))
        return out[:20]


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MED_KEYWORDS = re.compile(
    r"(препарат|лекарств|болезн|заболеван|лечени|профилактик|диагноз|эффект|безопасн|"
    r"риск|исследовани|врач|доз|вакцин|вирус|бактери|симптом|marjuana|drug|disease|"
    r"treatment|medicine|health|risk|study|clinical|percent|%)", re.I)
_MULTI_CLAIM = re.compile(r",(?:\s*(?:и|а также|also))?\s+(?:полностью|не|при|предотвр|вызыв|лечи|снижа|улучш)", re.I)


def heuristic_claims(doc: Document) -> list[Claim]:
    """Fallback без LLM: предложения с медицинской лексикой → claims.

    Используется только если LLM недоступна/ошиблась; помечается пониженной
    уверенностью через importance по эвристике.
    """
    out: list[Claim] = []
    sentences = _SENT_SPLIT.split(re.sub(r"\s+", " ", doc.text))
    for s in sentences:
        s = s.strip()
        if len(s) < 40 or not _MED_KEYWORDS.search(s):
            continue
        imp = 0.7
        low = s.lower()
        if any(w in low for w in ("%", "процент", "риск", "смерт")):
            imp = 0.85
            ctype = ClaimType.STATISTICAL_CLAIM
        elif "безопас" in low:
            imp = 0.95
            ctype = ClaimType.SAFETY_CLAIM
        elif "леч" in low or "терапи" in low:
            imp = 0.9
            ctype = ClaimType.TREATMENT_CLAIM
        elif any(w in low for w in ("вызыв", "приводит", "cause")):
            ctype = ClaimType.CAUSAL_CLAIM
        else:
            ctype = ClaimType.MEDICAL_FACT
        flags, hedging = heuristic_flags(s)
        out.append(Claim(document_id=doc.id, text=s, normalized_text=s,
                         claim_type=ctype, importance=imp, check_required=True,
                         context="", statistical_flags=flags, hedging_observed=hedging))
        if len(out) >= 12:
            break
    return out
