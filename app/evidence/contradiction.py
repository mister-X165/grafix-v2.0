"""Adversarial search & contradiction analysis (ТЗ §18, §38–§40).

После предварительного вывода система пытается его опровергнуть:
1) отдельный противоречащий поиск (purpose=contradiction);
2) ContradictionAnalyst сравнивает claim с формулировками источников
   (hedging vs категоричность, корреляция vs причинность, animals vs humans).
Результат влияет на финальный verdict через VerdictEngine.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from pydantic import BaseModel, Field

from app.config import AppConfig, get_config
from app.core.models import Claim, Evidence, EvidenceDirection, SourceMeta, Verdict, VerdictLabel
from app.llm.base import LLMProvider, StructuredOutputError
from app.prompts import build_messages, get_system_prompt

log = logging.getLogger(__name__)


class _ContradictionResult(BaseModel):
    counter_evidence_quotes: list[str] = Field(default_factory=list)
    semantic_exaggeration: str = ""
    preliminary_supported: bool = True
    suggested_label: VerdictLabel = VerdictLabel.INSUFFICIENT_EVIDENCE


_HEDGING = ("may", "might", "could", "suggests", "indicates", "associated with",
            "linked to", "in some cases", "может", "возможно", "предполага",
            "ассоциирован", "связан с")
_STRONG = ("proven", "causes", "prevents", "cures", "eliminates", "guaranteed",
           "доказано", "вызывает", "предотвращает", "излечивает", "гарантировано",
           "полностью безопасен")
_ANIMAL = ("in mice", "in rats", "animal model", "murine", "in vitro", "на мышах",
           "на животных", "в пробирке")
_HUMAN = ("patients", "humans", "clinical trial", "пациент", "люд", "клиническ")


def detect_semantic_gap(claim_text: str, quotes: list[str]) -> str:
    """Детерминированная проверка §38/§40: источники осторожнее claim?"""
    blob = " ".join(quotes).lower()
    claim_low = claim_text.lower()
    issues: list[str] = []
    if any(h in blob for h in _HEDGING) and any(s in claim_low for s in _STRONG):
        issues.append("источники используют осторожные формулировки («may», «associated»), "
                      "тогда как материал подаёт это как доказанный факт")
    if any(a in blob for a in _ANIMAL) and any(h in claim_low for h in _HUMAN):
        issues.append("данные получены on animals/in vitro, а утверждение перенесено на людей")
    if ("associated" in blob or "коррелирует" in blob or "связан" in blob) \
            and any(c in claim_low for c in ("вызыв", "приводит", "cause")):
        issues.append("корреляция подана как причинность")
    return "; ".join(issues)


class ContradictionAnalyst:
    def __init__(self, llm: Optional[LLMProvider], cfg: Optional[AppConfig] = None) -> None:
        self.llm = llm
        self.cfg = cfg or get_config()

    def analyze(self, claim: Claim, preliminary: Verdict,
                evidence: list[Evidence], sources_by_id: dict[str, SourceMeta]) -> dict:
        """Возвращает dict с полями: exaggeration, counter_strength, revised_label_hint."""
        quotes = [e.text for e in evidence if e.quote_verified]
        det_gap = detect_semantic_gap(claim.text, quotes)
        counters = [e for e in evidence if e.direction == EvidenceDirection.CONTRADICTS]
        counter_strength = sum(e.strength * e.relevance for e in counters)

        llm_result: Optional[_ContradictionResult] = None
        if self.llm is not None and quotes:
            system = get_system_prompt("contradiction_analyzer", self.cfg.prompts_dir)
            src_lines = "\n".join(f"- [{sources_by_id[e.source_id].url}] {e.text}"
                                  for e in evidence if e.source_id in sources_by_id)[:6000]
            messages = build_messages(
                system,
                f"Claim: {claim.text}\nПредварительный вывод: {preliminary.label.value} "
                f"({preliminary.explanation[:300]})\nПроверь его устойчивость.",
                {"evidence_quotes": src_lines},
            )
            try:
                llm_result = self.llm.structured_generate(messages, _ContradictionResult)
            except Exception as e:  # graceful degradation (§50): любая ошибка LLM — fallback
                log.warning("contradiction LLM pass failed: %s — deterministic only", e)

        exaggeration = det_gap or (llm_result.semantic_exaggeration if llm_result else "")
        hint: Optional[VerdictLabel] = None
        if llm_result is not None:
            # LLM может лишь ПОВЫСИТЬ скептицизм, но не «оправдать» вопреки детерминированным данным
            if not llm_result.preliminary_supported or llm_result.counter_evidence_quotes:
                hint = llm_result.suggested_label
        return {"exaggeration": exaggeration,
                "counter_strength": round(counter_strength, 3),
                "revised_label_hint": hint,
                "llm_used": llm_result is not None}
