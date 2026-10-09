"""ReportGenerator + Quality Gate (ТЗ §27, §42–§45, §60, §97–§98, §102).

Quality gate перед финальным отчётом: claim/evidence/source/quote/citation —
всё проверяется детерминированно. LLM пишет только человеческое объяснение
по уже посчитанным данным; ссылки в отчёте — только из реально полученных
источников.
"""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field

from app.config import AppConfig, get_config
from app.core.models import (AnalysisMode, Claim, Evidence, ModelInfo, Report,
                             SourceMeta, SourceScores, Verdict, VerdictLabel)
from app.evidence.extractor import quote_in_text
from app.llm.base import LLMProvider, StructuredOutputError
from app.prompts import build_messages, get_system_prompt
from app.verdict.engine import aggregate_article_verdict

log = logging.getLogger(__name__)

DISCLAIMER = ("Система проверяет достоверность информации и НЕ является "
              "медицинским диагнозом или рекомендацией по лечению. "
              "По вопросам здоровья обращайтесь к врачу.")


class _Narrative(BaseModel):
    summary: str = ""
    key_problem: str = ""


def quality_gate(claims: list[Claim], evidence: list[Evidence],
                 sources_by_id: dict[str, SourceMeta],
                 verdicts: list[Verdict]) -> tuple[list[Verdict], list[str]]:
    """§102: вернуть допустимые вердикты + список нарушений."""
    ev_by_id = {e.id: e for e in evidence}
    ok: list[Verdict] = []
    problems: list[str] = []
    for v in verdicts:
        claim = next((c for c in claims if c.id == v.claim_id), None)
        if claim is None:
            problems.append(f"{v.claim_id}: claim отсутствует")
            continue
        kept_ids_s, kept_ids_c = [], []
        valid = True
        for eid in v.supporting_evidence_ids + v.contradicting_evidence_ids:
            e = ev_by_id.get(eid)
            if e is None or e.source_id not in sources_by_id:
                problems.append(f"{eid}: evidence ссылается на несуществующий источник")
                valid = False
                continue
            src = sources_by_id[e.source_id]
            if not src.fetched:
                problems.append(f"{src.url[:60]}: источник не получен, но использован как доказательство")
                valid = False
                continue
            if not e.quote_verified or not quote_in_text(e.text, src.body_text):
                problems.append(f"{eid}: цитата не подтверждена в тексте источника")
                valid = False
                continue
            if eid in v.supporting_evidence_ids:
                kept_ids_s.append(eid)
            else:
                kept_ids_c.append(eid)
        if not valid and v.label != VerdictLabel.INSUFFICIENT_EVIDENCE:
            # не придумываем результат: понижаем до INSUFFICIENT с причиной
            v = v.model_copy(update={
                "label": VerdictLabel.INSUFFICIENT_EVIDENCE,
                "insufficient_reason": "quality gate: часть доказательств не прошла проверку",
                "confidence": min(v.confidence, 0.35),
                "supporting_evidence_ids": kept_ids_s,
                "contradicting_evidence_ids": kept_ids_c,
            })
        ok.append(v)
    return ok, problems


class ReportGenerator:
    def __init__(self, llm: Optional[LLMProvider], cfg: Optional[AppConfig] = None) -> None:
        self.llm = llm
        self.cfg = cfg or get_config()

    def build(self, *, project_id: str, url: str, mode: AnalysisMode,
              claims: list[Claim], verdicts: list[Verdict],
              sources: list[SourceMeta], scores: list[SourceScores],
              evidence: list[Evidence], limitations: list[str],
              model_info: ModelInfo) -> Report:
        """Синхронная версия (CLI/тесты): вне event loop."""
        return self._build_core(project_id=project_id, url=url, mode=mode,
                                claims=claims, verdicts=verdicts, sources=sources,
                                scores=scores, evidence=evidence,
                                limitations=limitations, model_info=model_info,
                                narrative=None)

    async def build_async(self, *, project_id: str, url: str, mode: AnalysisMode,
                          claims: list[Claim], verdicts: list[Verdict],
                          sources: list[SourceMeta], scores: list[SourceScores],
                          evidence: list[Evidence], limitations: list[str],
                          model_info: ModelInfo) -> Report:
        """Асинхронная версия для pipeline: LLM-нарратив вне event loop (§ стабильность)."""
        report = self._build_core(project_id=project_id, url=url, mode=mode,
                                  claims=claims, verdicts=verdicts, sources=sources,
                                  scores=scores, evidence=evidence,
                                  limitations=limitations, model_info=model_info,
                                  narrative=None, skip_narrative=True)
        report.summary = await self._narrative_async(report)
        return report

    def _build_core(self, *, project_id: str, url: str, mode: AnalysisMode,
                    claims: list[Claim], verdicts: list[Verdict],
                    sources: list[SourceMeta], scores: list[SourceScores],
                    evidence: list[Evidence], limitations: list[str],
                    model_info: ModelInfo, narrative: Optional[str] = None,
                    skip_narrative: bool = False) -> Report:
        sources_by_id = {s.id: s for s in sources}
        verdicts, gate_problems = quality_gate(claims, evidence, sources_by_id, verdicts)
        limitations += [f"Quality gate: {p}" for p in gate_problems]

        overall, conf = aggregate_article_verdict(claims, verdicts)
        report = Report(project_id=project_id, input_url=url, mode=mode,
                        overall_verdict=overall, overall_confidence=conf,
                        claims=claims, verdicts=verdicts, sources=sources,
                        scores=scores, evidence=evidence, limitations=limitations,
                        methodology=self._methodology(mode), model_info=model_info)
        report.counts = report.verdict_counts()
        if not skip_narrative:
            report.summary = narrative if narrative is not None else self._narrative(report)
        return report

    def _methodology(self, mode: AnalysisMode) -> str:
        m = ("Контролируемый режим: поиск только по доверенным категориям источников."
             if mode == AnalysisMode.CONTROLLED else
             "Исследовательский режим: расширенный поиск (PubMed, Europe PMC, Crossref, веб), "
             "оценка качества каждого источника, обязательный adversarial-поиск опровержений.")
        return (m + " Вердикты рассчитаны детерминированным движком по взвешенным "
                "подтверждённым цитатам; LLM использовалась для извлечения claims/evidence "
                "и формулирования объяснений, но не как источник фактов. " + DISCLAIMER)

    def _narrative(self, report: Report) -> str:
        """Синхронная версия (CLI/тесты)."""
        messages = self._narrative_messages(report)
        if messages is not None:
            try:
                res = self.llm.structured_generate(messages, _Narrative)
                clean = self._sanitize_narrative(res, report)
                if clean is not None:
                    return clean
            except Exception as e:  # graceful degradation (§50): любая ошибка LLM — fallback
                log.warning("narrative LLM failed: %s — deterministic summary", e)
        return self._deterministic_summary(report)

    async def _narrative_async(self, report: Report) -> str:
        """Асинхронная версия для pipeline: LLM-вызов вне event loop (стабильность)."""
        messages = self._narrative_messages(report)
        if messages is not None:
            try:
                res = await self.llm.structured_generate_async(messages, _Narrative)
                clean = self._sanitize_narrative(res, report)
                if clean is not None:
                    return clean
            except Exception as e:  # graceful degradation (§50): любая ошибка LLM — fallback
                log.warning("narrative LLM failed: %s — deterministic summary", e)
        return self._deterministic_summary(report)

    def _narrative_messages(self, report: Report):
        if self.llm is None:
            return None
        system = get_system_prompt("report_generator", self.cfg.prompts_dir)
        return build_messages(
            system,
            "Напиши краткое объяснение результата проверки на русском.",
            {"analysis_data": self._facts_block(report)},
        )

    @staticmethod
    def _sanitize_narrative(res: _Narrative, report: Report) -> Optional[str]:
        if not res.summary.strip():
            return None
        import re as _re
        urls = {s.url for s in report.sources}
        # анти-галлюцинация: если LLM вставила URL — оставляем только реальные
        found = _re.findall(r"https?://\S+", res.summary)
        clean = " ".join(u for u in found if u in urls)
        if found and not clean:
            log.warning("report narrative contained unknown URLs — dropped")
            return None
        extra = f" Главная проблема: {res.key_problem}" if res.key_problem else ""
        return res.summary.strip() + extra

    @staticmethod
    def _facts_block(report: Report) -> str:
        lines = [f"Общий вердикт: {report.overall_verdict.value}",
                 f"Confidence: {report.overall_confidence:.2f}"]
        for v in report.verdicts[:10]:
            lines.append(f"- {v.claim_id}: {v.label.value} ({v.confidence:.2f}) {v.explanation[:160]}")
        for s in report.sources[:15]:
            q = next((x.overall for x in report.scores if x.source_id == s.id), None)
            lines.append(f"* источник: {s.url[:100]} tier={s.tier} quality={q}")
        return "\n".join(lines)[:8000]

    @staticmethod
    def _deterministic_summary(report: Report) -> str:
        c = report.counts
        total = len(report.verdicts)
        parts = []
        order = [("VERIFIED", "подтверждено"), ("LIKELY_TRUE", "скорее всего верно"),
                 ("PARTIALLY_TRUE", "частично верно"), ("MISLEADING", "вводит в заблуждение"),
                 ("LIKELY_FALSE", "скорее всего неверно"), ("FALSE", "неверно"),
                 ("INSUFFICIENT_EVIDENCE", "недостаточно данных")]
        for k, ru in order:
            if c.get(k):
                parts.append(f"{c[k]} — {ru}")
        head = f"Проверено утверждений: {total}. " + "; ".join(parts) + "."
        worst = max((v for v in report.verdicts
                     if v.label in (VerdictLabel.FALSE, VerdictLabel.LIKELY_FALSE,
                                    VerdictLabel.MISLEADING)),
                    key=lambda v: v.confidence, default=None)
        if worst:
            head += f" Наиболее проблематичное утверждение: «{next((cl.text for cl in report.claims if cl.id == worst.claim_id), '')[:120]}»."
        return head
