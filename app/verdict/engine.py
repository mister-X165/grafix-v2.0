"""VerdictEngine (ТЗ §33–§37, §69, §100–§102).

ДЕТЕРМИНИРОВАННЫЙ расчёт вердикта и confidence. LLM не голосует: она лишь
повышает скептицизм через revised_label_hint из adversarial-прохода.
Правило качества важнее количества (§100): вклад evidence =
strength * relevance * source_overall, независимые кластеры считаются полнее.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.core.models import (Claim, Evidence, EvidenceDirection, SourceMeta,
                             SourceScores, Verdict, VerdictLabel)

INSUFFICIENT_REASONS = {
    "no_sources": "не найдено релевантных источников",
    "not_retrieved": "источники не удалось получить (сеть/paywall/robots)",
    "no_verified_quotes": "в полученных источниках нет дословно проверенных фрагментов по claim",
    "unresolved_conflict": "качественные источники противоречат друг другу без разрешения",
    "not_checkable": "утверждение сформулировано так, что его нельзя проверить имеющимися данными",
}

# Порог «сильного» доказательного вклада (strength*relevance*quality):
# ниже него конфликт считается неразрешимым (§33), выше — вывод возможен (§100).
STRONG_EVIDENCE = 0.6


@dataclass
class EvidenceSummary:
    support: float = 0.0        # взвешенная сила подтверждений
    contradict: float = 0.0     # взвешенная сила опровержений
    neutral: float = 0.0
    n_support_indep: int = 0
    n_contradict_indep: int = 0
    best_source_quality: float = 0.0
    coverage: float = 0.0       # research coverage: получено источников / нужно
    agreement: float = 0.0      # 0..1 согласие между направлениями
    contradiction_ratio: float = 0.0
    max_single_weight: float = 0.0  # вклад самого сильного отдельного evidence (§100)


def summarize(evidence: list[Evidence], sources_by_id: dict[str, SourceMeta],
              scores_by_id: dict[str, SourceScores], expected_sources: int) -> EvidenceSummary:
    s = EvidenceSummary()
    clusters_seen: dict[str, set[str]] = {"SUPPORTS": set(), "CONTRADICTS": set()}
    for e in evidence:
        if not e.quote_verified or e.source_id not in sources_by_id:
            continue   # quality gate §102: непроверенная цитата/несуществующий источник
        src = sources_by_id[e.source_id]
        q = scores_by_id.get(e.source_id)
        overall = q.overall if q else 0.4
        s.best_source_quality = max(s.best_source_quality, overall)
        weight = e.strength * e.relevance * overall
        if e.direction != EvidenceDirection.NEUTRAL:
            s.max_single_weight = max(s.max_single_weight, weight)
        cluster = src.independent_cluster_key or src.url
        if e.direction == EvidenceDirection.SUPPORTS:
            s.support += weight
            clusters_seen["SUPPORTS"].add(cluster)
        elif e.direction == EvidenceDirection.CONTRADICTS:
            s.contradict += weight
            clusters_seen["CONTRADICTS"].add(cluster)
        else:
            s.neutral += weight * 0.5
    s.n_support_indep = len(clusters_seen["SUPPORTS"])
    s.n_contradict_indep = len(clusters_seen["CONTRADICTS"])
    fetched = sum(1 for x in sources_by_id.values() if x.fetched)
    s.coverage = min(1.0, fetched / max(1, expected_sources))
    tot = s.support + s.contradict
    s.agreement = abs(s.support - s.contradict) / tot if tot > 0 else 0.0
    s.contradiction_ratio = (s.contradict / tot) if tot > 0 else 0.0
    return s


def compute_verdict(claim: Claim, summary: EvidenceSummary, *,
                    exaggeration: str = "", insufficient_flag: bool = False,
                    llm_skeptical_hint: VerdictLabel | None = None,
                    min_sources: int = 3) -> Verdict:
    """Чистая функция: evidence → verdict. Никаких вызовов LLM здесь."""
    d = {"support": round(summary.support, 3), "contradict": round(summary.contradict, 3),
         "coverage": round(summary.coverage, 3), "agreement": round(summary.agreement, 3),
         "best_source_quality": round(summary.best_source_quality, 3)}

    total = summary.support + summary.contradict

    # --- INSUFFICIENT_EVIDENCE только по объективным условиям (§33) ----------
    # Правило §100: если хотя бы с одной стороны есть СИЛЬНОЕ качественное
    # доказательство (weight >= STRONG_EVIDENCE), вывод делать можно —
    # INSUFFICIENT_EVIDENCE не используется для маскировки реального конфликта.
    reason = ""
    if insufficient_flag:
        reason = INSUFFICIENT_REASONS["not_checkable"]
    elif total < 1e-6:
        reason = INSUFFICIENT_REASONS["no_verified_quotes"]
    elif summary.coverage < 0.34 and summary.best_source_quality < 0.5:
        reason = INSUFFICIENT_REASONS["not_retrieved"]
    elif summary.support > 0 and summary.contradict > 0 and \
            summary.agreement < 0.25 and summary.best_source_quality >= 0.6:
        # «Сильный» вклад считаем по максимуму отдельного evidence, но порог
        # нормирован на единичное эталонное доказательство (1.0*1.0*0.6):
        # INSUFFICIENT_EVIDENCE — только когда НИ ОДНА сторона не имеет
        # сильного качественного доказательства (§33, §100).
        strong = max(summary.support, summary.contradict,
                    summary.max_single_weight * 3.0) >= STRONG_EVIDENCE
        if not strong:
            reason = INSUFFICIENT_REASONS["unresolved_conflict"]
    if reason:
        conf = _confidence(summary, base=0.35)
        return Verdict(claim_id=claim.id, label=VerdictLabel.INSUFFICIENT_EVIDENCE,
                       confidence=round(conf, 3),
                       explanation=f"Недостаточно доказательств: {reason}.",
                       insufficient_reason=reason, details=d)

    net = summary.support - summary.contradict
    strong_q = summary.best_source_quality >= 0.65

    # --- детерминированное правило -------------------------------------------
    if net > 0:
        if summary.contradiction_ratio < 0.15 and summary.n_support_indep >= 2 and strong_q \
                and not exaggeration and summary.support >= 0.8:
            label = VerdictLabel.VERIFIED
        elif summary.contradiction_ratio < 0.3 and not exaggeration:
            label = VerdictLabel.LIKELY_TRUE
        elif exaggeration:
            label = VerdictLabel.MISLEADING
        else:
            label = VerdictLabel.PARTIALLY_TRUE
    elif net < 0:
        if summary.contradiction_ratio > 0.7 and summary.n_contradict_indep >= 2 and strong_q:
            label = VerdictLabel.FALSE
        elif exaggeration:
            label = VerdictLabel.MISLEADING
        else:
            label = VerdictLabel.LIKELY_FALSE
    else:
        label = VerdictLabel.PARTIALLY_TRUE

    # --- adversarial revision: LLM может только «снизить уверенность» (§18, §36)
    if llm_skeptical_hint is not None and llm_skeptical_hint != label:
        rank = {VerdictLabel.VERIFIED: 0, VerdictLabel.LIKELY_TRUE: 1,
                VerdictLabel.PARTIALLY_TRUE: 2, VerdictLabel.MISLEADING: 2,
                VerdictLabel.LIKELY_FALSE: 3, VerdictLabel.FALSE: 4,
                VerdictLabel.INSUFFICIENT_EVIDENCE: 2}
        if rank[llm_skeptical_hint] > rank[label]:
            label = llm_skeptical_hint
        elif label in (VerdictLabel.VERIFIED, VerdictLabel.LIKELY_TRUE) and \
                llm_skeptical_hint == VerdictLabel.PARTIALLY_TRUE:
            label = VerdictLabel.PARTIALLY_TRUE

    if exaggeration and label in (VerdictLabel.VERIFIED, VerdictLabel.LIKELY_TRUE):
        label = VerdictLabel.MISLEADING   # семантическое преувеличение (§40, §99)

    conf = _confidence(summary, base=_LABEL_BASE_CONF[label])
    return Verdict(claim_id=claim.id, label=label, confidence=round(conf, 3),
                   explanation=_explain(label, summary, exaggeration), details=d)


_LABEL_BASE_CONF = {VerdictLabel.VERIFIED: 0.9, VerdictLabel.LIKELY_TRUE: 0.75,
                    VerdictLabel.PARTIALLY_TRUE: 0.6, VerdictLabel.MISLEADING: 0.7,
                    VerdictLabel.LIKELY_FALSE: 0.75, VerdictLabel.FALSE: 0.9,
                    VerdictLabel.INSUFFICIENT_EVIDENCE: 0.4}


def _confidence(s: EvidenceSummary, base: float) -> float:
    """Confidence из agreement/quality/independence/quantity/contradiction/coverage (§37).
    Сигнал LLM — не единственный и вообще не используется напрямую."""
    indep = min(1.0, (s.n_support_indep + s.n_contradict_indep) / 3.0)
    qty = min(1.0, math.log1p(s.support + s.contradict) / math.log(4))
    contra_penalty = 1.0 - 0.5 * min(1.0, s.contradiction_ratio)
    cov = 0.5 + 0.5 * s.coverage
    c = (base * (0.30 * s.agreement + 0.25 * s.best_source_quality + 0.15 * indep +
                 0.10 * qty + 0.10 * cov) * (1.0 + 0.15 * contra_penalty))
    return max(0.05, min(0.98, c))


def _explain(label: VerdictLabel, s: EvidenceSummary, exaggeration: str) -> str:
    ru = {
        VerdictLabel.VERIFIED: "Подтверждается несколькими независимыми качественными источниками.",
        VerdictLabel.LIKELY_TRUE: "Большинство качественных источников подтверждают утверждение.",
        VerdictLabel.PARTIALLY_TRUE: "Часть аспектов подтверждена, часть — нет или недоказуема.",
        VerdictLabel.MISLEADING: "Формально опирается на данные, но подаёт их искажённо.",
        VerdictLabel.LIKELY_FALSE: "Качественные источники преимущественно противоречат утверждению.",
        VerdictLabel.FALSE: "Противоречится несколькими независимыми качественными источниками.",
        VerdictLabel.INSUFFICIENT_EVIDENCE: "Недостаточно доказательств для вывода.",
    }[label]
    extra = f" Выявленное преувеличение: {exaggeration}." if exaggeration else ""
    return (f"{ru} Взвешенная поддержка {s.support:.2f}, противодействие {s.contradict:.2f}; "
            f"независимых подтверждающих кластеров: {s.n_support_indep}, опровергающих: "
            f"{s.n_contradict_indep}.{extra}")


def aggregate_article_verdict(claims: list[Claim], verdicts: list[Verdict]) -> tuple[VerdictLabel, float]:
    """Итог по статье с весами по importance (§69), а не простое среднее."""
    if not verdicts:
        return VerdictLabel.INSUFFICIENT_EVIDENCE, 0.0
    score_map = {
        VerdictLabel.VERIFIED: 1.0, VerdictLabel.LIKELY_TRUE: 0.75,
        VerdictLabel.PARTIALLY_TRUE: 0.35, VerdictLabel.MISLEADING: -0.45,
        VerdictLabel.LIKELY_FALSE: -0.8, VerdictLabel.FALSE: -1.0,
        VerdictLabel.INSUFFICIENT_EVIDENCE: 0.0,
    }
    imp = {c.id: c.importance for c in claims}
    num = den = 0.0
    wsum = 0.0
    for v in verdicts:
        w = imp.get(v.claim_id, 0.5)
        num += w * score_map[v.label]
        den += w
        wsum += w * v.confidence
    val = num / den if den else 0.0
    if val >= 0.7:
        label = VerdictLabel.VERIFIED
    elif val >= 0.35:
        label = VerdictLabel.LIKELY_TRUE
    elif val >= 0.1:
        label = VerdictLabel.PARTIALLY_TRUE
    elif val >= -0.25:
        label = VerdictLabel.MISLEADING
    elif val >= -0.7:
        label = VerdictLabel.LIKELY_FALSE
    else:
        label = VerdictLabel.FALSE
    avg_conf = wsum / den if den else 0.0
    return label, round(min(0.95, avg_conf), 3)
