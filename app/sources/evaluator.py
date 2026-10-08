"""SourceEvaluator (ТЗ §21–§24): шесть параметров качества + итог.

Детерминированная оценка по метаданным источника; LLM не участвует (чтобы
нельзя было «договориться» с ней через контент страницы). Все исходные
параметры сохраняются отдельно от итога (§22).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from urllib.parse import urlparse

from app.config import AppConfig, get_config
from app.core.models import AnalysisMode, SourceMeta, SourceScores

_TIER_WEIGHT = {1: 0.95, 2: 0.85, 3: 0.7, 4: 0.9, 5: 0.6, 6: 0.65, 7: 0.4, 8: 0.15}

_STUDY_TYPES = {
    "meta-analysis": ("meta-analys", "metaanalys", "meta analysis"),
    "systematic_review": ("systematic review",),
    "rct": ("randomized controlled trial", "randomised controlled trial", "rct"),
    "cohort": ("cohort study", "prospective study"),
    "case_control": ("case-control",),
    "animal": ("in mice", "in rats", "animal model", "в исследовании на животн"),
    "in_vitro": ("in vitro",),
    "editorial": ("editorial", "commentary", "letter"),
}


def classify_source(s: SourceMeta, cfg: AppConfig) -> None:
    """Заполнить source_type/tier/study_type/commercial_interest по URL и тексту."""
    host = (urlparse(s.url).hostname or "").lower().removeprefix("www.")
    trusted = set()
    for cat, domains in cfg.sources.trusted_domains.items():
        for d in domains:
            if host == d or host.endswith("." + d):
                s.trusted_category = cat
        trusted.update(domains)
    is_trusted = any(host == d or host.endswith("." + d) for d in trusted)

    # tier по конфигурации
    tier = None
    for t, doms in cfg.sources.tier_domains.items():
        if any(host == d or host.endswith("." + d) for d in doms):
            tier = t
            break
    if tier is None:
        if any(host == d or host.endswith("." + d) for d in cfg.sources.news_domains):
            tier = 7
        elif not is_trusted:
            blob = (s.url.lower() + " " + (s.body_text or "")[:1500].lower())
            signals = [w for w in cfg.sources.low_quality_signals if w in blob]
            tier = 8 if signals else 6 if _looks_scholarly(blob) else 7
    s.tier = tier or 8

    if s.source_type in ("journal_article", "systematic_review"):
        stype = s.source_type
    else:
        stype = "web_page"
    if s.trusted_category in ("who", "cdc", "fda", "ema", "nice", "nih", "nhs", "gov_ru"):
        s.source_type = "agency_page" if stype == "web_page" else stype
    elif s.doi or s.pmid:
        s.source_type = stype if stype != "web_page" else "journal_article"
    else:
        s.source_type = stype

    blob = ((s.title + " " + (s.body_text or "")[:6000]) + " " + s.snippet_hint()).lower() \
        if hasattr(s, "snippet_hint") else (s.title + " " + (s.body_text or "")[:6000]).lower()
    if not s.study_type:
        for name, pats in _STUDY_TYPES.items():
            if any(p in blob for p in pats):
                s.study_type = name
                break
    if s.source_type == "systematic_review":
        s.study_type = s.study_type or "systematic_review"

    commercial = bool(s.trusted_category == "" and
                      any(w in blob for w in ("купить", "заказать", "скидка", "buy now",
                                              "order now", "discount", "affiliate")))
    s.commercial_interest = commercial or s.content_nature.value in (
        "ADVERTISEMENT", "AFFILIATE_CONTENT", "SPONSORED_CONTENT", "NATIVE_ADVERTISEMENT")


def _looks_scholarly(blob: str) -> bool:
    return any(w in blob for w in ("doi.org", "pubmed", "journal", "et al", "pmid",
                                   "study", "trial", "research", "исследован", "журнал"))


def recency_score(date_str: str, now_year: int | None = None) -> float:
    m = re.search(r"(19|20)\d{2}", date_str or "")
    if not m:
        return 0.5   # неизвестно — нейтрально
    year = int(m.group(0))
    cur = now_year or datetime.now(timezone.utc).year
    age = max(0, cur - year)
    if age <= 2:
        return 0.95
    if age <= 5:
        return 0.8
    if age <= 10:
        return 0.6
    if age <= 20:
        return 0.4
    return 0.25


_METH_SCORE = {"meta-analysis": 0.95, "systematic_review": 0.9, "rct": 0.8,
               "cohort": 0.6, "case_control": 0.55, "animal": 0.35, "in_vitro": 0.3,
               "editorial": 0.25, "": 0.45}


def evaluate(s: SourceMeta, *, claim_relevance: float = 0.5, mode: AnalysisMode = AnalysisMode.RESEARCH,
             independent_rank: int = 0, cfg: AppConfig | None = None) -> SourceScores:
    """Итоговые 6+1 оценок. independent_rank: 0 = первый в своём независимом кластере."""
    cfg = cfg or get_config()
    authority = _TIER_WEIGHT.get(s.tier, 0.3)
    if s.trusted_category:
        authority = min(1.0, authority + 0.1)
    methodology = _METH_SCORE.get(s.study_type, 0.45)
    if s.source_type == "agency_page":
        methodology = max(methodology, 0.75)      # guidelines/regulators
    recency = recency_score(s.publication_date)
    relevance = max(0.0, min(1.0, claim_relevance))
    independence = 1.0 if independent_rank == 0 else max(0.15, 1.0 - 0.45 * independent_rank)
    transparency = 0.5
    if s.doi:
        transparency += 0.15
    if s.pmid:
        transparency += 0.1
    if s.author:
        transparency += 0.1
    if s.organization or s.publication_date:
        transparency += 0.05
    if s.funding:
        transparency += 0.05
    if s.conflicts_of_interest:
        transparency -= 0.1
    if s.commercial_interest:
        transparency -= 0.15
    transparency = max(0.05, min(1.0, transparency))

    overall = round(
        0.30 * authority + 0.22 * methodology + 0.12 * recency +
        0.20 * relevance + 0.10 * independence + 0.06 * transparency, 3)
    if mode == AnalysisMode.CONTROLLED and not s.trusted_category:
        overall = round(min(overall, 0.35), 3)   # вне белого списка — почти не считается
    scores = SourceScores(source_id=s.id, authority=round(authority, 3),
                          methodology=round(methodology, 3), recency=round(recency, 3),
                          relevance=round(relevance, 3), independence=round(independence, 3),
                          transparency=round(transparency, 3), overall=overall)
    s.metadata["scores"] = scores.model_dump()
    return scores
