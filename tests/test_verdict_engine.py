"""Unit-тесты verdict engine: детерминированность, INSUFFICIENT_EVIDENCE, confidence,
агрегация по importance (ТЗ §33–§37, §69, §100)."""

from __future__ import annotations

from app.core.models import (Claim, Evidence, EvidenceDirection, SourceMeta,
                             SourceScores, VerdictLabel)
from app.verdict.engine import (EvidenceSummary, aggregate_article_verdict,
                                compute_verdict, summarize)


def _src(url="https://who.int/x", tier=4, fetched=True, cluster=""):
    return SourceMeta(url=url, tier=tier, fetched=fetched,
                      independent_cluster_key=cluster or url)


def _score(sid, overall=0.8):
    return SourceScores(source_id=sid, authority=overall, methodology=overall,
                        recency=overall, relevance=overall,
                        independence=overall, transparency=overall, overall=overall)


def _ev(claim_id, src_id, direction=EvidenceDirection.SUPPORTS, strength=0.8,
        relevance=0.8, verified=True):
    return Evidence(claim_id=claim_id, source_id=src_id, text="q",
                    direction=direction, strength=strength, relevance=relevance,
                    quote_verified=verified)


def test_no_evidence_gives_insufficient_with_reason():
    claim = Claim(text="Препарат X лечит болезнь Y навсегда.")
    s = EvidenceSummary()
    v = compute_verdict(claim, s)
    assert v.label == VerdictLabel.INSUFFICIENT_EVIDENCE
    assert v.insufficient_reason, "причина обязательна (§33)"


def test_unverified_quotes_do_not_count():
    """Цитата, не найденная в реально полученном тексте, — не доказательство (§26–§27)."""
    claim = Claim(text="X снижает риск Y.")
    src = _src()
    ev = [_ev(claim.id, src.id, verified=False)]
    s = summarize(ev, {src.id: src}, {src.id: _score(src.id)}, expected_sources=3)
    assert s.support == 0.0
    v = compute_verdict(claim, s)
    assert v.label == VerdictLabel.INSUFFICIENT_EVIDENCE


def test_two_independent_high_quality_supports_verified():
    claim = Claim(text="Вакцина снижает риск госпитализации.", importance=0.9)
    s1, s2 = _src("https://who.int/a", tier=1), _src("https://cochrane.org/b", tier=1)
    ev = [_ev(claim.id, s1.id), _ev(claim.id, s2.id)]
    scores = {s.id: _score(s.id, 0.85) for s in (s1, s2)}
    summ = summarize(ev, {s.id: s for s in (s1, s2)}, scores, expected_sources=2)
    v = compute_verdict(claim, summ, min_sources=2)
    assert v.label == VerdictLabel.VERIFIED
    assert 0 < v.confidence <= 1.0


def test_contradiction_downgrades():
    claim = Claim(text="Препарат X полностью безопасен.")
    s1, s2 = _src("https://a.org/1"), _src("https://b.org/2")
    ev = [_ev(claim.id, s1.id, strength=0.6),
          _ev(claim.id, s2.id, direction=EvidenceDirection.CONTRADICTS, strength=0.9)]
    smap = {s.id: s for s in (s1, s2)}
    scores = {s.id: _score(s.id, 0.8) for s in (s1, s2)}
    summ = summarize(ev, smap, scores, expected_sources=2)
    v = compute_verdict(claim, summ, min_sources=2)
    assert v.label in (VerdictLabel.LIKELY_FALSE, VerdictLabel.PARTIALLY_TRUE,
                       VerdictLabel.FALSE)


def test_exaggeration_forces_misleading():
    claim = Claim(text="Учёные доказали, что X вызывает Y.")
    s1, s2 = _src("https://a.org/1"), _src("https://b.org/2")
    ev = [_ev(claim.id, s1.id), _ev(claim.id, s2.id)]
    smap = {s.id: s for s in (s1, s2)}
    scores = {s.id: _score(s.id, 0.8) for s in (s1, s2)}
    summ = summarize(ev, smap, scores, expected_sources=2)
    plain = compute_verdict(claim, summ, min_sources=2)
    exagger = compute_verdict(claim, summ, exaggeration="may→causes", min_sources=2)
    assert plain.label in (VerdictLabel.VERIFIED, VerdictLabel.LIKELY_TRUE)
    assert exagger.label == VerdictLabel.MISLEADING   # §40, §99


def test_dependent_reposts_are_not_multiple_confirmations():
    """20 копий одной публикации ≠ 20 подтверждений (§24)."""
    claim = Claim(text="X снижает риск Y.")
    base = "https://blog.example/original"
    srcs = [_src(f"https://mirror{i}.com/repost", cluster=base) for i in range(5)]
    ev = [_ev(claim.id, s.id) for s in srcs]
    smap = {s.id: s for s in srcs}
    scores = {s.id: _score(s.id, 0.5) for s in srcs}
    summ = summarize(ev, smap, scores, expected_sources=3)
    assert summ.n_support_indep == 1


def test_confidence_is_computed_not_taken_from_llm():
    claim = Claim(text="X влияет на Y.")
    s1 = _src()
    ev = [_ev(claim.id, s1.id, strength=0.3, relevance=0.3)]
    summ = summarize(ev, {s1.id: s1}, {s1.id: _score(s1.id, 0.5)}, expected_sources=3)
    v = compute_verdict(claim, summ)
    assert v.confidence <= 0.75  # слабое evidence → низкая уверенность


def test_aggregate_weights_by_importance():
    hi = Claim(text="Препарат безопасен.", importance=1.0)
    lo = Claim(text="Небо голубое сегодня.", importance=0.1)
    vs = []
    from app.core.models import Verdict
    vs.append(Verdict(claim_id=hi.id, label=VerdictLabel.FALSE, confidence=0.9))
    vs.append(Verdict(claim_id=lo.id, label=VerdictLabel.VERIFIED, confidence=0.9))
    label, conf = aggregate_article_verdict([hi, lo], vs)
    assert label in (VerdictLabel.LIKELY_FALSE, VerdictLabel.FALSE,
                     VerdictLabel.MISLEADING)   # ложный важный claim доминирует (§69)
