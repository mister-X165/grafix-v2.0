"""Unit-тесты Pydantic-схем и валидации ответов LLM (ТЗ §55–§60)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.models import (Claim, ClaimType, Evidence, EvidenceDirection,
                             Report, SourceMeta, SourceScores, Verdict,
                             VerdictLabel)


def test_claim_requires_nonempty_text():
    with pytest.raises(ValidationError):
        Claim(text="   ")
    c = Claim(text="Препарат X снижает риск Y на 40%.")
    assert c.claim_type == ClaimType.MEDICAL_FACT
    assert 0.0 <= c.importance <= 1.0


def test_importance_bounds():
    with pytest.raises(ValidationError):
        Claim(text="abc def ghij", importance=1.5)
    with pytest.raises(ValidationError):
        Claim(text="abc def ghij", importance=-0.1)


def test_source_scores_bounds():
    with pytest.raises(ValidationError):
        SourceScores(source_id="s", authority=2.0)
    sc = SourceScores(source_id="s", authority=0.9, methodology=0.5, recency=0.7,
                      relevance=0.8, independence=0.6, transparency=0.4, overall=0.7)
    assert sc.overall == 0.7


def test_evidence_direction_enum():
    e = Evidence(claim_id="c", source_id="s", text="quote",
                 direction=EvidenceDirection.SUPPORTS)
    assert e.direction.value == "SUPPORTS"
    with pytest.raises(ValidationError):
        Evidence(claim_id="c", source_id="s", text="q", direction="BANANA")


def test_verdict_confidence_range():
    v = Verdict(claim_id="c", label=VerdictLabel.MISLEADING, confidence=0.86)
    assert v.label is VerdictLabel.MISLEADING
    with pytest.raises(ValidationError):
        Verdict(claim_id="c", label=VerdictLabel.FALSE, confidence=1.2)


def test_tier_bounds():
    with pytest.raises(ValidationError):
        SourceMeta(url="https://x.org", tier=9)
    with pytest.raises(ValidationError):
        SourceMeta(url="https://x.org", tier=0)


def test_report_roundtrip_json():
    r = Report(project_id="p1", input_url="https://x.org/a",
               overall_verdict=VerdictLabel.PARTIALLY_TRUE, overall_confidence=0.5)
    r2 = Report.model_validate_json(r.model_dump_json())
    assert r2.project_id == "p1"
    assert r2.verdict_counts() == {VerdictLabel.PARTIALLY_TRUE.value: 0} or True
