"""Unit-тесты claim engine: парсинг, декомпозиция, эвристики (ТЗ §12–§15, §39–§40)."""

from __future__ import annotations

import json

from app.claims.extractor import ClaimExtractor, heuristic_claims, heuristic_flags, normalize
from app.core.models import ClaimType, Document


def _doc(text: str) -> Document:
    return Document(url="https://example.com/a", title="T", text=text)


def test_heuristic_claims_extracts_medical_sentences():
    doc = _doc(
        "Препарат X снижает риск заболевания Y на 40%. "
        "Это исследование проводилось в лабораторных условиях. "
        "Новое лечение полностью безопасно для пациентов с заболеванием Z.")
    claims = heuristic_claims(doc)
    assert len(claims) >= 2
    types = {c.claim_type for c in claims}
    assert ClaimType.STATISTICAL_CLAIM in types or ClaimType.SAFETY_CLAIM in types
    for c in claims:
        assert c.check_required and c.importance >= 0.7


def test_heuristic_flags_relative_risk_and_extrapolation():
    flags, hedging = heuristic_flags("Препарат снижает риск на 40% по сравнению с плацебо")
    assert "relative_vs_absolute_risk" in flags
    flags2, _ = heuristic_flags("В исследовании на мышах препарат лечит болезнь у людей")
    assert "extrapolation" in flags2


def test_normalize_lowercases_and_collapses_ws():
    assert normalize("  Привет   МИР ") == "привет мир"


def test_llm_claim_not_in_document_kept_but_deduped(tmp_cfg):
    """Ответ LLM валидируется; дубликаты отбрасываются (§56)."""
    from tests.conftest import FakeLLM
    doc = _doc("Препарат X снижает риск заболевания Y на 40%. Текст статьи продолжается тут.")
    payload = {"claims": [
        {"text": "Препарат X снижает риск заболевания Y на 40%.",
         "claim_type": "STATISTICAL_CLAIM", "importance": 0.9},
        {"text": "Препарат X снижает риск заболевания Y на 40%.",   # дубль
         "claim_type": "STATISTICAL_CLAIM", "importance": 0.9},
        {"text": "мелкий"},                                        # слишком короткий
    ]}
    llm = FakeLLM(json.dumps(payload))
    claims = ClaimExtractor(llm, tmp_cfg).extract(doc)
    assert len(claims) == 1
    assert claims[0].claim_type == ClaimType.STATISTICAL_CLAIM


def test_llm_invalid_json_falls_back_to_heuristic(tmp_cfg):
    from tests.conftest import FakeLLM

    class BrokenLLM(FakeLLM):
        def structured_generate(self, messages, schema, **kw):
            raise Exception("bad json")

    doc = _doc("Препарат X полностью безопасен и предотвращает рецидивы заболевания Y у пациентов.")
    claims = ClaimExtractor(BrokenLLM("{}"), tmp_cfg).extract(doc)
    assert claims, "должен сработать детерминированный fallback"
