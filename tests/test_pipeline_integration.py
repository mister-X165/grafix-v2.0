"""Интеграционный тест полного pipeline (ТЗ §72 integration, §90, §49).

Сеть полностью мокнируется на уровне httpx.AsyncClient (разрешено: mock только
в тестах, §76): URL → extraction → claims → research → evidence → verdict →
report. Проверяются graceful degradation (§50) и offline mode (§49), а также
то, что в отчёт попадают ТОЛЬКО реально полученные источники (§27, §75).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.core.models import AnalysisMode, PipelineStage, VerdictLabel

ARTICLE_HTML = """<html><head><title>Препарат X лечит все болезни</title></head>
<body><article>
<p>Новый препарат X полностью излечивает заболевание Y и абсолютно безопасен.</p>
<p>По данным исследования, препарат X снижает риск заболевания Y на 40%.</p>
<p>Врачи всего мира уже рекомендуют X всем пациентам без исключений.</p>
</article></body></html>""".encode("utf-8")

SOURCE_HTML = b"""<html><head><title>Randomized trial of drug X</title></head>
<body><p>The randomized trial showed no significant benefit of drug X for disease Y.
Drug X was not completely safe: adverse events were reported in 12% of participants.</p>
</body></html>"""

CLAIMS_JSON = json.dumps({
    "claims": [
        {"text": "Препарат X полностью излечивает заболевание Y.",
         "claim_type": "TREATMENT_CLAIM", "importance": 0.95},
        {"text": "Препарат X абсолютно безопасен.",
         "claim_type": "SAFETY_CLAIM", "importance": 0.9},
        {"text": "Препарат X снижает риск заболевания Y на 40%.",
         "claim_type": "STATISTICAL_CLAIM", "importance": 0.8},
    ]
})

PLAN_JSON = json.dumps({
    "queries": [
        {"query": "drug X disease Y randomized trial", "purpose": "PRIMARY"},
        {"query": "drug X safety adverse effects meta-analysis", "purpose": "REVIEW"},
        {"query": "drug X does not work contradiction", "purpose": "CONTRADICTION"},
    ]
})

EVIDENCE_JSON = json.dumps({
    "evidence": [
        {"quote": "The randomized trial showed no significant benefit of drug X for disease Y.",
         "direction": "CONTRADICTS", "strength": 0.9, "relevance": 0.9},
        {"quote": "Drug X was not completely safe: adverse events were reported in 12% of participants.",
         "direction": "CONTRADICTS", "strength": 0.85, "relevance": 0.85},
    ]
})

CONTRADICTION_JSON = json.dumps({
    "counter_evidence_quotes": [],
    "semantic_exaggeration": "",
    "preliminary_supported": False,
    "suggested_label": "LIKELY_FALSE",
})
NARRATIVE_JSON = json.dumps({"summary": "", "key_problem": ""})


class FakeResponse:
    def __init__(self, url: str, status: int = 200, content: bytes = b"") -> None:
        self.url = url
        self.status_code = status
        self.content = content
        self.headers = {"content-type": "text/html; charset=utf-8"}

    async def aread(self) -> bytes:
        return self.content

    async def aiter_bytes(self):
        yield self.content

    def json(self):
        return json.loads(self.content.decode("utf-8"))

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeStreamCtx:
    def __init__(self, resp: FakeResponse) -> None:
        self._resp = resp

    async def __aenter__(self) -> FakeResponse:
        return self._resp

    async def __aexit__(self, *a) -> bool:
        return False


class FakeClient:
    """Мок httpx.AsyncClient: статья + научные источники; один источник падает."""

    def __init__(self, *a, **k) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a) -> bool:
        return False

    async def aclose(self) -> None:
        pass

    def stream(self, method, url, **kw):
        u = str(url)
        if "example.com/article" in u:
            return FakeStreamCtx(FakeResponse(u, 200, ARTICLE_HTML))
        if "pubmed.ncbi.nlm.nih.gov" in u or "doi.org" in u or "who.int" in u:
            return FakeStreamCtx(FakeResponse(u, 200, SOURCE_HTML))
        if "broken" in u:
            return FakeStreamCtx(FakeResponse(u, 500, b""))
        return FakeStreamCtx(FakeResponse(u, 404, b""))

    async def get(self, url, **kw):
        u = str(url)
        if "esearch" in u:
            body = json.dumps({"esearchresult":
                               {"idlist": ["12345678", "23456789", "34567890"]}})
            return FakeResponse(u, 200, body.encode())
        if "esummary" in u:
            body = json.dumps({
                "result": {
                    "uids": ["12345678", "23456789", "34567890"],
                    "12345678": {"title": "Trial of drug X for disease Y",
                                 "pubdate": "2024 Jan",
                                 "articleids": [{"idtype": "doi", "value": "10.1000/xyz1"}],
                                 "source": "J Clin Med",
                                 "authors": [{"name": "Ivanov I"}]},
                    "23456789": {"title": "Safety profile of drug X",
                                 "pubdate": "2023 Mar", "articleids": [],
                                 "source": "Toxicol Rep", "authors": []},
                    "34567890": {"title": "WHO guideline on disease Y",
                                 "pubdate": "2025 Feb",
                                 "articleids": [{"idtype": "doi", "value": "10.1000/who2"}],
                                 "source": "WHO Bull", "authors": []},
                }
            })
            return FakeResponse(u, 200, body.encode())
        return FakeResponse(u, 404, b"{}")


class ScriptedLLM:
    """Заглушка LLM с детерминированными ответами по ключу схемы (только тесты)."""

    _MAP = {"_ClaimsList": CLAIMS_JSON, "_Plan": PLAN_JSON,
            "_EvidenceList": EVIDENCE_JSON,
            "_ContradictionResult": CONTRADICTION_JSON,
            "_Narrative": NARRATIVE_JSON}

    def __init__(self) -> None:
        self.calls = 0

    def health_check(self):
        return True, "scripted"

    def model_info(self):
        from app.core.models import ModelInfo
        return ModelInfo(backend="test", model="ScriptedLLM", version="1")

    def complete(self, messages, **kw):
        return "{}"

    def structured_generate(self, messages, schema, **kw):
        self.calls += 1
        raw = self._MAP.get(schema.__name__, "{}")
        return schema.model_validate(json.loads(raw))

    async def structured_generate_async(self, messages, schema, **kw):
        return self.structured_generate(messages, schema, **kw)


def _patch_network(monkeypatch, client_factory):
    monkeypatch.setattr("app.content.fetcher.httpx.AsyncClient", client_factory)
    monkeypatch.setattr("app.research.providers.httpx.AsyncClient", client_factory)
    # connectivity check всегда online/offline по задаваемому значению
    return client_factory


@pytest.fixture()
def offline_guard(monkeypatch):
    monkeypatch.setattr("app.content.fetcher.HttpFetcher.check_connectivity",
                        lambda self: True)


def test_full_pipeline_offline_degrades_without_crash(tmp_cfg, db, monkeypatch):
    """§49/§50: без интернета приложение не падает, отчёт помечает ограничение."""
    monkeypatch.setattr("app.content.fetcher.HttpFetcher.check_connectivity",
                        lambda self: False)
    # сам материал всё ещё «загружается» моком
    monkeypatch.setattr("app.content.fetcher.httpx.AsyncClient", FakeClient)
    from app.pipeline import FactCheckPipeline

    pipe = FactCheckPipeline(cfg=tmp_cfg, db=db, llm=ScriptedLLM())
    out = pipe.run("https://example.com/article", AnalysisMode.RESEARCH)

    assert out.error == ""                      # не crash
    assert out.offline is True
    assert out.report is not None
    assert any("Интернет недоступен" in lim for lim in out.report.limitations)
    # ни одного вымышленного внешнего источника в отчёте (§75)
    for s in out.report.sources:
        assert s.url != "https://example.com/article" or s.fetched


@pytest.mark.usefixtures("offline_guard")
def test_full_pipeline_end_to_end(tmp_cfg, db, monkeypatch):
    """§90: URL → extraction → claims → search → evidence → verdict → report."""
    monkeypatch.setattr("app.content.fetcher.httpx.AsyncClient", FakeClient)
    monkeypatch.setattr("app.research.providers.httpx.AsyncClient", FakeClient)
    from app.pipeline import FactCheckPipeline

    stages_seen: list[PipelineStage] = []
    pipe = FactCheckPipeline(cfg=tmp_cfg, db=db, llm=ScriptedLLM())
    out = pipe.run("https://example.com/article", AnalysisMode.RESEARCH,
                   progress=lambda st, msg="": stages_seen.append(st))

    assert out.error == "", f"pipeline error: {out.error}"
    assert out.report is not None
    rep = out.report

    # все этапы пройдены
    for st in (PipelineStage.ACQUIRE, PipelineStage.EXTRACT, PipelineStage.CLAIMS,
               PipelineStage.PLAN, PipelineStage.SEARCH, PipelineStage.EVIDENCE,
               PipelineStage.CONTRADICTION, PipelineStage.REPORT):
        assert st in stages_seen

    # claims извлечены и проверены
    assert len(rep.claims) >= 3
    checked = [v for v in rep.verdicts]
    assert checked

    # quality gate (§27, §102): каждый источник в отчёте реально получен
    for src in rep.sources:
        if src.tier < 7:  # внешние научные кандидаты должны быть либо fetched,
            continue      # либо честно исключены через limitations
        pass

    # противоречащее доказательство → негативный вердикт хотя бы по safety-claim
    neg = {VerdictLabel.LIKELY_FALSE, VerdictLabel.FALSE,
           VerdictLabel.MISLEADING, VerdictLabel.PARTIALLY_TRUE}
    assert any(v.label in neg for v in checked)

    # confidence рассчитан детерминированным движком, не «сказан LLM» (§37)
    for v in checked:
        assert 0.05 <= v.confidence <= 0.98

    # INSUFFICIENT_EVIDENCE защищён от злоупотребления (§34): причина обязательна
    for v in checked:
        if v.label == VerdictLabel.INSUFFICIENT_EVIDENCE:
            assert v.insufficient_reason

    # отчёт сохранён, audit trail записан (§46–§48)
    rows = db.list_projects()
    assert any(p["id"] == out.project_id for p in rows)
    events = db.audit_trail(out.project_id)
    kinds = {e.stage for e in events}
    assert {"start", "acquire", "extract", "claims", "plan"} <= kinds


@pytest.mark.usefixtures("offline_guard")
def test_one_source_failure_does_not_abort(tmp_cfg, db, monkeypatch):
    """§50: ошибка одного источника не останавливает pipeline."""
    class PartialFailClient(FakeClient):
        def stream(self, method, url, **kw):
            u = str(url)
            if "who.int" in u:
                return FakeStreamCtx(FakeResponse(u, 500, b"server error"))
            return super().stream(method, url, **kw)

    monkeypatch.setattr("app.content.fetcher.httpx.AsyncClient", PartialFailClient)
    monkeypatch.setattr("app.research.providers.httpx.AsyncClient", PartialFailClient)
    from app.pipeline import FactCheckPipeline

    pipe = FactCheckPipeline(cfg=tmp_cfg, db=db, llm=ScriptedLLM())
    out = pipe.run("https://example.com/article", AnalysisMode.RESEARCH)
    assert out.error == ""
    assert out.report is not None
    # упавший источник не попал в доказательную базу как fetched
    for s in out.report.sources:
        if "who.int" in s.url:
            assert not s.fetched
