"""Оркестратор pipeline (ТЗ §2, §78–§103).

Полный конвейер:

    URL → получение материала → извлечение текста → claims → план исследования
        → поиск источников → оценка источников → evidence → подтверждения
        → опровержения (adversarial) → противоречия → синтез → verdict → отчёт

Ключевые свойства:
- LLM НЕ является источником истины (§3): она извлекает/классифицирует,
  детерминированный код считает verdict/confidence (§36, §37);
- graceful degradation (§50, §103): ошибка любого источника/провайдера
  не останавливает анализ, а попадает в limitations;
- OFFLINE MODE (§49): без интернета анализ деградирует до локальных
  возможностей и понятного сообщения, приложение не падает;
- каждый этап пишется в SQLite + audit trail (§46–§48);
- кэш документов (§48) переиспользуется через HttpFetcher.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from app.config import AppConfig, get_config
from app.content.extractor import extract_document
from app.content.fetcher import HttpFetcher, RequestBudget, extract_pdf_text
from app.claims.extractor import ClaimExtractor
from app.core.models import (AnalysisMode, AuditEvent, Claim, ContentNature, Document,
                             Evidence, ModelInfo, PipelineStage, Project, Report,
                             SearchQuery, SearchResult, SourceMeta, SourceScores,
                             Verdict, VerdictLabel)
from app.evidence.contradiction import ContradictionAnalyst
from app.evidence.extractor import EvidenceExtractor
from app.llm.base import (CancellationRequested, LLMProvider, LLMUnavailableError,
                          _generic_structured_async, create_provider,
                          to_thread_complete, to_thread_health_check)
from app.report.generator import ReportGenerator
from app.research.dedup import cluster_independent, dedup_results, dedup_sources
from app.research.planner import ResearchPlanner
from app.research.providers import build_providers
from app.sources.evaluator import classify_source, evaluate
from app.storage.db import Database
from app.verdict.engine import compute_verdict, summarize

log = logging.getLogger(__name__)

ProgressCallback = Callable[[PipelineStage, str], None]


@dataclass
class PipelineOutcome:
    """Результат работы оркестратора для UI/CLI."""

    report: Optional[Report] = None
    error: str = ""              # фатальная ошибка (только критические стадии)
    offline: bool = False        # интернет недоступен (§49)
    duration_s: float = 0.0
    project_id: str = ""


def _source_from_result(r: SearchResult) -> SourceMeta:
    return SourceMeta(
        url=r.url, title=r.title, doi=r.doi, pmid=r.pmid,
        source_type=r.source_type or "", publication_date=r.publication_date,
        metadata={**r.metadata, "snippet": r.snippet, "provider": r.provider},
    )


class FactCheckPipeline:
    """Многоэтапный evidence-based конвейер проверки одного URL."""

    def __init__(self, cfg: Optional[AppConfig] = None,
                 llm: Optional[LLMProvider] = None,
                 db: Optional[Database] = None,
                 stop_event: Optional["threading.Event"] = None) -> None:
        self.cfg = cfg or get_config()
        self.db = db or Database.instance(self.cfg)
        self._llm_override = llm
        self.llm: Optional[LLMProvider] = None
        self.model_info = ModelInfo()
        #: поток отмены анализа (ставится UI-воркером; см. app.llm.base)
        self.stop_event = stop_event

    # ------------------------------------------------------------- audit ---
    def _check_cancelled(self) -> None:
        """Точечная проверка отмены между этапами (стабильность: UI не «зависает»)."""
        if self.stop_event is not None and self.stop_event.is_set():
            raise CancellationRequested("Анализ остановлен пользователем.")

    def _audit(self, project_id: str, stage: str, event: str, **detail) -> None:
        try:
            self.db.audit(AuditEvent(project_id=project_id, stage=stage,
                                     event=event, detail=detail))
        except Exception:  # аудит не должен ронять pipeline
            log.debug("audit write failed", exc_info=True)

    @staticmethod
    def _emit(cb: Optional[ProgressCallback], stage: PipelineStage, msg: str = "") -> None:
        if cb is not None:
            try:
                cb(stage, msg)
            except Exception:  # callback из GUI — не критичен
                log.debug("progress callback failed", exc_info=True)

    # --------------------------------------------------------------- run ---
    def run(self, url: str, mode: AnalysisMode,
            progress: Optional[ProgressCallback] = None) -> PipelineOutcome:
        try:
            return asyncio.run(self.run_async(url, mode, progress))
        except CancellationRequested:
            log.info("analysis cancelled")
            return PipelineOutcome(error="Анализ остановлен пользователем.",
                                   duration_s=0.0)
        except Exception as e:  # последняя линия обороны: UI не должен видеть crash (§50)
            log.exception("pipeline crashed")
            return PipelineOutcome(error=f"Внутренняя ошибка анализа: {type(e).__name__}: {e}")

    async def run_async(self, url: str, mode: AnalysisMode,
                        progress: Optional[ProgressCallback] = None) -> PipelineOutcome:
        """Общий таймаут анализа (§32): при превышении — graceful деградация,
        отчёт формируется из уже накопленных данных, приложение не падает."""
        t0 = time.monotonic()
        timeout = self.cfg.limits.total_analysis_timeout_s
        try:
            return await asyncio.wait_for(self._run_inner(url, mode, progress, t0),
                                          timeout=timeout)
        except asyncio.TimeoutError:
            log.warning("total analysis timeout after %.0fs", timeout)
            outcome = PipelineOutcome(error=(
                f"Превышен общий лимит времени анализа ({timeout:.0f} с). "
                "Проверьте настройки LLM и доступность источников."),
                duration_s=time.monotonic() - t0)
            return outcome
        except CancellationRequested:
            return PipelineOutcome(error="Анализ остановлен пользователем.",
                                   duration_s=time.monotonic() - t0)

    async def _run_inner(self, url: str, mode: AnalysisMode,
                         progress: Optional[ProgressCallback],
                         t0: float) -> PipelineOutcome:
        cfg = self.cfg
        limits = cfg.limits
        budget = RequestBudget(limits.max_requests_per_analysis)
        fetcher = HttpFetcher(cfg, budget=budget)
        limitations: list[str] = []
        outcome = PipelineOutcome()

        project = Project(input_url=url, mode=mode)
        self.db.save_project(project)
        pid = project.id
        outcome.project_id = pid
        self._audit(pid, "start", "analysis_started", url=url, mode=mode.value)

        # ---- LLM health check (§4, §51) ------------------------------------
        self._check_cancelled()
        llm = self._llm_override
        if llm is not None:
            # duck-typed провайдеры (тестовые заглушки, сторонние адаптеры)
            # могут не наследовать LLMProvider — подмешиваем базовые async-методы,
            # чтобы блокирующий complete() исполнялся в executor, а не в loop.
            if not callable(getattr(llm, "structured_generate_async", None)):
                llm.structured_generate_async = (
                    lambda msgs, schema, **kw: _generic_structured_async(
                        llm, msgs, schema, **kw))
            if not callable(getattr(llm, "check_cancelled", None)):
                llm.check_cancelled = lambda: None
            if getattr(llm, "stop_event", None) is None and self.stop_event is not None:
                llm.stop_event = self.stop_event
        else:
            try:
                llm = create_provider(cfg)
                if self.stop_event is not None:
                    llm.stop_event = self.stop_event
                # сетевой health_check — вне event loop (иначе блокирует цикл на таймаут)
                ready, msg = await to_thread_health_check(llm, self.stop_event)
                if not ready:
                    limitations.append(f"LLM недоступна ({msg}); используется "
                                       "детерминированный режим без нейросетевого анализа.")
                    llm = None
                else:
                    self.model_info = llm.model_info()
            except CancellationRequested:
                raise
            except LLMUnavailableError as e:
                limitations.append(f"LLM недоступна: {e}. Детерминированный режим.")
                llm = None
            except Exception as e:
                log.warning("LLM init failed: %s", e)
                limitations.append(f"Ошибка инициализации LLM: {type(e).__name__}. "
                                   "Детерминированный режим.")
                llm = None
        project.model_info = self.model_info
        self.db.save_project(project)

        # ---- STAGE 1: acquire material -------------------------------------
        self._check_cancelled()
        self._emit(progress, PipelineStage.ACQUIRE)
        try:
            fetched = await asyncio.wait_for(fetcher.fetch(url),
                                             timeout=limits.request_timeout_s * 2 + 10)
        except asyncio.TimeoutError:
            fetched = None
        if fetched is None or not fetched.ok:
            err = fetched.error if fetched else "Таймаут загрузки"
            self.db.update_project_status(pid, "FAILED")
            self._audit(pid, "acquire", "fetch_failed", error=err)
            outcome.error = (f"Не удалось получить материал по URL: {err}. "
                             "Проверьте адрес и доступность сайта.")
            outcome.duration_s = time.monotonic() - t0
            return outcome
        self._audit(pid, "acquire", "material_fetched", final_url=fetched.url,
                    bytes=len(fetched.raw), from_cache=fetched.from_cache)
        self._emit(progress, PipelineStage.ACQUIRE, fetched.url)

        # ---- STAGE 2: content extraction -----------------------------------
        self._emit(progress, PipelineStage.EXTRACT)
        doc: Document = extract_document(fetched, cfg)
        if not doc.text.strip():
            self.db.update_project_status(pid, "FAILED")
            outcome.error = ("Из страницы не удалось извлечь текст "
                             "(JS-рендеринг или пустой документ).")
            outcome.duration_s = time.monotonic() - t0
            return outcome
        self.db.save_document(pid, doc)
        project.title = (doc.title or url)[:200]
        self.db.save_project(project)
        self._audit(pid, "extract", "text_extracted", chars=len(doc.text),
                    title=doc.title[:120], nature=doc.content_nature.value)
        if doc.content_nature != ContentNature.INFORMATION:
            limitations.append(f"Материал помечен как {doc.content_nature.value}: "
                               "коммерческий интерес отображён в отчёте (§41).")

        # ---- STAGE 3: claims ------------------------------------------------
        self._check_cancelled()
        self._emit(progress, PipelineStage.CLAIMS)
        extractor = ClaimExtractor(llm, cfg)
        claims = await extractor.extract_async(doc)
        for c in claims:
            self.db.save_claim(pid, c)
        self._audit(pid, "claims", "claims_extracted", count=len(claims),
                    llm=llm is not None)
        if not claims:
            self.db.update_project_status(pid, "DONE")
            outcome.report = self._empty_report(pid, url, mode, doc, limitations,
                                                "В материале не найдено проверяемых утверждений.")
            outcome.duration_s = time.monotonic() - t0
            return outcome

        to_check = [c for c in claims if c.check_required][: limits.max_pages_per_analysis]
        if not to_check:
            to_check = claims[:3]

        # ---- connectivity / offline guard (§49) ----------------------------
        online = await asyncio.get_running_loop().run_in_executor(None,
                                                                  fetcher.check_connectivity)
        if not online:
            outcome.offline = True
            limitations.append("Интернет недоступен: внешний исследовательский поиск "
                               "выполнить невозможно. Анализ деградирован до оценки "
                               "материала без внешних доказательств (§49).")

        # ---- STAGE 4: research plan ----------------------------------------
        self._check_cancelled()
        self._emit(progress, PipelineStage.PLAN)
        planner = ResearchPlanner(llm, cfg)
        # планы для всех claims строятся параллельно (LLM-вызовы в executor'е)
        plan_pairs = await asyncio.gather(
            *[planner.plan_async(c, pid) for c in to_check], return_exceptions=True)
        plans: dict[str, list] = {}
        for c, p in zip(to_check, plan_pairs):
            if isinstance(p, CancellationRequested):
                raise p
            if isinstance(p, BaseException):
                log.warning("plan failed for %s: %s — template plan", c.id, p)
                plans[c.id] = planner.template_plan(c, pid)
            else:
                plans[c.id] = p
        n_queries = sum(len(v) for v in plans.values())
        self._audit(pid, "plan", "research_planned", queries=n_queries,
                    claims=len(to_check))

        sources: dict[str, SourceMeta] = {}
        scores: dict[str, SourceScores] = {}
        evidence: list[Evidence] = []

        if online:
            # ---- STAGE 5: search + fetch + evaluate + evidence -------------
            self._emit(progress, PipelineStage.SEARCH)
            providers = build_providers(cfg, mode)
            sem_search = asyncio.Semaphore(limits.search_concurrency)
            evx = EvidenceExtractor(llm, cfg)

            async def _run_query(q: SearchQuery) -> tuple[SearchQuery, list[SearchResult]]:
                async with sem_search:
                    res_list: list[SearchResult] = []
                    for p in providers:
                        rs = await p.search(q.query, max_results=6, mode=mode)
                        res_list.extend(rs)
                        if len(res_list) >= 12:
                            break
                    q.result_count = len(res_list)
                    return q, res_list

            query_tasks = [_run_query(q) for qs in plans.values() for q in qs]
            gathered = await asyncio.gather(*query_tasks, return_exceptions=True)
            results_by_claim: dict[str, list[SearchResult]] = {c.id: [] for c in to_check}
            for g in gathered:
                if not isinstance(g, tuple):
                    log.warning("search query task failed: %s", g)
                    continue
                q, res = g
                self.db.save_search_query(q)
                results_by_claim.setdefault(q.claim_id, []).extend(res)

            # candidates per claim: dedup (§67) + limit
            candidates_by_claim: dict[str, list[SourceMeta]] = {}
            for cid, res_list in results_by_claim.items():
                uniq = dedup_results(res_list)[:8]
                candidates_by_claim[cid] = [_source_from_result(r) for r in uniq]

            # global URL-level dedup before fetching (performance §108)
            seen_urls: set[str] = set()
            ordered_srcs: list[SourceMeta] = []
            for srcs in candidates_by_claim.values():
                for s in srcs:
                    if s.url in seen_urls:
                        continue
                    seen_urls.add(s.url)
                    ordered_srcs.append(s)

            # fetch pages concurrently (shared budget & cache); errors never abort (§50)
            async def _fetch_source(s: SourceMeta) -> None:
                try:
                    fr = await fetcher.fetch(s.url)
                except Exception as e:
                    s.fetch_error = f"{type(e).__name__}"
                    return
                if not fr.ok:
                    s.fetch_error = fr.error or f"HTTP {fr.status_code}"
                    return
                if fr.is_pdf:
                    text, err = await asyncio.get_running_loop().run_in_executor(
                        None, extract_pdf_text, fr.raw, limits.max_pdf_pages)
                    if err:
                        s.fetch_error = err
                        return
                    s.body_text = text
                else:
                    d = extract_document(fr, cfg)
                    s.body_text = d.text
                    s.title = s.title or d.title
                    s.author = s.author or d.author
                    s.publication_date = s.publication_date or d.published
                s.fetched = True

            await asyncio.gather(*[_fetch_source(s) for s in
                                   ordered_srcs[: limits.max_pages_per_analysis]],
                                return_exceptions=True)

            failed_fetches = sum(1 for s in ordered_srcs if not s.fetched)
            if failed_fetches:
                limitations.append(f"{failed_fetches} источников не удалось получить "
                                   "(таймаут/paywall/robots) — они исключены из доказательной базы.")

            # dedup by DOI/PMID/canonical + independence clustering (§24, §67, §68)
            ordered_srcs = dedup_sources(ordered_srcs)
            cluster_independent(ordered_srcs)
            for s in ordered_srcs:
                classify_source(s, cfg)
                sources[s.id] = s
                self.db.save_source(pid, s)

            # scoring: шесть параметров качества, лучшие по релевантности для claim (§22)
            best_rank: dict[str, int] = {}
            for cid, srcs in candidates_by_claim.items():
                claim = next(c for c in to_check if c.id == cid)
                for s in srcs:
                    rank = best_rank.get(s.id, 0)
                    best_rank[s.id] = rank + 1
                    relevance = _lexical_relevance(claim.normalized_text,
                                                   (s.title + " " + (s.body_text or "")[:3000]))
                    sc = evaluate(s, claim_relevance=relevance, mode=mode,
                                  independent_rank=min(rank, 3), cfg=cfg)
                    if s.id not in scores or sc.relevance > scores[s.id].relevance:
                        scores[s.id] = sc
            for sc in scores.values():
                self.db.save_source_score(pid, sc)

            self._emit(progress, PipelineStage.EVIDENCE)
            evx_sem = asyncio.Semaphore(max(1, cfg.limits.fetch_concurrency))

            async def _extract_for(cid: str) -> None:
                claim = next(c for c in to_check if c.id == cid)
                for s in candidates_by_claim.get(cid, []):
                    src = sources.get(s.id)
                    if src is None or not src.fetched:
                        continue   # §27: недоказуемо без реально полученного текста
                    if mode == AnalysisMode.CONTROLLED and not src.trusted_category \
                            and src.tier >= 7:
                        continue   # CONTROLLED: только доверенные категории (§7.1)
                    async with evx_sem:
                        try:
                            evs = await evx.extract(claim, src)
                        except Exception as e:
                            log.warning("evidence extraction error (%s): %s", src.url[:60], e)
                            continue
                    evidence.extend(evs)

            await asyncio.gather(*[_extract_for(c.id) for c in to_check],
                                 return_exceptions=True)
            for e in evidence:
                self.db.save_evidence(pid, e)
            self._audit(pid, "evidence", "evidence_extracted", count=len(evidence),
                        verified=sum(1 for e in evidence if e.quote_verified))

        # ---- STAGES 6–7: preliminary verdict + adversarial revision --------
        self._check_cancelled()
        self._emit(progress, PipelineStage.CONTRADICTION)
        analyst = ContradictionAnalyst(llm, cfg)
        verdicts: list[Verdict] = []

        async def _verdict_for(claim: Claim) -> Verdict:
            claim_ev = [e for e in evidence if e.claim_id == claim.id]
            src_ids = {e.source_id for e in claim_ev}
            s_by_id = {sid: sources[sid] for sid in src_ids if sid in sources}
            sc_by_id = {sid: scores[sid] for sid in src_ids if sid in scores}
            summary = summarize(claim_ev, s_by_id, sc_by_id,
                                expected_sources=cfg.research.min_sources_per_claim)
            prelim = compute_verdict(claim, summary, min_sources=cfg.research.min_sources_per_claim)
            hint = None
            exaggeration = ""
            if cfg.research.adversarial_search and claim_ev:
                try:
                    ana = await analyst.analyze_async(claim, prelim, claim_ev, s_by_id)
                    exaggeration = ana["exaggeration"]
                    hint = ana["revised_label_hint"]
                except CancellationRequested:
                    raise
                except Exception as e:  # adversarial-шаг опционален (§50)
                    log.warning("contradiction analysis failed (%s): %s", claim.id, e)
            final = compute_verdict(claim, summary, exaggeration=exaggeration,
                                    llm_skeptical_hint=hint,
                                    min_sources=cfg.research.min_sources_per_claim)
            final.supporting_evidence_ids = [e.id for e in claim_ev
                                             if e.direction.value == "SUPPORTS"]
            final.contradicting_evidence_ids = [e.id for e in claim_ev
                                                if e.direction.value == "CONTRADICTS"]
            return final

        vres = await asyncio.gather(*[_verdict_for(c) for c in to_check],
                                    return_exceptions=True)
        for r in vres:
            if isinstance(r, CancellationRequested):
                raise r
            if isinstance(r, BaseException):
                log.warning("verdict task failed: %s", r)
                continue
            verdicts.append(r)
            self.db.save_verdict(pid, r)

        self._check_cancelled()
        self._emit(progress, PipelineStage.VERDICT)

        insufficient = sum(1 for v in verdicts
                           if v.label == VerdictLabel.INSUFFICIENT_EVIDENCE)
        rate = insufficient / len(verdicts) if verdicts else 0.0
        self._audit(pid, "verdict", "verdicts_computed", total=len(verdicts),
                    insufficient=insufficient, insufficient_rate=round(rate, 3))
        if rate > 0.8 and len(verdicts) >= 4:
            limitations.append("Высокая доля «недостаточно доказательств» "
                               f"({rate:.0%}) — покрытие исследования ограничено; "
                               "результаты следует трактовать осторожно (§34, §104).")

        # ---- STAGE 8: report + quality gate --------------------------------
        self._emit(progress, PipelineStage.REPORT)
        gen = ReportGenerator(llm, cfg)
        report = await gen.build_async(project_id=pid, url=url, mode=mode, claims=claims,
                                       verdicts=verdicts, sources=list(sources.values()),
                                       scores=list(scores.values()), evidence=evidence,
                                       limitations=limitations, model_info=self.model_info)
        report_path = self.db.save_report(report)
        outcome.duration_s = time.monotonic() - t0
        self.db.finish_project(pid, status="DONE", title=project.title,
                               duration_s=outcome.duration_s)
        self._audit(pid, "report", "report_saved", path=str(report_path))

        outcome.report = report
        return outcome

    # ------------------------------------------------------------ helpers --
    def _empty_report(self, pid: str, url: str, mode: AnalysisMode, doc: Document,
                      limitations: list[str], why: str) -> Report:
        gen = ReportGenerator(None, self.cfg)
        rep = gen.build(project_id=pid, url=url, mode=mode, claims=[], verdicts=[],
                        sources=[], scores=[], evidence=[],
                        limitations=[*limitations, why], model_info=self.model_info)
        rep.summary = why + " " + rep.summary
        self.db.save_report(rep)
        return rep


def _lexical_relevance(claim_norm: str, text: str) -> float:
    """Детерминированная лексическая релевантность claim ↔ источник (0..1)."""
    import re

    terms = [w for w in re.findall(r"[A-Za-zА-Яа-яЁё]{5,}", claim_norm.lower())][:10]
    if not terms:
        return 0.3
    low = text.lower()
    hits = sum(1 for t in terms if t in low)
    return round(min(1.0, 0.15 + 0.85 * hits / len(terms)), 3)
