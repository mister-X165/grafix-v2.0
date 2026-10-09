"""Тесты стабильности GUI (реgress-тесты против зависаний).

Воспроизводят корневые причины «зависаний»:
1. Синхронные блокирующие вызовы (health_check / complete) выполняются внутри
   asyncio event loop → весь конвейер становится последовательным и долгие
   сетевые таймауты LLM (~300 с по умолчанию) блокируют обработку событий;
2. Worker не поддерживает отмену — закрытие окна во время анализа «вешает» приложение;
3. UI-обработчики падают на неполных данных отчёта (None/пустой verdict).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.core.models import (AnalysisMode, PipelineStage, Report, VerdictLabel)


# ------------------------------------------------------------------ helpers --
class SlowSyncProvider:
    """LLM-провайдер только с синхронными методами, каждый «зависает» на sleep."""

    name = "slow-sync"

    def __init__(self, delay: float = 0.2) -> None:
        self.delay = delay

    def health_check(self):
        time.sleep(self.delay)          # имитация долгоразрешающегося DNS/connect
        return True, "ok"

    def complete(self, messages, *, max_tokens=None, temperature=None):
        time.sleep(self.delay)          # имитация long request_timeout_s
        return '{"claims": []}'

    def model_info(self):
        from app.core.models import ModelInfo
        return ModelInfo(backend="slow", model="test")


class TimedSlowProvider(SlowSyncProvider):
    """Как SlowSyncProvider, но фиксирует абсолютный момент каждого обращения к LLM.

    Возвращает валидные JSON-ответы для всех ролей конвейера (claims/plan/…),
    чтобы каждый этап реально обращался к модели — иначе срабатывают эвристические
    fallback-и и тест перестает что-либо проверять (источник flaky-поведения).
    """

    def __init__(self, delay: float = 0.15) -> None:
        super().__init__(delay)
        self.calls: list[float] = []   # time.monotonic() на вход в complete()

    def complete(self, messages, *, max_tokens=None, temperature=None):
        self.calls.append(time.monotonic())
        time.sleep(self.delay)         # имитация long request_timeout_s
        blob = " ".join(str(m.get("content", "")) for m in messages).lower()
        if "claim" in blob:
            return ('{"claims": [{"text": "Витамин C лечит грипп.", '
                    '"claim_type": "MEDICAL_FACT", "importance": 0.9, '
                    '"check_required": true}]}')
        if "план" in blob or "query" in blob or "research" in blob:
            return ('{"queries": [{"query": "витамин C грипп исследования", '
                    '"purpose": "SUPPORT"}, {"query": "витамин C грипп критика", '
                    '"purpose": "CONTRADICTION"}]}')
        return '{"verdict": "INSUFFICIENT_EVIDENCE", "confidence": 0.3}'


def _run_pipeline(tmp_cfg, monkeypatch, llm, url="https://example.com/a"):
    """Прогон pipeline c заглушками сети; возвращает outcome и замеры.

    events: список (t_abs, stage), где t_abs — абсолютный time.monotonic()
    момента доставки события прогресса.
    """
    from app import pipeline as pl

    events: list[tuple[float, object]] = []

    class FakeFetcher:
        def __init__(self, cfg, budget=None):
            self.cfg = cfg

        async def fetch(self, u):
            from app.content.fetcher import FetchResult
            return FetchResult(url=u, status_code=200,
                               raw="<html><body><p>Витамин C лечит грипп.</p>"
                                  "<p>Исследования подтверждают эффект.</p></body></html>"
                                  .encode(), content_type="text/html")

        def check_connectivity(self):
            return False      # offline: поиск пропускается, путь короткий

    monkeypatch.setattr(pl, "HttpFetcher", FakeFetcher)
    pipe = pl.FactCheckPipeline(tmp_cfg, llm=llm)
    outcome = pipe.run(url, AnalysisMode.RESEARCH,
                       progress=lambda st, msg="": events.append((time.monotonic(), st)))
    return outcome, events


# ------------------------------------------------------------ test: cancel --
def test_worker_cancel_within_second(qtbot, tmp_cfg, monkeypatch):
    """Отмена анализа должна завершать worker ≤1 с, а не ждать конца этапа."""
    from app.ui.main_window import AnalysisWorker

    worker = AnalysisWorker("http://127.0.0.1:9/nope", AnalysisMode.RESEARCH)
    started = time.monotonic()

    def fake_run(self, url, mode, progress=None):
        from app.pipeline import PipelineOutcome
        for _ in range(40):                    # потенциально ~20 секунд без отмены
            if worker.should_stop():
                return PipelineOutcome(error="cancelled-marker")
            time.sleep(0.05)
            if progress:
                progress(PipelineStage.SEARCH, "поиск")
        return PipelineOutcome(error="not-cancelled")

    monkeypatch.setattr("app.pipeline.FactCheckPipeline.run", fake_run)
    results = {}
    worker.failed.connect(lambda m: results.setdefault("failed", m))
    worker.start()
    time.sleep(0.2)
    worker.cancel()
    with qtbot.waitSignal(worker.finished, timeout=3000):
        pass
    elapsed = time.monotonic() - started
    assert elapsed < 1.5, f"cancel не сработал быстро: {elapsed:.2f} с"
    # worker сообщает об отмене пользователю; pipeline при этом вернулся досрочно
    # (не дождав 40 итераций фейкового этапа) — именно это и проверяет тест выше.
    assert results.get("failed") in ("cancelled-marker",
                                     "Анализ остановлен пользователем.")


# ----------------------------------------------------- test: no UI freeze ----
def test_async_llm_does_not_serial_sleep_in_loop(tmp_cfg, monkeypatch):
    """Синхронный провайдер исполняется в executor: event loop остаётся отзывчивым
    и несколько вызовов перекрываются по времени (concurrency)."""
    from app.llm.base import to_thread_complete, to_thread_health_check

    llm = SlowSyncProvider(delay=0.3)

    async def scenario():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        t = asyncio.create_task(ticker())
        t0 = time.monotonic()
        # три «долгих» вызова параллельно
        res = await asyncio.gather(*[to_thread_complete(llm, [{"role": "user", "content": "x"}])
                                     for _ in range(3)])
        wall = time.monotonic() - t0
        ok, msg = await to_thread_health_check(llm)
        t.cancel()
        return wall, res, ok, ticks

    wall, res, ok, ticks = asyncio.run(scenario())
    assert ok and res[0]                      # вызовы прошли
    # без executor 3×0.3 с выполнялись бы строго последовательно (wall ≥ 0.9);
    # в пуле потоков они перекрываются
    assert wall < 0.8, f"вызовы сериализованы в event loop: wall={wall:.2f}"
    assert ticks >= 5, "event loop блокировался (тикеры не успевали)"


class TimedSlowProvider(SlowSyncProvider):
    """Как SlowSyncProvider, но фиксирует момент каждого обращения к LLM."""

    def __init__(self, delay: float = 0.15) -> None:
        super().__init__(delay)
        self.calls: list[float] = []

    def complete(self, messages, *, max_tokens=None, temperature=None):
        self.calls.append(time.monotonic())
        return super().complete(messages, max_tokens=max_tokens,
                                temperature=temperature)


def test_progress_events_are_paced(tmp_cfg, monkeypatch):
    """Регресс «зависания»: события прогресса приходят инкрементально, а не
    всей пачкой в самом конце анализа.

    Детерминированная проверка вместо замера wall-clock span (flaky на медленных
    CI): сравниваются МОМЕНТЫ доставки этапов и моменты обращений к LLM
    (оба — абсолютный time.monotonic() одного процесса). Инвариант порядка:
    ранние этапы (ACQUIRE/EXTRACT/CLAIMS/PLAN) обязаны попасть в UI ДО начала
    первого долгого LLM-шага. Если конвейер блокирует event loop (старый баг),
    все события «замораживаются» и доставляются только после последнего
    LLM-вызова — тогда early < 2 и тест падает детерминированно.
    """
    slow = TimedSlowProvider(delay=0.15)
    outcome, events = _run_pipeline(tmp_cfg, monkeypatch, slow)
    assert outcome.error == "", f"pipeline упал: {outcome.error}"
    assert len(events) >= 3
    assert len(slow.calls) >= 2, "LLM вызывалась <2 раз — тест ничего не проверяет"

    first_llm_call = min(slow.calls)
    early = sum(1 for t_abs, _st in events if t_abs < first_llm_call)
    assert early >= 2, (
        f"события прогресса пришли пачкой после LLM-вызовов: "
        f"events={[round(t - first_llm_call, 3) for t, _ in events]}, "
        f"llm_calls={[round(c - first_llm_call, 3) for c in slow.calls]}")


# --------------------------------------------------------- test: null-safety --
@pytest.mark.parametrize("label", [None, VerdictLabel.FALSE])
def test_summary_view_survives_degenerate_report(qtbot, tmp_cfg, label):
    """Итоговый экран не падает на пустом/неполном отчёте (раньше — KeyError/TypeError)."""
    from app.ui.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    report = Report(project_id="p1", input_url="http://x")
    if label is not None:
        from app.core.models import Verdict
        report.verdicts = [Verdict(claim_id="c1", label=label, confidence=0.5)]
        report.overall_verdict = label
    win.summary.set_report(report)            # не должно бросать исключение
    win.detail.show_claim(report, "c1")       # и здесь
    win.detail.show_claim(report, "missing")  # несуществующий claim


def test_stage_mark_tolerates_unknown_message(qtbot, tmp_cfg):
    from app.ui.main_window import ProgressPanel
    p = ProgressPanel()
    qtbot.addWidget(p)
    p.mark(PipelineStage.SEARCH, "x" * 500)   # очень длинное сообщение
    p.mark(PipelineStage.VERDICT)
    p.finish_all()
