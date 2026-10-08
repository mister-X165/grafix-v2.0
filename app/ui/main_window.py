"""Главное окно PySide6 (ТЗ §10, §11, §42–§45, §97–§98, §106).

Принципы:
- минималистичный интерфейс для небольшого экрана;
- прогресс по этапам пайплайна без внутреннего chain-of-thought (§11, §106);
- итоговый экран: общий вердикт + сводка + список claims с кнопкой «Подробнее»;
- все URL берутся только из реально полученных источников, они кликабельны (§45);
- тяжёлый анализ выполняется в фоновом потоке, UI не блокируется;
- стектрейсы и технические ID пользователю не показываются — только в log (§52, §106).
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QLabel, QLineEdit, QProgressBar,
    QPushButton, QRadioButton, QScrollArea, QVBoxLayout, QWidget,
)
from PySide6.QtCore import QUrl

from app.config import get_config
from app.core.models import (AnalysisMode, Claim, EvidenceDirection, PipelineStage,
                             Report, VerdictLabel)
from app.dependencies import check_dependencies
from app.logging_setup import setup_logging
from app.pipeline import FactCheckPipeline, PipelineOutcome
from app.report.generator import DISCLAIMER

log = logging.getLogger(__name__)

LABEL_RU = {
    VerdictLabel.VERIFIED: "ПОДТВЕРЖДЕНО",
    VerdictLabel.LIKELY_TRUE: "СКОРЕЕ ВСЕГО ВЕРНО",
    VerdictLabel.PARTIALLY_TRUE: "ЧАСТИЧНО ВЕРНО",
    VerdictLabel.MISLEADING: "⚠ ВВОДИТ В ЗАБЛУЖДЕНИЕ",
    VerdictLabel.LIKELY_FALSE: "СКОРЕЕ ВСЕГО НЕВЕРНО",
    VerdictLabel.FALSE: "НЕДОСТОВЕРНО",
    VerdictLabel.INSUFFICIENT_EVIDENCE: "НЕДОСТАТОЧНО ДОКАЗАТЕЛЬСТВ",
}

STAGES_ORDER = [PipelineStage.ACQUIRE, PipelineStage.EXTRACT, PipelineStage.CLAIMS,
                PipelineStage.PLAN, PipelineStage.SEARCH, PipelineStage.EVIDENCE,
                PipelineStage.CONTRADICTION, PipelineStage.VERDICT, PipelineStage.REPORT]

_ACCENT = "#2f6fed"
_BG = "#ffffff"
_MUTED = "#6b7280"

_STYLE = f"""
QWidget {{ background: {_BG}; color: #1f2430; font-size: 14px; }}
QLineEdit {{ padding: 10px 12px; border: 1px solid #d5dae3; border-radius: 8px;
             background: #fbfcfe; selection-background-color: {_ACCENT}; }}
QLineEdit:focus {{ border: 1px solid {_ACCENT}; }}
QPushButton#primary {{ background: {_ACCENT}; color: white; border: none;
    border-radius: 8px; padding: 11px 26px; font-weight: 600; font-size: 15px; }}
QPushButton#primary:hover {{ background: #2559c9; }}
QPushButton#primary:disabled {{ background: #b9c6e4; }}
QPushButton.link {{ color: {_ACCENT}; text-decoration: underline; border: none;
    background: transparent; padding: 0; }}
QPushButton#back {{ color: {_ACCENT}; border: 1px solid #d5dae3; border-radius: 8px;
    padding: 8px 18px; background: white; }}
QLabel.verdict {{ font-size: 21px; font-weight: 700; }}
QFrame.card {{ background: #f7f9fc; border: 1px solid #e6eaf2; border-radius: 10px; }}
QRadioButton {{ spacing: 6px; }}
QScrollBar:vertical {{ width: 10px; background: transparent; }}
QScrollBar::handle:vertical {{ background: #cfd6e4; border-radius: 5px; min-height: 30px; }}
QProgressBar {{ border: none; background: #eef1f6; border-radius: 5px; height: 10px; }}
QProgressBar::chunk {{ background: {_ACCENT}; border-radius: 5px; }}
"""


def _card() -> QFrame:
    f = QFrame()
    f.setProperty("class", "card")
    f.setObjectName("card")
    f.setStyleSheet(
        "QFrame#card { background:#f7f9fc; border:1px solid #e6eaf2; border-radius:10px; }")
    return f


class AnalysisWorker(QThread):
    """Фоновый запуск pipeline: GUI не блокируется и не падает при ошибках (§50)."""

    stage_done = Signal(object, str)      # PipelineStage, сообщение
    finished_ok = Signal(object)          # PipelineOutcome
    failed = Signal(str)                  # человекочитаемое сообщение

    def __init__(self, url: str, mode: AnalysisMode, parent=None) -> None:
        super().__init__(parent)
        self.url, self.mode = url, mode

    def run(self) -> None:  # noqa: D102
        try:
            cfg = get_config()
            pipe = FactCheckPipeline(cfg)
            outcome = pipe.run(self.url, self.mode, progress=self._progress)
            if outcome.error:
                self.failed.emit(outcome.error)
            else:
                self.finished_ok.emit(outcome)
        except Exception as e:  # последняя линия обороны UI
            log.exception("worker crashed")
            self.failed.emit("Непредвиденная ошибка анализа. Подробности — в журнале.")

    def _progress(self, stage: PipelineStage, msg: str) -> None:
        self.stage_done.emit(stage, msg)


class ProgressPanel(QWidget):
    """Чек-лист этапов в стиле ТЗ §11: ✓ выполнено / ● выполняется / ○ ожидает."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._labels: dict[PipelineStage, QLabel] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(6)
        for st in STAGES_ORDER:
            lbl = QLabel(f"○ {st.value}")
            lbl.setStyleSheet(f"color:{_MUTED};")
            self._labels[st] = lbl
            lay.addWidget(lbl)
        self.bar = QProgressBar()
        self.bar.setRange(0, len(STAGES_ORDER))
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        lay.addWidget(self.bar)

    def mark(self, stage: PipelineStage, msg: str = "") -> None:
        idx = STAGES_ORDER.index(stage)
        for i, st in enumerate(STAGES_ORDER):
            lbl = self._labels[st]
            if i < idx:
                lbl.setText(f"✓ {st.value}")
                lbl.setStyleSheet("color:#15803d;")
            elif i == idx:
                extra = f" — {msg}" if msg and len(msg) < 70 else ""
                lbl.setText(f"● {st.value}{extra}")
                lbl.setStyleSheet(f"color:{_ACCENT}; font-weight:600;")
            else:
                lbl.setText(f"○ {st.value}")
                lbl.setStyleSheet(f"color:{_MUTED};")
        self.bar.setValue(idx + 1)

    def finish_all(self) -> None:
        for st in STAGES_ORDER:
            lbl = self._labels[st]
            lbl.setText(f"✓ {st.value}")
            lbl.setStyleSheet("color:#15803d;")
        self.bar.setValue(len(STAGES_ORDER))


class SummaryView(QWidget):
    """Итоговый экран отчёта (§42, §97)."""

    def __init__(self, on_back, on_open_claim, parent=None) -> None:
        super().__init__(parent)
        self.on_back = on_back
        self.on_open_claim = on_open_claim
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(10)

        self.verdict_lbl = QLabel("")
        self.verdict_lbl.setObjectName("verdict")
        self.verdict_lbl.setProperty("class", "verdict")
        self.verdict_lbl.setWordWrap(True)
        lay.addWidget(self.verdict_lbl)

        self.conf_lbl = QLabel("")
        self.conf_lbl.setStyleSheet(f"color:{_MUTED};")
        lay.addWidget(self.conf_lbl)

        self.summary_lbl = QLabel("")
        self.summary_lbl.setWordWrap(True)
        self.summary_lbl.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.summary_lbl)

        self.counts_lbl = QLabel("")
        self.counts_lbl.setWordWrap(True)
        self.counts_lbl.setStyleSheet(f"color:{_MUTED};")
        lay.addWidget(self.counts_lbl)

        self.limit_lbl = QLabel("")
        self.limit_lbl.setWordWrap(True)
        self.limit_lbl.setStyleSheet("color:#b45309;")
        self.limit_lbl.hide()
        lay.addWidget(self.limit_lbl)

        self.list_lay = QVBoxLayout()
        self.list_lay.setSpacing(8)
        lay.addLayout(self.list_lay)

        lay.addStretch(1)

    def set_report(self, report: Report) -> None:
        for i in reversed(range(self.list_lay.count())):
            item = self.list_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        label = report.overall_verdict
        self.verdict_lbl.setText(LABEL_RU.get(label, label.value))
        color = {"good": "#15803d", "bad": "#b91c1c", "warn": "#b45309"}
        tone = ("good" if label in (VerdictLabel.VERIFIED, VerdictLabel.LIKELY_TRUE)
                else "bad" if label in (VerdictLabel.FALSE, VerdictLabel.LIKELY_FALSE)
                else "warn")
        self.verdict_lbl.setStyleSheet(f"color:{color[tone]}; font-size:21px; font-weight:700;")
        self.conf_lbl.setText(f"Уверенность системы: {report.overall_confidence:.0%} · "
                              f"Проверено утверждений: {len(report.verdicts)}")
        self.summary_lbl.setText(report.summary.replace("\n", "<br>"))

        c = report.verdict_counts()
        parts = []
        for lab in (VerdictLabel.VERIFIED, VerdictLabel.LIKELY_TRUE,
                    VerdictLabel.PARTIALLY_TRUE, VerdictLabel.MISLEADING,
                    VerdictLabel.LIKELY_FALSE, VerdictLabel.FALSE,
                    VerdictLabel.INSUFFICIENT_EVIDENCE):
            n = c.get(lab.value, 0)
            if n:
                short = LABEL_RU.get(lab, lab.value).replace("⚠ ", "")
                parts.append(f"{short.lower()} — {n}")
        self.counts_lbl.setText(" · ".join(parts) if parts else "Вердикты отсутствуют.")

        if report.limitations:
            self.limit_lbl.setText("<b>Ограничения проверки:</b><br>" +
                                   "<br>".join(f"— {x}" for x in report.limitations))
            self.limit_lbl.show()
        else:
            self.limit_lbl.hide()

        by_id: dict[str, Claim] = {cl.id: cl for cl in report.claims}
        shown = sorted(report.verdicts,
                       key=lambda v: -(by_id.get(v.claim_id).importance
                                       if v.claim_id in by_id else 0.0))[:8]
        for v in shown:
            card = _card()
            hl = QHBoxLayout(card)
            hl.setContentsMargins(12, 10, 12, 10)
            cl = by_id.get(v.claim_id)
            text = (cl.text if cl else v.claim_id)
            if len(text) > 160:
                text = text[:160] + "…"
            left = QLabel(f"<b>{v.label.value.replace('_', ' ').title()}</b>"
                          f" ({v.confidence:.0%})<br>{text}")
            left.setWordWrap(True)
            hl.addWidget(left, 1)
            btn = QPushButton("Подробнее")
            btn.setObjectName("back")
            btn.clicked.connect(lambda _=False, vid=v.claim_id: self.on_open_claim(vid))
            hl.addWidget(btn, 0, Qt.AlignTop)
            self.list_lay.addWidget(card)


class ClaimDetailView(QWidget):
    """Подробный view claim (§43, §98): verdict, почему, evidence за/против, источники."""

    def __init__(self, on_back, parent=None) -> None:
        super().__init__(parent)
        self.on_back = on_back
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)
        back = QPushButton("← Назад к итогу")
        back.setObjectName("back")
        back.clicked.connect(on_back)
        root.addWidget(back)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(scroll)
        self._scroll = scroll
        self._inner = QWidget()
        self._lay = QVBoxLayout(self._inner)
        self._lay.setSpacing(10)
        self._lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self._inner)

    def show_claim(self, report: Report, claim_id: str) -> None:
        for i in reversed(range(self._lay.count())):
            it = self._lay.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()

        v = next((x for x in report.verdicts if x.claim_id == claim_id), None)
        cl = next((c for c in report.claims if c.id == claim_id), None)
        if v is None or cl is None:
            self._lay.addWidget(QLabel("Данные по утверждению недоступны."))
            return
        src_by_id = {s.id: s for s in report.sources}
        sc_by_id = {s.source_id: s for s in report.scores}

        def add(title: str, widget: QWidget) -> None:
            h = QLabel(title)
            h.setStyleSheet(f"color:{_MUTED}; font-weight:600;")
            self._lay.addWidget(h)
            self._lay.addWidget(widget)

        head = QLabel(f"<b>{cl.text}</b><br>"
                      f"<span style='color:{_MUTED}'>Тип: {cl.claim_type.value} · "
                      f"Важность: {cl.importance:.2f}</span>")
        head.setWordWrap(True)
        add("Утверждение", head)

        verdict = QLabel(LABEL_RU.get(v.label, v.label.value) +
                         f" · уверенность {v.confidence:.0%}")
        verdict.setWordWrap(True)
        add("Вердикт", verdict)

        why = QLabel(v.explanation or "—")
        why.setWordWrap(True)
        add("Почему", why)
        if v.insufficient_reason:
            r = QLabel(f"Причина недостаточности: {v.insufficient_reason}")
            r.setWordWrap(True)
            add("", r)

        ev_for = [e for e in report.evidence
                  if e.claim_id == claim_id and e.direction == EvidenceDirection.SUPPORTS]
        ev_against = [e for e in report.evidence
                      if e.claim_id == claim_id and e.direction == EvidenceDirection.CONTRADICTS]
        add("Доказательства ЗА", self._evidence_block(ev_for, src_by_id))
        add("Доказательства ПРОТИВ", self._evidence_block(ev_against, src_by_id))

        used_ids = {e.source_id for e in ev_for + ev_against}
        quality_rows = []
        for sid in used_ids:
            sc = sc_by_id.get(sid)
            if sc:
                quality_rows.append(
                    f"authority {sc.authority:.2f} · methodology {sc.methodology:.2f} · "
                    f"recency {sc.recency:.2f} · relevance {sc.relevance:.2f} · "
                    f"independence {sc.independence:.2f} · transparency {sc.transparency:.2f} "
                    f"→ итог {sc.overall:.2f}")
        q = QLabel("<br>".join(quality_rows) or "Оценка качества не выполнялась.")
        q.setWordWrap(True)
        add("Качество источников", q)

        primary = [src_by_id[sid] for sid in used_ids
                   if sid in src_by_id and src_by_id[sid].tier <= 2]
        secondary = [src_by_id[sid] for sid in used_ids
                     if sid in src_by_id and src_by_id[sid].tier > 2]
        add("Первичные исследования", self._sources_block(primary))
        add("Вторичные источники", self._sources_block(secondary))
        add("Источники утверждения", self._sources_block(
            [src_by_id[sid] for sid in used_ids if sid in src_by_id]))

    def _evidence_block(self, evs: list, src_by_id: dict) -> QWidget:
        box = _card()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)
        if not evs:
            lay.addWidget(QLabel("—"))
            return box
        for e in evs:
            src = src_by_id.get(e.source_id)
            origin = src.url if src else ""
            item = QLabel(f"«{e.text}»<br><span style='color:{_MUTED}'>"
                          f"сила {e.strength:.2f} · релевантность {e.relevance:.2f}"
                          f"{' · цитата проверена' if e.quote_verified else ''}</span>")
            item.setWordWrap(True)
            item.setTextFormat(Qt.TextFormat.RichText)
            lay.addWidget(item)
            if origin:
                lay.addWidget(self._link(origin))
        return box

    def _sources_block(self, sources: list) -> QWidget:
        box = _card()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        if not sources:
            lay.addWidget(QLabel("—"))
            return box
        for s in sources:
            title = s.title or s.url
            info = QLabel(f"T{s.tier} · {title}")
            info.setWordWrap(True)
            lay.addWidget(info)
            lay.addWidget(self._link(s.url))
        return box

    @staticmethod
    def _link(url: str) -> QPushButton:
        b = QPushButton(url if len(url) <= 90 else url[:90] + "…")
        b.setProperty("class", "link")
        b.setStyleSheet("text-align:left; padding:0;")
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.clicked.connect(lambda _=False: QDesktopServices.openUrl(QUrl(url)))
        return b


class MainWindow(QWidget):
    """Окно приложения: главный экран → прогресс → итог → детали claim."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Медицинский фактчекер")
        self.setMinimumSize(420, 560)
        self.resize(480, 640)
        self.setStyleSheet(_STYLE)
        self._cfg = get_config()
        self._outcome: Optional[PipelineOutcome] = None
        self._worker: Optional[AnalysisWorker] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        self.stack_host = QWidget()
        outer.addWidget(self.stack_host)
        self._layout = QVBoxLayout(self.stack_host)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self.pages: dict[str, QWidget] = {}
        self._build_start_page()
        self._progress = ProgressPanel()
        self.pages["progress"] = self._wrap_scroll(self._progress)
        self.summary = SummaryView(self._show_start, self._open_claim)
        self.pages["summary"] = self._wrap_scroll(self.summary)
        self.detail = ClaimDetailView(lambda: self._show_page("summary"))
        self.pages["detail"] = self._wrap_scroll(self.detail)

        self._show_page("start")

    # ------------------------------------------------------------- pages ----
    def _wrap_scroll(self, w: QWidget) -> QWidget:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(w)
        lay.addStretch(1)
        area.setWidget(inner)
        return area

    def _build_start_page(self) -> None:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 16, 8, 8)
        lay.setSpacing(12)

        title = QLabel("Медицинский фактчекер")
        f = QFont()
        f.setPointSize(19)
        f.setBold(True)
        title.setFont(f)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)

        subtitle = QLabel("Локальная проверка достоверности медицинской информации\n"
                          "в интернете на основе доказательств, а не мнений модели.")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"color:{_MUTED};")
        subtitle.setWordWrap(True)
        lay.addWidget(subtitle)

        lay.addSpacing(10)
        lay.addWidget(QLabel("URL материала"))
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("https://example.com/article")
        self.url_edit.returnPressed.connect(self._on_check_clicked)
        lay.addWidget(self.url_edit)

        lay.addWidget(QLabel("Режим"))
        row = QHBoxLayout()
        self.rb_controlled = QRadioButton("Контролируемый (только доверенные источники)")
        self.rb_research = QRadioButton("Исследовательский (расширенный поиск)")
        default = getattr(self._cfg.research, "mode_default", "research")
        self.rb_controlled.setChecked(default == "controlled")
        self.rb_research.setChecked(not self.rb_controlled.isChecked())
        group = QButtonGroup(self)
        group.addButton(self.rb_controlled)
        group.addButton(self.rb_research)
        row.addWidget(self.rb_controlled)
        lay.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(self.rb_research)
        lay.addLayout(row2)

        lay.addSpacing(6)
        self.check_btn = QPushButton("ПРОВЕРИТЬ")
        self.check_btn.setObjectName("primary")
        self.check_btn.clicked.connect(self._on_check_clicked)
        lay.addWidget(self.check_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        self.err_lbl = QLabel("")
        self.err_lbl.setWordWrap(True)
        self.err_lbl.setStyleSheet("color:#b91c1c;")
        self.err_lbl.hide()
        lay.addWidget(self.err_lbl)

        lay.addStretch(1)
        note = QLabel(DISCLAIMER)
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{_MUTED}; font-size:12px;")
        lay.addWidget(note)

        deps = check_dependencies()
        if not deps.ok:
            warn = QLabel(deps.human_report())
            warn.setWordWrap(True)
            warn.setStyleSheet("color:#b45309; font-size:12px;")
            lay.addWidget(warn)

        self.pages["start"] = page

    def _show_page(self, name: str) -> None:
        cur = self.pages.get(self._current) if hasattr(self, "_current") else None
        if cur is not None:
            self._layout.removeWidget(cur)
            cur.setParent(None)
        w = self.pages[name]
        self._layout.addWidget(w)
        self._current = name

    def _show_start(self) -> None:
        self._show_page("start")

    # ------------------------------------------------------------ actions ---
    def _open_claim(self, claim_id: str) -> None:
        if self._outcome and self._outcome.report:
            self.detail.show_claim(self._outcome.report, claim_id)
            self._show_page("detail")

    def _on_check_clicked(self) -> None:
        url = self.url_edit.text().strip()
        if not url:
            self._set_error("Вставьте URL интернет-материала.")
            return
        if not url.lower().startswith(("http://", "https://")):
            self._set_error("URL должен начинаться с http:// или https://")
            return
        mode = (AnalysisMode.CONTROLLED if self.rb_controlled.isChecked()
                else AnalysisMode.RESEARCH)
        self._set_error("")
        self.check_btn.setEnabled(False)
        self._progress = ProgressPanel()
        self.pages["progress"] = self._wrap_scroll(self._progress)
        self._show_page("progress")

        self._worker = AnalysisWorker(url, mode)
        self._worker.stage_done.connect(self._on_stage)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_stage(self, stage: PipelineStage, msg: str) -> None:
        self._progress.mark(stage, msg)

    def _on_finished(self, outcome: PipelineOutcome) -> None:
        self._progress.finish_all()
        self._outcome = outcome
        if outcome.report is not None:
            self.summary.set_report(outcome.report)
        self.check_btn.setEnabled(True)
        self._show_page("summary")

    def _on_failed(self, message: str) -> None:
        self.check_btn.setEnabled(True)
        self._show_page("start")
        self._set_error(message)

    def _set_error(self, text: str) -> None:
        self.err_lbl.setText(text)
        self.err_lbl.setVisible(bool(text))


def run_gui() -> int:
    """Точка входа GUI. Возвращает код завершения приложения."""
    from PySide6.QtWidgets import QApplication
    setup_logging()
    app = QApplication([])
    app.setApplicationName("Медицинский фактчекер")
    win = MainWindow()
    win.show()
    return app.exec()
