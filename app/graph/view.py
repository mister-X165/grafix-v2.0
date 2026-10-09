"""Виджет графа связей для PySide6: отрисовка, зум, перетаскивание узлов.

Рисует готовый layout-граф (app.graph.layout.layout_graph) на QGraphicsScene.
Лёгкий: без внешних JS/CDN; сборка графа выполняется в QThread, чтобы тяжёлый
вызов не блокировал UI (стабильность).
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt, QRectF, QSize, QThread, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QPainter, QPen)
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem,
                               QGraphicsScene, QGraphicsSimpleTextItem,
                               QGraphicsView, QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QWidget, QSizePolicy)

log = logging.getLogger(__name__)

_NODE_COLORS = {
    "Article": "#2f6fed", "Claim": "#7c3aed", "Evidence": "#0891b2",
    "Source": "#15803d", "Verdict": "#b45309", "Entity": "#64748b",
}
_EDGE_COLORS = {"supports": "#15803d", "contradicts": "#b91c1c",
                "derived_from": "#b45309", "cites": "#6366f1"}


class _EdgeLine(QGraphicsItem):
    """Линия ребра с подписью отношения."""

    def __init__(self, x1, y1, x2, y2, label: str, color: str, dashed=False):
        super().__init__()
        self._pts = (x1, y1, x2, y2)
        self._label = label[:24]
        self._color = QColor(color)
        self._dashed = dashed

    def boundingRect(self) -> QRectF:  # noqa: N802
        x1, y1, x2, y2 = self._pts
        return QRectF(min(x1, x2) - 60, min(y1, y2) - 20,
                      abs(x2 - x1) + 120, abs(y2 - y1) + 40)

    def paint(self, p: QPainter, *args):
        x1, y1, x2, y2 = self._pts
        pen = QPen(self._color)
        pen.setWidthF(1.4)
        if self._dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawLine(int(x1), int(y1), int(x2), int(y2))
        p.setPen(QPen(QColor("#94a3b8")))
        f = QFont(); f.setPointSize(7); p.setFont(f)
        p.drawText(int((x1 + x2) / 2) - 20, int((y1 + y2) / 2) - 4, self._label)


class GraphBuildWorker(QThread):
    """Фоновая сборка+компоновка графа из отчёта (UI не блокируется)."""

    built = Signal(object)      # dict {'nodes','edges'} c координатами
    failed = Signal(str)

    def __init__(self, report, include_entities: bool = True, parent=None):
        super().__init__(parent)
        self._report = report
        self._include_entities = include_entities

    def run(self) -> None:
        try:
            from app.graph.builder import build_report_graph
            from app.graph.layout import layout_graph
            g = build_report_graph(self._report,
                                   include_entities=self._include_entities)
            self.built.emit(layout_graph(g))
        except Exception as e:  # стабильность: ошибка слоя графа не роняет UI
            log.exception("graph build failed")
            self.failed.emit(f"Не удалось построить граф: {type(e).__name__}")


class GraphView(QGraphicsView):
    """Интерактивный холст: зум колесом, перетаскивание узлов и сцены."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._node_items: dict[str, QGraphicsEllipseItem] = {}
        self._node_data: dict[str, dict] = {}
        self.node_clicked = None  # callback(node_dict)

    def clear_graph(self) -> None:
        self._scene.clear()
        self._node_items.clear()
        self._node_data.clear()

    def set_graph(self, graph: dict) -> None:
        """Отобразить {'nodes': [...c x,y], 'edges': [...]}"""
        self.clear_graph()
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        pos = {n["id"]: (n.get("x", 0.0), n.get("y", 0.0)) for n in nodes}
        by_id = {n["id"]: n for n in nodes}
        for e in edges:
            a, b = pos.get(e["source"]), pos.get(e["target"])
            if not a or not b:
                continue
            color = _EDGE_COLORS.get(e.get("relation", ""), "#475569")
            self._scene.addItem(_EdgeLine(a[0], a[1], b[0], b[1],
                                          e.get("relation", ""), color,
                                          dashed=e.get("kind") == "hidden"))
        for n in nodes:
            x, y = pos[n["id"]]
            r = 14 if n.get("type") == "Article" else (
                11 if n.get("type") == "Verdict" else 8)
            it = self._scene.addEllipse(x - r, y - r, 2 * r, 2 * r,
                                        QPen(QColor("#0f172a")),
                                        QBrush(QColor(_NODE_COLORS.get(
                                            n.get("type", ""), "#64748b"))))
            it.setFlag(QGraphicsItem.ItemIsMovable, True)
            it.setToolTip(f"{n.get('type','')}: {(n.get('label') or '')[:120]}")
            lbl = self._scene.addSimpleText(
                (n.get("label") or "")[:38], QFont("", 7))
            lbl.setPos(x - 40, y - r - 16)
            lbl.setBrush(QBrush(QColor("#cbd5e1")))
            self._node_items[n["id"]] = it
            self._node_data[n["id"]] = by_id[n["id"]]
        xs = [p[0] for p in pos.values()] or [0]
        ys = [p[1] for p in pos.values()] or [0]
        self._scene.setSceneRect(min(xs) - 80, min(ys) - 60,
                                 max(xs) - min(xs) + 160,
                                 max(ys) - min(ys) + 120)
        self.fitInView(self._scene.sceneRect(),
                       Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, ev) -> None:  # noqa: N802
        factor = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        super().mouseReleaseEvent(ev)
        if self.node_clicked is not None:
            item = self.itemAt(ev.pos())
            for nid, it in self._node_items.items():
                if it is item:
                    try:
                        self.node_clicked(self._node_data[nid])
                    finally:
                        break


class GraphPanel(QWidget):
    """Панель «Граф связей»: заголовок, легенда, кнопки, холст."""

    def __init__(self, on_back, on_open_claim, parent=None) -> None:
        super().__init__(parent)
        self.on_back = on_back
        self.on_open_claim = on_open_claim
        self._worker: Optional[GraphBuildWorker] = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)

        top = QHBoxLayout()
        back = QPushButton("← Назад")
        back.setObjectName("back")
        back.clicked.connect(on_back)
        top.addWidget(back)
        self.status = QLabel("Граф связей")
        self.status.setStyleSheet("font-weight:600;")
        top.addWidget(self.status, 1)
        self.btn_reload = QPushButton("Обновить")
        self.btn_reload.setObjectName("back")
        top.addWidget(self.btn_reload)
        lay.addLayout(top)

        legend = QHBoxLayout()
        legend.setSpacing(10)
        for name, color in _NODE_COLORS.items():
            chip = QLabel(f"<span style='color:{color}'>●</span> {name}")
            chip.setTextFormat(Qt.TextFormat.RichText)
            legend.addWidget(chip)
        legend.addStretch(1)
        lay.addLayout(legend)

        self.view = GraphView()
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding,
                                QSizePolicy.Policy.Expanding)
        self.view.setMinimumHeight(380)
        self.view.node_clicked = self._on_node_clicked
        lay.addWidget(self.view, 1)
        self.btn_reload.clicked.connect(self.reload)
        self._last_report = None

    def show_report(self, report) -> None:
        """Запустить фоновую сборку графа для отчёта."""
        self._last_report = report
        self.status.setText("Граф связей — строю…")
        self.reload()

    def reload(self) -> None:
        if self._last_report is None:
            return
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(200)
        self._worker = GraphBuildWorker(self._last_report)
        self._worker.built.connect(self._on_built)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_built(self, graph: dict) -> None:
        self.view.set_graph(graph)
        from app.graph.builder import graph_stats
        st = graph_stats(graph)
        self.status.setText(
            f"Граф связей · узлов: {st.get('nodes', 0)}, "
            f"рёбер: {st.get('edges', 0)}")

    def _on_failed(self, msg: str) -> None:
        self.status.setText(msg)

    def _on_node_clicked(self, node: dict) -> None:
        if node.get("id", "").startswith("claim:") and self.on_open_claim:
            self.on_open_claim(node["id"].split(":", 1)[1])
