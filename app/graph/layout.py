"""Детерминированная 2D-компоновка графа (без внешних зависимостей).

ИспользуетFORCE-LAYOUT на основе модели пружин (Fruchterman–Reingold),
с фиксированным начальным размещением (seed) — одинаковый вход даёт
одинаковый выход (важно для стабильности UI и тестов).
"""

from __future__ import annotations

import math
import random
from typing import Any

ITERATIONS = 160


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def compute_layout(nodes: list[dict], edges: list[dict],
                   width: float = 800.0, height: float = 560.0,
                   seed: int = 42) -> dict[str, tuple[float, float]]:
    """Вернуть {node_id: (x, y)} — позиции в прямоугольнике [0..width, 0..height]."""
    ids = [n["id"] for n in nodes]
    if not ids:
        return {}
    rnd = random.Random(seed)
    pos = {nid: (rnd.uniform(0.1 * width, 0.9 * width),
                 rnd.uniform(0.1 * height, 0.9 * height)) for nid in ids}
    if len(ids) == 1:
        pos[ids[0]] = (width / 2.0, height / 2.0)
        return pos

    links = [(e["source"], e["target"]) for e in edges
             if e.get("source") in pos and e.get("target") in pos]
    area = width * height
    k = math.sqrt(area / len(ids))
    temp = min(width, height) / 6.0
    mx, my = width / 2.0, height / 2.0

    for it in range(ITERATIONS):
        disp: dict[str, list[float]] = {nid: [0.0, 0.0] for nid in ids}
        # отталкивание всех пар
        for i in range(len(ids)):
            a = ids[i]
            ax, ay = pos[a]
            for j in range(i + 1, len(ids)):
                b = ids[j]
                bx, by = pos[b]
                dx, dy = ax - bx, ay - by
                d = math.hypot(dx, dy) or 0.01
                f = (k * k) / d
                ux, uy = dx / d, dy / d
                disp[a][0] += ux * f
                disp[a][1] += uy * f
                disp[b][0] -= ux * f
                disp[b][1] -= uy * f
        # притяжение по рёбрам
        for a, b in links:
            dx, dy = pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]
            d = math.hypot(dx, dy) or 0.01
            f = (d * d) / k
            ux, uy = dx / d, dy / d
            disp[a][0] -= ux * f
            disp[a][1] -= uy * f
            disp[b][0] += ux * f
            disp[b][1] += uy * f
        # смещение с охлаждением
        cool = temp * (1.0 - it / ITERATIONS)
        for nid in ids:
            dx, dy = disp[nid]
            d = math.hypot(dx, dy) or 0.01
            step = min(d, cool)
            x, y = pos[nid]
            pos[nid] = (_clamp(x + dx / d * step, 30.0, width - 30.0),
                        _clamp(y + dy / d * step, 26.0, height - 26.0))
        # лёгкое стягивание к центру, чтобы граф не упирался в края
        for nid in ids:
            x, y = pos[nid]
            pos[nid] = (x + (mx - x) * 0.02, y + (my - y) * 0.02)
    return pos


def layout_graph(graph: dict[str, Any], width: float = 800.0,
                 height: float = 560.0) -> dict[str, Any]:
    """Обогатить {'nodes','edges'} координатами: у узлов появляются x, y."""
    pos = compute_layout(graph.get("nodes", []), graph.get("edges", []),
                         width, height)
    out_nodes = []
    for n in graph.get("nodes", []):
        m = dict(n)
        m["x"], m["y"] = pos.get(n["id"], (width / 2, height / 2))
        out_nodes.append(m)
    return {"nodes": out_nodes, "edges": graph.get("edges", [])}
