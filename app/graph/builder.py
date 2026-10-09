"""Построение графа связей из результатов анализа (репорт → nodes/edges).

Два слоя графа:
1. Доказательный слой (детерминированный, всегда доступен):
   claim —supports/contradicts→ evidence ←source— источник;
   связи источников: derived_from (репосты, ТЗ §24), cites (DOI-пересечения);
   вердикт статьи связан с ключевыми claims.
2. Сущностный слой (как в исходном модуле ``graph/``): тройки
   subject|relation|object, извлечённые эвристикой ``model.infer``
   (``heuristic_extract`` + ``cooccurrence_extract``) из текста материала.
   Извлечение выполняется лениво и изолировано: ошибка модели не должна
   ронять UI (стабильность).

Узел хранит тип: Claim / Evidence / Source / Verdict / Entity / Article.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.models import EvidenceDirection, Report
from app.sources.graph import SourceGraph

log = logging.getLogger(__name__)

#: Максимум троек в слое сущностей (чтобы граф оставался читаемым)
MAX_ENTITY_TRIPLES = 60


def _node(nid: str, label: str, ntype: str, **extra: Any) -> dict[str, Any]:
    n: dict[str, Any] = {"id": nid, "label": label, "type": ntype}
    n.update(extra)
    return n


def _edge(src: str, dst: str, relation: str, kind: str = "explicit",
          **extra: Any) -> dict[str, Any]:
    e: dict[str, Any] = {
        "id": f"{src}|{relation}|{dst}|{kind}",
        "source": src,
        "target": dst,
        "label": relation,
        "relation": relation,
        "kind": kind,
    }
    e.update(extra)
    return e


def _entity_triples(text: str) -> list[dict[str, Any]]:
    """Тройки из текста материала через эвристики model.infer (как в graph/).

    Возвращает [] при недоступности модуля или пустом результате —
    graceful degradation без исключения.
    """
    if not text or len(text.strip()) < 80:
        return []
    try:
        from model.infer import cooccurrence_extract, heuristic_extract
    except Exception:  # модель/зависимости могут отсутствовать в лёгкой сборке
        log.debug("model.infer unavailable for entity layer", exc_info=True)
        return []
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    try:
        triples = list(heuristic_extract(text)) + list(cooccurrence_extract(text))
    except Exception:
        log.warning("entity triple extraction failed", exc_info=True)
        return []
    for t in triples:
        key = (t.subject, t.relation, t.object)
        if key in seen or not t.subject or not t.object:
            continue
        seen.add(key)
        d = t.as_dict()
        d.setdefault("kind", "explicit")
        out.append(d)
        if len(out) >= MAX_ENTITY_TRIPLES:
            break
    return out


def build_report_graph(report: Report, *, include_entities: bool = True) -> dict[str, list]:
    """Собрать {'nodes': [...], 'edges': [...]} из отчёта анализа."""
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def add_node(n: dict) -> None:
        if n["id"] not in nodes:
            nodes[n["id"]] = n

    def add_edge(e: dict) -> None:
        if all(e["id"] != x["id"] for x in edges):
            edges.append(e)

    by_id = {c.id: c for c in report.claims}
    src_by_id = {s.id: s for s in report.sources}

    # корень — статья
    root_id = "article"
    add_node(_node(root_id, report.input_url or "Материал", "Article",
                   url=report.input_url, verdict=report.overall_verdict.value))

    # вердикт статьи
    vroot = "verdict:overall"
    add_node(_node(vroot, report.overall_verdict.value, "Verdict",
                   confidence=round(report.overall_confidence, 2)))
    add_edge(_edge(root_id, vroot, "has_verdict"))

    # claims + их вердикты
    for v in report.verdicts:
        cl = by_id.get(v.claim_id)
        cid = f"claim:{v.claim_id}"
        text = (cl.text if cl else v.claim_id)
        if len(text) > 120:
            text = text[:120] + "…"
        add_node(_node(cid, text, "Claim",
                       verdict=v.label.value, confidence=round(v.confidence, 2)))
        add_edge(_edge(root_id, cid, "contains_claim",
                       importance=round(cl.importance, 2) if cl else 0.5))
        add_edge(_edge(cid, vroot, "contributes_to", kind="hidden"))

    # evidence + источники
    for ev in report.evidence:
        eid = f"evidence:{ev.id}"
        etext = ev.text if len(ev.text) <= 140 else ev.text[:140] + "…"
        add_node(_node(eid, etext, "Evidence", direction=ev.direction.value,
                       strength=round(ev.strength, 2), verified=ev.quote_verified))
        cid = f"claim:{ev.claim_id}"
        rel = ("supports" if ev.direction == EvidenceDirection.SUPPORTS
               else "contradicts" if ev.direction == EvidenceDirection.CONTRADICTS
               else "relates_to")
        if cid in nodes:
            add_edge(_edge(eid, cid, rel, confidence=round(ev.strength, 2)))
        sid = f"source:{ev.source_id}"
        sm = src_by_id.get(ev.source_id)
        if sm is not None:
            stitle = (sm.title or sm.url)[:90]
            add_node(_node(sid, stitle, "Source", url=sm.url, tier=sm.tier,
                           source_type=sm.source_type, fetched=sm.fetched))
            add_edge(_edge(sid, eid, "provides"))

    # связи между источниками (original vs derivative, cites) — ТЗ §68
    sg = SourceGraph().build(list(report.sources))
    for der, orig in sg.derived_from.items():
        d, o = f"source:{der}", f"source:{orig}"
        if d in nodes and o in nodes:
            nodes[d]["is_derivative"] = True
            add_edge(_edge(d, o, "derived_from"))
    for sid, refs in sg.cites.items():
        for rid in refs:
            a, b = f"source:{sid}", f"source:{rid}"
            if a in nodes and b in nodes:
                add_edge(_edge(a, b, "cites"))

    # сущностный слой (тройки из текста материала) — как в исходном graph/
    if include_entities:
        doc_text = ""
        for s in report.sources:
            if s.url == report.input_url and s.body_text:
                doc_text = s.body_text
                break
        if not doc_text:
            doc_text = report.summary
        for t in _entity_triples(doc_text):
            subj, obj = t["subject"], t["object"]
            for name in (subj, obj):
                add_node(_node(f"entity:{name}", name, "Entity"))
            add_edge(_edge(f"entity:{subj}", f"entity:{obj}", t["relation"],
                           kind=t.get("kind", "explicit"),
                           origin=t.get("origin", "base")))

    return {"nodes": list(nodes.values()), "edges": edges}


def graph_stats(graph: dict[str, list]) -> dict[str, int]:
    """Краткая сводка по графу для подписи в UI."""
    types: dict[str, int] = {}
    for n in graph.get("nodes", []):
        t = n.get("type", "?")
        types[t] = types.get(t, 0) + 1
    return {"nodes": len(graph.get("nodes", [])),
            "edges": len(graph.get("edges", [])),
            **types}
