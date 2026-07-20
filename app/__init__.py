"""Grafix FastAPI backend."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.documents import ALLOWED_EXTENSIONS, DocumentExtractError, extract_text
from graph.history import GraphHistory
from graph.qa import answer_question
from graph.store import connect_graph_store, new_document_id
from model.infer import Extractor

app = FastAPI(title="Grafix", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = connect_graph_store()
extractor = Extractor()
history = GraphHistory()
DOC_TEXTS: dict[str, str] = {}
DOC_COMMENTS: dict[str, dict[str, str]] = {}


class AnalyzeBody(BaseModel):
    text: str
    document_id: str | None = None
    engine: str = "gemma"  # gemma | microgpt | auto


class AskBody(BaseModel):
    question: str
    document_id: str | None = None
    document_text: str | None = None
    web_search: bool = False


class ToggleBody(BaseModel):
    disabled: list[str] = Field(default_factory=list)
    document_id: str | None = None


class EditBody(BaseModel):
    action: str
    document_id: str | None = None
    subject: str | None = None
    relation: str | None = None
    object: str | None = None
    name: str | None = None
    history_id: str | None = None


class SaveGraphBody(BaseModel):
    document_id: str | None = None
    text: str | None = None
    triples: list[dict[str, Any]] | None = None
    nodes: list[dict[str, Any]] | None = None
    edges: list[dict[str, Any]] | None = None
    engine: str | None = None
    source: str | None = None
    title: str | None = None
    history_id: str | None = None
    comments: dict[str, str] | None = None


class RenameGraphBody(BaseModel):
    title: str


class EntityCommentBody(BaseModel):
    name: str
    comment: str = ""
    document_id: str | None = None
    history_id: str | None = None


class EntityGenerateBody(BaseModel):
    name: str
    document_id: str | None = None
    document_text: str | None = None
    history_id: str | None = None
    save: bool = True
    web_search: bool = False


def _resolve_doc_id(document_id: str | None) -> str:
    return document_id or new_document_id()


def _triples_from_edges(edges: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for e in edges:
        s = (e.get("source") or "").strip()
        r = (e.get("relation") or e.get("label") or "").strip()
        o = (e.get("target") or "").strip()
        if s and o:
            out.append({"subject": s, "relation": r, "object": o})
    return out


def _comments_for(doc_id: str) -> dict[str, str]:
    return DOC_COMMENTS.setdefault(doc_id, {})


def _entity_payload(doc_id: str, name: str) -> dict[str, Any]:
    name = (name or "").strip()
    relations = store.neighbors_of(name, doc_id)
    snap = store.get_graph(doc_id)
    highlight_edges = [
        e["id"]
        for e in snap.edges
        if e["source"] == name or e["target"] == name
    ]
    return {
        "document_id": doc_id,
        "name": name,
        "relations": relations,
        "comment": _comments_for(doc_id).get(name, ""),
        "highlight_nodes": [name],
        "highlight_edges": highlight_edges,
        "nodes": snap.nodes,
        "edges": snap.edges,
    }


def _sync_history_comments(history_id: str | None, doc_id: str) -> None:
    if not history_id:
        return
    item = history.get(history_id)
    if not item:
        return
    history.save(
        text=item.get("text") or DOC_TEXTS.get(doc_id, ""),
        triples=item.get("triples") or _triples_from_edges(item.get("edges") or []),
        nodes=item.get("nodes") or [],
        edges=item.get("edges") or [],
        document_id=doc_id,
        engine=item.get("engine"),
        source=item.get("source"),
        title=item.get("title"),
        graph_id=history_id,
        comments=dict(_comments_for(doc_id)),
    )


def _persist_snapshot(
    *,
    document_id: str,
    text: str,
    triples: list[dict],
    nodes: list[dict],
    edges: list[dict],
    engine: str | None = None,
    source: str | None = None,
    title: str | None = None,
    history_id: str | None = None,
    comments: dict[str, str] | None = None,
) -> dict[str, Any]:
    if comments is not None:
        DOC_COMMENTS[document_id] = dict(comments)
    return history.save(
        text=text,
        triples=triples,
        nodes=nodes,
        edges=edges,
        document_id=document_id,
        engine=engine,
        source=source,
        title=title,
        graph_id=history_id,
        comments=dict(_comments_for(document_id)),
    )

@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/><title>Grafix API</title>
<style>
  @font-face{font-family:'RF Dewi Expanded';src:url('http://127.0.0.1:5173/brand/fonts/RFDewiExpanded-Bold.ttf') format('truetype');font-weight:700}
  body{font-family:system-ui,sans-serif;max-width:40rem;margin:3rem auto;padding:0 1rem;line-height:1.5;background:#07090b;color:#dadee1}
  h1{font-family:'RF Dewi Expanded',sans-serif;color:#f4f6f7;letter-spacing:.04em}
  code{background:#1c242c;padding:.1rem .35rem;border-radius:2px}
  a{color:#8a9aa6}
</style></head><body>
<h1>Grafix API</h1>
<p>Бэкенд запущен. Интерфейс — на Vite с бренд-ассетами из <code>media/</code>:</p>
<ul>
  <li><a href="http://127.0.0.1:5173">http://127.0.0.1:5173</a> — UI</li>
  <li><a href="/docs">/docs</a> — Swagger</li>
  <li><a href="/api/health">/api/health</a></li>
</ul>
</body></html>"""


@app.get("/api/health")
def health() -> dict[str, object]:
    from model.lmstudio import LMStudioExtractor

    lm = LMStudioExtractor()
    ready = lm.ping()
    return {
        "status": "ok",
        "history_db": str(history.db_path),
        "uploads": {
            "extensions": sorted(ALLOWED_EXTENSIONS),
            "max_mb": int(__import__("os").environ.get("GRAFIX_MAX_UPLOAD_MB", "12")),
        },
        "lm_studio": {
            "ready": ready,
            "base_url": lm.base_url,
            "model": lm.resolved_model if ready else None,
            "models": lm.list_models() if ready else [],
        },
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = file.filename or "document.txt"
    data = await file.read()
    try:
        text = extract_text(filename, data)
    except DocumentExtractError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Не удалось прочитать файл: {e}",
        ) from e
    return {
        "filename": filename,
        "chars": len(text),
        "text": text,
    }


@app.post("/api/analyze")
def analyze(body: AnalyzeBody) -> dict[str, Any]:
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Пустой текст")

    doc_id = _resolve_doc_id(body.document_id)
    result = extractor.extract(text, engine=body.engine or "gemma")
    snap = store.upsert_triples(doc_id, text, result["triples"])
    DOC_TEXTS[doc_id] = text
    saved = _persist_snapshot(
        document_id=doc_id,
        text=text,
        triples=result["triples"],
        nodes=snap.nodes,
        edges=snap.edges,
        engine=result.get("engine", body.engine),
        source=result.get("source"),
    )
    return {
        "document_id": doc_id,
        "text": text,
        "triples": result["triples"],
        "source": result["source"],
        "engine": result.get("engine", body.engine),
        "model_ready": result["model_ready"],
        "lm_ready": result.get("lm_ready"),
        "lm_model": result.get("lm_model"),
        "hint": result.get("hint"),
        "debug": result.get("debug") or {},
        "nodes": snap.nodes,
        "edges": snap.edges,
        "history_id": saved["id"],
        "history": saved,
    }

@app.get("/api/graph")
def get_graph(document_id: str | None = Query(default=None)) -> dict[str, Any]:
    doc_id = _resolve_doc_id(document_id)
    snap = store.get_graph(doc_id)
    return {
        "document_id": doc_id,
        "nodes": snap.nodes,
        "edges": snap.edges,
    }


@app.post("/api/ask")
def ask(body: AskBody) -> dict[str, Any]:
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Пустой вопрос")
    doc_id = _resolve_doc_id(body.document_id)
    doc_text = body.document_text or DOC_TEXTS.get(doc_id) or ""
    result = answer_question(
        store,
        question,
        doc_id,
        document_text=doc_text,
        prefer_gemma_for_text=True,
        web_search=bool(body.web_search),
    )
    snap = store.get_graph(doc_id)
    result["document_id"] = doc_id
    result["nodes"] = snap.nodes
    result["edges"] = snap.edges
    return result


@app.post("/api/toggle")
def toggle(body: ToggleBody) -> dict[str, Any]:
    disabled = body.disabled or []
    doc_id = _resolve_doc_id(body.document_id)
    lost = store.lost_edges_if_disabled(disabled, doc_id)
    snap = store.get_graph(doc_id)
    disabled_set = set(disabled)
    visible_nodes = [n for n in snap.nodes if n["id"] not in disabled_set]
    visible_edges = [
        e
        for e in snap.edges
        if e["source"] not in disabled_set and e["target"] not in disabled_set
    ]
    return {
        "document_id": doc_id,
        "disabled": disabled,
        "lost_edges": lost,
        "nodes": visible_nodes,
        "edges": visible_edges,
        "all_nodes": snap.nodes,
        "all_edges": snap.edges,
    }


@app.post("/api/edit")
def edit_graph(body: EditBody) -> dict[str, Any]:
    doc_id = _resolve_doc_id(body.document_id)
    if body.action == "add_triple":
        if not all([body.subject, body.relation, body.object]):
            raise HTTPException(status_code=400, detail="Нужны subject, relation, object")
        snap = store.add_triple(doc_id, body.subject, body.relation, body.object)
    elif body.action == "delete_node":
        if not body.name:
            raise HTTPException(status_code=400, detail="Нужен name")
        snap = store.delete_node(doc_id, body.name)
    else:
        raise HTTPException(status_code=400, detail="Неизвестное действие")

    text = DOC_TEXTS.get(doc_id, "")
    triples = _triples_from_edges(snap.edges)
    saved = None
    if body.history_id:
        saved = _persist_snapshot(
            document_id=doc_id,
            text=text,
            triples=triples,
            nodes=snap.nodes,
            edges=snap.edges,
            history_id=body.history_id,
        )
    return {
        "document_id": doc_id,
        "nodes": snap.nodes,
        "edges": snap.edges,
        "triples": triples,
        "history_id": saved["id"] if saved else body.history_id,
        "history": saved,
    }


@app.get("/api/graphs")
def list_graphs(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
    items = history.list_graphs(limit=limit)
    return {"items": items, "count": len(items)}


@app.post("/api/graphs")
def save_graph(body: SaveGraphBody) -> dict[str, Any]:
    doc_id = body.document_id
    text = body.text
    triples = body.triples
    nodes = body.nodes
    edges = body.edges

    if doc_id and (triples is None or nodes is None or edges is None or text is None):
        snap = store.get_graph(doc_id)
        text = text if text is not None else DOC_TEXTS.get(doc_id, "")
        nodes = nodes if nodes is not None else snap.nodes
        edges = edges if edges is not None else snap.edges
        triples = triples if triples is not None else _triples_from_edges(edges)

    if text is None:
        text = ""
    if nodes is None or edges is None:
        raise HTTPException(status_code=400, detail="Нет данных графа для сохранения")
    if triples is None:
        triples = _triples_from_edges(edges)
    if not doc_id:
        doc_id = new_document_id()

    store.upsert_triples(doc_id, text, triples)
    DOC_TEXTS[doc_id] = text
    if body.comments is not None:
        DOC_COMMENTS[doc_id] = dict(body.comments)
    saved = _persist_snapshot(
        document_id=doc_id,
        text=text,
        triples=triples,
        nodes=nodes,
        edges=edges,
        engine=body.engine,
        source=body.source,
        title=body.title,
        history_id=body.history_id,
        comments=DOC_COMMENTS.get(doc_id),
    )
    return {
        "document_id": doc_id,
        "history_id": saved["id"],
        "history": saved,
        "text": saved["text"],
        "triples": saved["triples"],
        "nodes": saved["nodes"],
        "edges": saved["edges"],
        "title": saved["title"],
        "engine": saved.get("engine"),
        "source": saved.get("source"),
        "comments": saved.get("comments") or {},
    }


@app.get("/api/graphs/{graph_id}")
def get_graph_history(graph_id: str, restore: bool = Query(default=True)) -> dict[str, Any]:
    item = history.get(graph_id)
    if not item:
        raise HTTPException(status_code=404, detail="Граф не найден")

    doc_id = item.get("document_id") or new_document_id()
    if restore:
        snap = store.upsert_triples(doc_id, item["text"], item["triples"])
        DOC_TEXTS[doc_id] = item["text"]
        DOC_COMMENTS[doc_id] = dict(item.get("comments") or {})
        item = {**item, "nodes": snap.nodes, "edges": snap.edges, "document_id": doc_id}

    return {
        "document_id": doc_id,
        "history_id": item["id"],
        "text": item["text"],
        "triples": item["triples"],
        "nodes": item["nodes"],
        "edges": item["edges"],
        "engine": item.get("engine"),
        "source": item.get("source"),
        "title": item.get("title"),
        "comments": item.get("comments") or DOC_COMMENTS.get(doc_id, {}),
        "history": item,
    }


@app.patch("/api/graphs/{graph_id}")
def rename_graph(graph_id: str, body: RenameGraphBody) -> dict[str, Any]:
    item = history.rename(graph_id, body.title)
    if not item:
        raise HTTPException(status_code=404, detail="Граф не найден")
    return item


@app.delete("/api/graphs/{graph_id}")
def delete_graph(graph_id: str) -> dict[str, Any]:
    if not history.delete(graph_id):
        raise HTTPException(status_code=404, detail="Граф не найден")
    return {"ok": True, "id": graph_id}


@app.get("/api/entity")
def get_entity(
    name: str = Query(...),
    document_id: str | None = Query(default=None),
) -> dict[str, Any]:
    name = (name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Нужно имя сущности")
    doc_id = _resolve_doc_id(document_id)
    return _entity_payload(doc_id, name)


@app.post("/api/entity/comment")
def save_entity_comment(body: EntityCommentBody) -> dict[str, Any]:
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Нужно имя сущности")
    doc_id = _resolve_doc_id(body.document_id)
    comment = (body.comment or "").strip()
    comments = _comments_for(doc_id)
    if comment:
        comments[name] = comment
    else:
        comments.pop(name, None)
    _sync_history_comments(body.history_id, doc_id)
    payload = _entity_payload(doc_id, name)
    payload["saved"] = True
    return payload


@app.post("/api/entity/comment/generate")
def generate_entity_comment(body: EntityGenerateBody) -> dict[str, Any]:
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Нужно имя сущности")
    doc_id = _resolve_doc_id(body.document_id)
    doc_text = body.document_text or DOC_TEXTS.get(doc_id) or ""
    relations = store.neighbors_of(name, doc_id)

    from model.lmstudio import LMStudioExtractor
    from graph.websearch import search_web

    lm = LMStudioExtractor()
    if not lm.ping():
        raise HTTPException(
            status_code=503,
            detail="LM Studio выключен — запусти Gemma Local Server",
        )

    web_payload: dict[str, Any] = {"results": [], "error": None, "query": name}
    if body.web_search:
        rel_bits = []
        for r in relations[:8]:
            neighbor = r.get("neighbor") or ""
            rel = r.get("relation") or ""
            if neighbor:
                rel_bits.append(f"{rel} {neighbor}".strip())
        query = name
        if rel_bits:
            query = f"{name} {' '.join(rel_bits[:4])}"
        web_payload = search_web(query, max_results=3)

    has_doc = bool(doc_text.strip())
    has_rel = bool(relations)
    has_web = bool(web_payload.get("results"))
    if not has_doc and not has_rel and not has_web:
        raise HTTPException(status_code=400, detail="Нет текста, связей и результатов поиска")

    gem = lm.generate_entity_comment(
        doc_text,
        name,
        relations,
        web_results=web_payload.get("results") if has_web else None,
    )
    if not gem.get("ok"):
        raise HTTPException(
            status_code=502,
            detail=gem.get("error") or "Gemma не смогла сгенерировать комментарий",
        )

    comment = (gem.get("comment") or "").strip()
    if body.save and comment:
        _comments_for(doc_id)[name] = comment
        _sync_history_comments(body.history_id, doc_id)

    payload = _entity_payload(doc_id, name)
    payload["comment"] = comment or payload["comment"]
    payload["generated"] = True
    payload["web_results"] = web_payload.get("results") or []
    payload["web_error"] = web_payload.get("error")
    debug_err = gem.get("error")
    if web_payload.get("error"):
        debug_err = (debug_err + "; " if debug_err else "") + f"web: {web_payload['error']}"
    payload["debug"] = {
        "prompt_user": gem.get("prompt_user") or "",
        "raw_response": gem.get("raw_response") or "",
        "error": debug_err,
        "model": gem.get("model"),
    }
    return payload
