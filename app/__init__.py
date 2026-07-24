"""Grafix FastAPI backend."""

from __future__ import annotations

import os
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


def _load_dotenv() -> None:
    """Load KEY=VALUE from .env; fill missing or empty env vars."""
    path = ROOT / ".env"
    if not path.is_file():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if not key:
                continue
            # Allow .env to replace blank env stubs (common Windows/shell pitfall)
            if key not in os.environ or not str(os.environ.get(key) or "").strip():
                os.environ[key] = val
    except OSError:
        pass


_load_dotenv()

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
DOC_MARKERS: dict[str, list[dict[str, Any]]] = {}


class AnalyzeBody(BaseModel):
    text: str
    document_id: str | None = None
    engine: str = "gemma"  # gemma | deepseek | deepseek-v4 | gigachat | microgpt | auto
    geo_graph: bool = True  # места/события → карта и глобус (DeepSeek)
    geo_engine: str = "deepseek-v4"  # deepseek | deepseek-v4
    reasoning: bool = False  # DeepSeek / GigaChat thinking mode
    language: str = "ru"  # ru | en | es | pt | fr | de | nl | sr | kk | tt | vi | tr


class TermAnalyzeBody(BaseModel):
    """Research one term/event via web search + network LLM, then extract a graph."""

    term: str
    document_id: str | None = None
    engine: str = "deepseek-v4"  # deepseek | deepseek-v4 | gigachat
    geo_graph: bool = True
    geo_engine: str = "deepseek-v4"
    reasoning: bool = False
    language: str = "ru"
    max_results: int = Field(default=8, ge=3, le=12)


class AppendAnalyzeBody(BaseModel):
    """Extract new facts from extra text and merge into an existing graph."""

    text: str
    document_id: str | None = None
    history_id: str | None = None
    engine: str = "gemma"
    reasoning: bool = False
    language: str = "ru"


class BridgeAnalyzeBody(BaseModel):
    """Link base graph/text with new text (bridge + new edges). DeepSeek or GigaChat."""

    text: str
    document_id: str | None = None
    history_id: str | None = None
    document_text: str | None = None
    engine: str = "deepseek-v4"  # deepseek | deepseek-v4 | gigachat
    reasoning: bool = False


class GeoGraphBody(BaseModel):
    document_id: str | None = None
    document_text: str | None = None
    triples: list[dict[str, Any]] | None = None
    replace: bool = True  # заменить авто-метки; ручные сохранить
    engine: str = "deepseek-v4"  # deepseek | deepseek-v4
    reasoning: bool = False


class AskBody(BaseModel):
    question: str
    document_id: str | None = None
    document_text: str | None = None
    web_search: bool = False
    engine: str = "gemma"  # gemma | deepseek | deepseek-v4 | … (text QA)
    reasoning: bool = False


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
    markers: list[dict[str, Any]] | None = None


class RenameGraphBody(BaseModel):
    title: str


class MarkersBody(BaseModel):
    document_id: str | None = None
    history_id: str | None = None
    markers: list[dict[str, Any]] = Field(default_factory=list)


class LocateBody(BaseModel):
    entity: str = ""
    query: str = ""
    document_id: str | None = None
    document_text: str | None = None
    use_deepseek: bool = True
    limit: int = 5


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
    engine: str = "gemma"  # gemma | deepseek | deepseek-v4
    reasoning: bool = False


def _resolve_doc_id(document_id: str | None) -> str:
    return document_id or new_document_id()


def _triples_from_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from model.triples import normalize_edge_kind, normalize_origin

    out: list[dict[str, Any]] = []
    for e in edges:
        s = (e.get("source") or "").strip()
        r = (e.get("relation") or e.get("label") or "").strip()
        o = (e.get("target") or "").strip()
        if s and o:
            kind = normalize_edge_kind(e.get("kind"))
            origin = normalize_origin(e.get("origin"))
            item: dict[str, Any] = {
                "subject": s,
                "relation": r,
                "object": o,
                "kind": kind,
                "origin": origin,
            }
            evidence = str(e.get("evidence") or "").strip()
            if evidence:
                item["evidence"] = evidence
            conf = e.get("confidence")
            if conf is not None and conf != "":
                try:
                    item["confidence"] = float(conf)
                except (TypeError, ValueError):
                    pass
            out.append(item)
    return out


def _triple_key(t: dict[str, Any]) -> tuple[str, str, str, str]:
    from model.triples import normalize_edge_kind

    return (
        str(t.get("subject") or "").strip().lower(),
        str(t.get("relation") or "").strip().lower(),
        str(t.get("object") or "").strip().lower(),
        normalize_edge_kind(t.get("kind")),
    )


def _coerce_triple_dict(t: Any, *, origin: str = "base") -> dict[str, Any] | None:
    from model.triples import Triple, normalize_edge_kind, normalize_origin

    if isinstance(t, Triple):
        d = t.as_dict()
    elif isinstance(t, dict):
        s = str(t.get("subject") or "").strip()
        r = str(t.get("relation") or "").strip()
        o = str(t.get("object") or "").strip()
        if not s or not o:
            return None
        d = {
            "subject": s,
            "relation": r,
            "object": o,
            "kind": normalize_edge_kind(t.get("kind")),
        }
        evidence = str(t.get("evidence") or "").strip()
        if evidence:
            d["evidence"] = evidence
        conf = t.get("confidence")
        if conf is not None and conf != "":
            try:
                d["confidence"] = float(conf)
            except (TypeError, ValueError):
                pass
        if t.get("origin"):
            d["origin"] = normalize_origin(t.get("origin"))
    else:
        return None
    if "origin" not in d:
        d["origin"] = normalize_origin(origin)
    else:
        d["origin"] = normalize_origin(d.get("origin"))
    d["kind"] = normalize_edge_kind(d.get("kind"))
    return d


def _merge_triples(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge triples; first occurrence wins (keeps base over append/bridge dupes)."""
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for src in (existing, incoming):
        for raw in src:
            item = _coerce_triple_dict(raw, origin="base")
            if not item:
                continue
            key = _triple_key(item)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def _append_document_text(base: str, extra: str) -> str:
    base = (base or "").strip()
    extra = (extra or "").strip()
    if not extra:
        return base
    if not base:
        return extra
    sep = "\n\n---\n\n"
    if extra in base:
        return base
    return f"{base}{sep}{extra}"


def _comments_for(doc_id: str) -> dict[str, str]:
    return DOC_COMMENTS.setdefault(doc_id, {})


def _markers_for(doc_id: str) -> list[dict[str, Any]]:
    return DOC_MARKERS.setdefault(doc_id, [])


def _normalize_markers(raw: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        try:
            lat = float(item.get("lat"))
            lng = float(item.get("lng"))
        except (TypeError, ValueError):
            continue
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            continue
        entity = str(item.get("entity") or item.get("name") or "").strip()
        mid = str(item.get("id") or "").strip() or f"{entity}:{lat:.5f}:{lng:.5f}"
        if mid in seen:
            continue
        seen.add(mid)
        note = str(item.get("note") or "").strip()
        kind = str(item.get("kind") or "place").strip().lower()
        if kind not in {"place", "event", "manual"}:
            kind = "place"
        out.append(
            {
                "id": mid,
                "entity": entity or "Точка",
                "lat": lat,
                "lng": lng,
                "note": note,
                "kind": kind,
                "auto": bool(item.get("auto")),
            }
        )
    return out


def _geo_links_from_markers_and_triples(
    markers: list[dict[str, Any]],
    triples: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    by_entity = {str(m.get("entity") or "").strip().lower(): m for m in markers if m.get("entity")}
    links: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for t in triples or []:
        s = str((t or {}).get("subject") or "").strip()
        o = str((t or {}).get("object") or "").strip()
        r = str((t or {}).get("relation") or "").strip() or "связан_с"
        kind = str((t or {}).get("kind") or "explicit")
        a = by_entity.get(s.lower())
        b = by_entity.get(o.lower())
        if not a or not b or a.get("id") == b.get("id"):
            continue
        key = (a["id"], b["id"], r)
        if key in seen:
            continue
        seen.add(key)
        links.append(
            {
                "source": a["entity"],
                "target": b["entity"],
                "source_id": a["id"],
                "target_id": b["id"],
                "relation": r,
                "kind": kind,
            }
        )
    return links


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
        markers=list(_markers_for(doc_id)),
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
    markers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if comments is not None:
        DOC_COMMENTS[document_id] = dict(comments)
    if markers is not None:
        DOC_MARKERS[document_id] = _normalize_markers(markers)
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
        markers=list(_markers_for(document_id)),
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
    from model.gigachat import GigaChatExtractor
    from model.lmstudio import LMStudioExtractor
    from model.openrouter import OpenRouterExtractor

    lm = LMStudioExtractor()
    ready = lm.ping()
    or_client = OpenRouterExtractor()
    or_ready = or_client.ready
    gc = GigaChatExtractor()
    gc_ready = bool(gc.credentials)
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
        "openrouter": {
            "ready": or_ready,
            "base_url": or_client.base_url,
            "model": or_client.resolved_model if or_ready else None,
            "has_api_key": bool(or_client.api_key),
        },
        "gigachat": {
            "configured": gc_ready,
            "base_url": gc.base_url,
            "model": gc.model if gc_ready else None,
            "scope": gc.scope if gc_ready else None,
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
    result = extractor.extract(
        text,
        engine=body.engine or "gemma",
        reasoning=bool(body.reasoning),
        language=body.language,
    )
    triples = []
    for t in result.get("triples") or []:
        item = _coerce_triple_dict(t, origin="base")
        if item:
            item["origin"] = "base"
            triples.append(item)
    result = {**result, "triples": triples}
    snap = store.upsert_triples(doc_id, text, triples)
    DOC_TEXTS[doc_id] = text
    comments = result.get("comments")
    if isinstance(comments, dict):
        DOC_COMMENTS[doc_id] = {str(k): str(v) for k, v in comments.items() if str(k).strip() and str(v).strip()}
    elif (body.engine or "").strip().lower() in {"deepseek", "deepseek-v4"}:
        # Fresh DeepSeek run without comments → clear stale ones
        DOC_COMMENTS[doc_id] = {}

    geo_meta: dict[str, Any] = {"ok": False, "error": None, "links": []}
    engine_l = (result.get("engine") or body.engine or "").strip().lower()
    if body.geo_graph and result.get("triples") and engine_l in {"deepseek", "deepseek-v4"}:
        from graph.geocode import build_geo_graph

        geo_eng = (body.geo_engine or "deepseek-v4").strip().lower()
        if geo_eng not in {"deepseek", "deepseek-v4"}:
            geo_eng = "deepseek-v4"
        geo = build_geo_graph(
            text, result["triples"], engine=geo_eng, reasoning=bool(body.reasoning)
        )
        geo_meta = {
            "ok": bool(geo.get("ok")),
            "error": geo.get("error"),
            "links": geo.get("links") or [],
            "locations_found": geo.get("locations_found"),
            "engine": geo.get("engine") or geo_eng,
            "model": geo.get("model"),
        }
        if geo.get("markers"):
            # keep manually placed markers, replace previous auto geo markers
            kept = [m for m in _markers_for(doc_id) if not m.get("auto")]
            DOC_MARKERS[doc_id] = _normalize_markers(kept + list(geo["markers"]))
        debug = result.get("debug") or {}
        debug["geo_raw"] = geo.get("raw") or ""
        debug["geo_error"] = geo.get("error")
        result["debug"] = debug
        if geo.get("markers"):
            n = len(geo["markers"])
            hint = result.get("hint") or ""
            extra = f"гео: {n} мест/событий, связей на карте: {len(geo.get('links') or [])}"
            result["hint"] = f"{hint} · {extra}" if hint else extra

    saved = _persist_snapshot(
        document_id=doc_id,
        text=text,
        triples=result["triples"],
        nodes=snap.nodes,
        edges=snap.edges,
        engine=result.get("engine", body.engine),
        source=result.get("source"),
        comments=DOC_COMMENTS.get(doc_id),
        markers=DOC_MARKERS.get(doc_id),
    )
    markers = list(_markers_for(doc_id))
    links = geo_meta.get("links") or _geo_links_from_markers_and_triples(
        markers, result["triples"]
    )
    return {
        "document_id": doc_id,
        "text": text,
        "triples": result["triples"],
        "entities_meta": result.get("entities_meta") or [],
        "contradictions": result.get("contradictions") or [],
        "comments": DOC_COMMENTS.get(doc_id, {}),
        "markers": markers,
        "geo_links": links,
        "geo": geo_meta,
        "source": result["source"],
        "engine": result.get("engine", body.engine),
        "model_ready": result["model_ready"],
        "lm_ready": result.get("lm_ready"),
        "lm_model": result.get("lm_model"),
        "openrouter_ready": result.get("openrouter_ready"),
        "openrouter_model": result.get("openrouter_model"),
        "hint": result.get("hint"),
        "debug": result.get("debug") or {},
        "nodes": snap.nodes,
        "edges": snap.edges,
        "history_id": saved["id"],
        "history": saved,
    }


def _network_research_client(engine: str):
    eng = (engine or "").strip().lower()
    if eng in {"deepseek", "deepseek-v4"}:
        from model.openrouter import openrouter_for_engine

        client = openrouter_for_engine(eng)
        if not client.ready:
            raise HTTPException(
                status_code=503,
                detail="Нет OPENROUTER_API_KEY — задай ключ OpenRouter для DeepSeek",
            )
        label = "DeepSeek V4 Pro" if eng == "deepseek-v4" else "DeepSeek 3.2"
        return client, eng, label
    if eng == "gigachat":
        from model.gigachat import GigaChatExtractor

        client = GigaChatExtractor()
        if not client.ready:
            raise HTTPException(
                status_code=503,
                detail="Нет GIGACHAT_CREDENTIALS — задай Auth key GigaChat в .env",
            )
        return client, eng, "GigaChat"
    raise HTTPException(
        status_code=400,
        detail="Исследование термина доступно только для сетевых моделей: DeepSeek или GigaChat",
    )


def _research_term_dossier(
    client: Any,
    term: str,
    web_results: list[dict[str, Any]],
    *,
    reasoning: bool,
    language: str,
) -> dict[str, Any]:
    from graph.websearch import format_results_for_prompt
    from model.locale import language_instruction
    from model.lmstudio import TERM_RESEARCH_SYSTEM

    lang_block = language_instruction(language)
    system = TERM_RESEARCH_SYSTEM + "\n\n" + lang_block
    user = (
        f"Термин / слово / событие для исследования:\n{term}\n\n"
        f"Результаты веб-поиска:\n{format_results_for_prompt(web_results)}\n\n"
        "Напиши связный справочный текст для извлечения графа знаний."
    )
    chat = client.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.15,
        max_tokens=3500,
        reasoning=reasoning,
    )
    raw = (chat.get("raw") or "").strip()
    return {
        "text": raw,
        "ok": bool(chat.get("ok") and raw),
        "error": chat.get("error"),
        "model": chat.get("model"),
        "raw_response": chat.get("raw") or "",
        "prompt_user": user,
        "reasoning": chat.get("reasoning") or "",
    }


@app.post("/api/analyze/term")
def analyze_term(body: TermAnalyzeBody) -> dict[str, Any]:
    """Web-search a term/event, write a dossier with a network LLM, extract a graph."""
    from graph.websearch import format_results_for_prompt, search_web

    term = (body.term or "").strip()
    if not term:
        raise HTTPException(status_code=400, detail="Укажите термин, слово или событие")

    client, engine, label = _network_research_client(body.engine or "deepseek-v4")
    web_payload = search_web(term, max_results=int(body.max_results or 8))
    web_results = list(web_payload.get("results") or [])
    if not web_results:
        detail = web_payload.get("error") or "Поиск ничего не нашёл"
        raise HTTPException(
            status_code=502,
            detail=f"Не удалось найти информацию о «{term}»: {detail}",
        )

    dossier = _research_term_dossier(
        client,
        term,
        web_results,
        reasoning=bool(body.reasoning),
        language=body.language or "ru",
    )
    text = (dossier.get("text") or "").strip()
    if not text:
        # Fallback: use search snippets as the document for extract
        text = (
            f"Справка по запросу «{term}» (по результатам поиска):\n\n"
            + format_results_for_prompt(web_results)
        )

    out = analyze(
        AnalyzeBody(
            text=text,
            document_id=body.document_id,
            engine=engine,
            geo_graph=bool(body.geo_graph),
            geo_engine=body.geo_engine or "deepseek-v4",
            reasoning=bool(body.reasoning),
            language=body.language or "ru",
        )
    )
    debug = dict(out.get("debug") or {})
    debug["term"] = term
    debug["research_raw"] = dossier.get("raw_response") or ""
    debug["research_error"] = dossier.get("error")
    debug["research_prompt"] = dossier.get("prompt_user") or ""
    debug["research_reasoning"] = dossier.get("reasoning") or ""
    debug["web_query"] = web_payload.get("query") or term
    debug["web_error"] = web_payload.get("error")
    out["debug"] = debug
    out["mode"] = "term"
    out["term"] = term
    out["web_sources"] = web_results
    out["research_model"] = dossier.get("model") or label
    hint = out.get("hint") or ""
    extra = f"термин «{term}»: поиск {len(web_results)} ист., досье → граф ({label})"
    out["hint"] = f"{hint} · {extra}" if hint else extra
    return out


@app.post("/api/analyze/append")
def analyze_append(body: AppendAnalyzeBody) -> dict[str, Any]:
    """Extract triples from new text and add them to an existing document graph."""
    new_text = (body.text or "").strip()
    if not new_text:
        raise HTTPException(status_code=400, detail="Пустой текст для дополнения")

    doc_id = _resolve_doc_id(body.document_id)
    snap = store.get_graph(doc_id)
    existing = _triples_from_edges(snap.edges)
    if not existing and not DOC_TEXTS.get(doc_id):
        raise HTTPException(
            status_code=400,
            detail="Сначала постройте граф («Анализировать»), затем дополняйте",
        )

    result = extractor.extract(
        new_text,
        engine=body.engine or "gemma",
        reasoning=bool(body.reasoning),
        language=body.language,
    )
    incoming: list[dict[str, Any]] = []
    for t in result.get("triples") or []:
        item = _coerce_triple_dict(t, origin="append")
        if item:
            item["origin"] = "append"
            incoming.append(item)

    merged = _merge_triples(existing, incoming)
    combined_text = _append_document_text(DOC_TEXTS.get(doc_id, ""), new_text)
    DOC_TEXTS[doc_id] = combined_text
    snap2 = store.upsert_triples(doc_id, combined_text, merged)

    hist_id = (body.history_id or "").strip() or None
    saved = _persist_snapshot(
        document_id=doc_id,
        text=combined_text,
        triples=merged,
        nodes=snap2.nodes,
        edges=snap2.edges,
        engine=result.get("engine", body.engine),
        source=result.get("source"),
        history_id=hist_id,
        comments=DOC_COMMENTS.get(doc_id),
        markers=DOC_MARKERS.get(doc_id),
    )
    added = len(incoming)
    # Count how many actually new vs merged-away
    before = {_triple_key(t) for t in existing}
    truly_new = sum(1 for t in incoming if _triple_key(t) not in before)
    return {
        "document_id": doc_id,
        "text": combined_text,
        "triples": merged,
        "added_triples": incoming,
        "added_count": truly_new,
        "extracted_count": added,
        "entities_meta": result.get("entities_meta") or [],
        "contradictions": result.get("contradictions") or [],
        "comments": DOC_COMMENTS.get(doc_id, {}),
        "markers": list(_markers_for(doc_id)),
        "source": result.get("source"),
        "engine": result.get("engine", body.engine),
        "model_ready": result.get("model_ready"),
        "lm_ready": result.get("lm_ready"),
        "lm_model": result.get("lm_model"),
        "openrouter_ready": result.get("openrouter_ready"),
        "openrouter_model": result.get("openrouter_model"),
        "hint": (
            f"дополнено: +{truly_new} новых рёбер (извлечено {added})"
            + (f" · {result.get('hint')}" if result.get("hint") else "")
        ),
        "debug": result.get("debug") or {},
        "nodes": snap2.nodes,
        "edges": snap2.edges,
        "history_id": saved["id"],
        "history": saved,
        "mode": "append",
    }


@app.post("/api/analyze/bridge")
def analyze_bridge(body: BridgeAnalyzeBody) -> dict[str, Any]:
    """DeepSeek / GigaChat: analyze base+new texts and add bridging edges onto the workspace."""
    from model.gigachat import GigaChatExtractor
    from model.openrouter import is_deepseek_engine, openrouter_for_engine

    new_text = (body.text or "").strip()
    if not new_text:
        raise HTTPException(status_code=400, detail="Пустой новый текст для склейки")

    engine = (body.engine or "deepseek-v4").strip().lower()
    use_gigachat = engine == "gigachat"
    if not use_gigachat and not is_deepseek_engine(engine):
        raise HTTPException(
            status_code=400,
            detail="Склейка доступна через DeepSeek (deepseek / deepseek-v4) или GigaChat",
        )

    doc_id = _resolve_doc_id(body.document_id)
    snap = store.get_graph(doc_id)
    existing = _triples_from_edges(snap.edges)
    base_text = (
        (body.document_text if body.document_text is not None else None)
        or DOC_TEXTS.get(doc_id)
        or ""
    ).strip()
    if not existing and not base_text:
        raise HTTPException(
            status_code=400,
            detail="Нужен готовый граф или базовый текст документа",
        )

    if use_gigachat:
        client = GigaChatExtractor()
        if not client.ready:
            raise HTTPException(
                status_code=503,
                detail="GigaChat недоступен (проверьте GIGACHAT_CREDENTIALS)",
            )
        source = "gigachat-bridge"
        engine_label = "GigaChat"
    else:
        client = openrouter_for_engine(engine)
        if not client.ready:
            raise HTTPException(
                status_code=503,
                detail="OpenRouter / DeepSeek недоступен (проверьте OPENROUTER_API_KEY)",
            )
        source = "openrouter-bridge"
        engine_label = "DeepSeek"

    bridged = client.bridge_texts(
        base_text,
        new_text,
        existing_triples=existing,
        reasoning=bool(body.reasoning),
    )
    if not bridged.get("ok") and bridged.get("error"):
        raise HTTPException(
            status_code=502,
            detail=f"{engine_label} склейка не удалась: {bridged.get('error')}",
        )

    incoming: list[dict[str, Any]] = []
    for t in bridged.get("new_triples") or []:
        item = _coerce_triple_dict(t, origin="append")
        if item:
            item["origin"] = "append"
            incoming.append(item)
    for t in bridged.get("bridge_triples") or []:
        item = _coerce_triple_dict(t, origin="bridge")
        if item:
            item["origin"] = "bridge"
            incoming.append(item)

    merged = _merge_triples(existing, incoming)
    combined_text = _append_document_text(base_text, new_text)
    DOC_TEXTS[doc_id] = combined_text
    snap2 = store.upsert_triples(doc_id, combined_text, merged)

    hist_id = (body.history_id or "").strip() or None
    saved = _persist_snapshot(
        document_id=doc_id,
        text=combined_text,
        triples=merged,
        nodes=snap2.nodes,
        edges=snap2.edges,
        engine=engine,
        source=source,
        history_id=hist_id,
        comments=DOC_COMMENTS.get(doc_id),
        markers=DOC_MARKERS.get(doc_id),
    )
    before = {_triple_key(t) for t in existing}
    new_n = sum(
        1
        for t in incoming
        if t.get("origin") == "append" and _triple_key(t) not in before
    )
    bridge_n = sum(
        1
        for t in incoming
        if t.get("origin") == "bridge" and _triple_key(t) not in before
    )
    out: dict[str, Any] = {
        "document_id": doc_id,
        "text": combined_text,
        "triples": merged,
        "added_triples": incoming,
        "new_count": new_n,
        "bridge_count": bridge_n,
        "comments": DOC_COMMENTS.get(doc_id, {}),
        "markers": list(_markers_for(doc_id)),
        "source": source,
        "engine": engine,
        "model_ready": True,
        "hint": f"склейка {engine_label}: +{new_n} новых, +{bridge_n} мостов",
        "debug": {
            "raw_response": bridged.get("raw_response") or "",
            "parsed_json": bridged.get("parsed_json"),
            "error": bridged.get("error"),
            "reasoning": bridged.get("reasoning") or "",
            "prompt_user": bridged.get("prompt_user") or "",
        },
        "nodes": snap2.nodes,
        "edges": snap2.edges,
        "history_id": saved["id"],
        "history": saved,
        "mode": "bridge",
    }
    if use_gigachat:
        out["gigachat_ready"] = True
        out["gigachat_model"] = bridged.get("model") or client.resolved_model
    else:
        out["openrouter_ready"] = True
        out["openrouter_model"] = bridged.get("model") or client.resolved_model
    return out


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
        engine=body.engine or "gemma",
        reasoning=bool(body.reasoning),
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
    if body.markers is not None:
        DOC_MARKERS[doc_id] = _normalize_markers(body.markers)
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
        markers=DOC_MARKERS.get(doc_id),
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
        "markers": saved.get("markers") or [],
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
        DOC_MARKERS[doc_id] = _normalize_markers(item.get("markers") or [])
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
        "markers": item.get("markers") or DOC_MARKERS.get(doc_id, []),
        "history": item,
    }


@app.get("/api/markers")
def get_markers(document_id: str | None = Query(default=None)) -> dict[str, Any]:
    doc_id = _resolve_doc_id(document_id)
    return {"document_id": doc_id, "markers": list(_markers_for(doc_id))}


@app.put("/api/markers")
def put_markers(body: MarkersBody) -> dict[str, Any]:
    doc_id = _resolve_doc_id(body.document_id)
    markers = _normalize_markers(body.markers)
    DOC_MARKERS[doc_id] = markers
    if body.history_id:
        item = history.get(body.history_id)
        if item:
            history.save(
                text=item.get("text") or DOC_TEXTS.get(doc_id, ""),
                triples=item.get("triples") or _triples_from_edges(item.get("edges") or []),
                nodes=item.get("nodes") or [],
                edges=item.get("edges") or [],
                document_id=doc_id,
                engine=item.get("engine"),
                source=item.get("source"),
                title=item.get("title"),
                graph_id=body.history_id,
                comments=dict(_comments_for(doc_id)),
                markers=markers,
            )
    return {"document_id": doc_id, "markers": markers, "count": len(markers)}


@app.post("/api/geo/graph")
def generate_geo_graph(body: GeoGraphBody) -> dict[str, Any]:
    """Build place/event markers + links for map/globe from text & triples."""
    from graph.geocode import build_geo_graph

    doc_id = _resolve_doc_id(body.document_id)
    text = body.document_text if body.document_text is not None else DOC_TEXTS.get(doc_id, "")
    triples = body.triples
    if triples is None:
        snap = store.get_graph(doc_id)
        triples = _triples_from_edges(snap.edges)
    geo_eng = (body.engine or "deepseek-v4").strip().lower()
    if geo_eng not in {"deepseek", "deepseek-v4"}:
        geo_eng = "deepseek-v4"
    geo = build_geo_graph(
        text or "", triples or [], engine=geo_eng, reasoning=bool(body.reasoning)
    )
    if not geo.get("ok") and not geo.get("markers"):
        raise HTTPException(
            status_code=502,
            detail=geo.get("error") or "Не удалось построить географ",
        )
    if body.replace:
        kept = [m for m in _markers_for(doc_id) if not m.get("auto")]
        DOC_MARKERS[doc_id] = _normalize_markers(kept + list(geo.get("markers") or []))
    else:
        DOC_MARKERS[doc_id] = _normalize_markers(
            list(_markers_for(doc_id)) + list(geo.get("markers") or [])
        )
    markers = list(_markers_for(doc_id))
    links = geo.get("links") or _geo_links_from_markers_and_triples(markers, triples)
    return {
        "document_id": doc_id,
        "markers": markers,
        "geo_links": links,
        "ok": True,
        "error": geo.get("error"),
        "model": geo.get("model"),
        "engine": geo.get("engine") or geo_eng,
        "locations_found": geo.get("locations_found"),
        "debug": {"geo_raw": geo.get("raw") or "", "geo_error": geo.get("error")},
    }


@app.get("/api/geocode")
def geocode_place(
    q: str = Query(..., min_length=1),
    limit: int = Query(default=5, ge=1, le=10),
) -> dict[str, Any]:
    """Nominatim-only geocoder."""
    from graph.geocode import nominatim_search

    query = (q or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Пустой запрос")
    results = nominatim_search(query, limit=limit)
    return {"query": query, "results": results}


@app.post("/api/geocode/locate")
def geocode_locate(body: LocateBody) -> dict[str, Any]:
    """DeepSeek + OSM map search for entity / free-text query."""
    from graph.geocode import locate_on_map

    entity = (body.entity or "").strip()
    query = (body.query or "").strip()
    if not entity and not query:
        raise HTTPException(status_code=400, detail="Нужна сущность или поисковый запрос")
    doc_id = body.document_id
    doc_text = body.document_text
    if doc_text is None and doc_id:
        doc_text = DOC_TEXTS.get(doc_id, "")
    limit = max(1, min(int(body.limit or 5), 10))
    return locate_on_map(
        entity=entity,
        query=query,
        document_text=doc_text or "",
        use_deepseek=bool(body.use_deepseek),
        limit=limit,
    )


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

    from graph.websearch import search_web

    engine = (body.engine or "gemma").strip().lower()
    if engine in {"deepseek", "deepseek-v4"}:
        from model.openrouter import openrouter_for_engine

        client = openrouter_for_engine(engine)
        if not client.ready:
            raise HTTPException(
                status_code=503,
                detail="Нет OPENROUTER_API_KEY — задай ключ OpenRouter для DeepSeek",
            )
        label = "DeepSeek V4 Pro" if engine == "deepseek-v4" else "DeepSeek 3.2"
    elif engine == "gigachat":
        from model.gigachat import GigaChatExtractor

        client = GigaChatExtractor()
        if not client.ready:
            raise HTTPException(
                status_code=503,
                detail="Нет GIGACHAT_CREDENTIALS — задай Auth key GigaChat в .env",
            )
        label = "GigaChat"
    else:
        from model.lmstudio import LMStudioExtractor

        client = LMStudioExtractor()
        if not client.ping():
            raise HTTPException(
                status_code=503,
                detail="LM Studio выключен — запусти Gemma Local Server или выбери DeepSeek / GigaChat",
            )
        label = "Gemma"

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

    gem = client.generate_entity_comment(
        doc_text,
        name,
        relations,
        web_results=web_payload.get("results") if has_web else None,
        **(
            {"reasoning": bool(body.reasoning)}
            if engine in {"deepseek", "deepseek-v4", "gigachat"}
            else {}
        ),
    )
    if not gem.get("ok"):
        raise HTTPException(
            status_code=502,
            detail=gem.get("error") or f"{label} не смогла сгенерировать комментарий",
        )

    comment = (gem.get("comment") or "").strip()
    if body.save and comment:
        _comments_for(doc_id)[name] = comment
        _sync_history_comments(body.history_id, doc_id)

    payload = _entity_payload(doc_id, name)
    payload["comment"] = comment or payload["comment"]
    payload["generated"] = True
    payload["engine"] = (
        engine if engine in {"deepseek", "deepseek-v4", "gigachat"} else "gemma"
    )
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
