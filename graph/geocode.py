"""Map place search: Nominatim (OSM) + optional DeepSeek query expansion."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from model.lmstudio import _clip_text, _excerpt_around_entity

GEOCODE_SYSTEM = """Ты помогаешь поставить метку на карте (OpenStreetMap) для программы Grafix.

ЗАДАЧА: по имени сущности и фрагменту документа предложи геопоисковые запросы и, если уверен, координаты.

ФОРМАТ (строго JSON, без markdown):
{"queries":["запрос1","запрос2"],"places":[{"name":"место","lat":55.75,"lng":37.62,"confidence":0.0}],"note":"кратко"}

ПРАВИЛА:
1. queries — 1–4 строки для геокодера (город, завод, страна; на языке места и/или английском).
2. places — только если есть разумная уверенность в координатах (0..1). Иначе [].
3. Не выдумывай точные координаты, если в тексте нет явной геопривязки — лучше хорошие queries.
4. Если сущность не географическая — queries=[] и places=[].
"""


def nominatim_search(query: str, limit: int = 5) -> list[dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []
    params = urllib.parse.urlencode(
        {
            "q": q,
            "format": "json",
            "limit": str(limit),
            "addressdetails": "0",
        }
    )
    url = f"https://nominatim.openstreetmap.org/search?{params}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Grafix/0.1 (local knowledge-graph demo)",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for row in data if isinstance(data, list) else []:
        try:
            out.append(
                {
                    "display_name": row.get("display_name") or "",
                    "lat": float(row["lat"]),
                    "lng": float(row["lon"]),
                    "source": "nominatim",
                    "query": q,
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _parse_locate_json(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # salvage queries via comments-like pairs is weak; return empty
        return {"queries": [], "places": [], "note": ""}
    if not isinstance(data, dict):
        return {"queries": [], "places": [], "note": ""}
    queries: list[str] = []
    for q in data.get("queries") or []:
        s = str(q or "").strip()
        if s and s not in queries:
            queries.append(s)
    places: list[dict[str, Any]] = []
    for p in data.get("places") or []:
        if not isinstance(p, dict):
            continue
        try:
            lat = float(p.get("lat"))
            lng = float(p.get("lng"))
        except (TypeError, ValueError):
            continue
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            continue
        conf = p.get("confidence")
        try:
            conf_f = float(conf) if conf is not None else 0.5
        except (TypeError, ValueError):
            conf_f = 0.5
        places.append(
            {
                "display_name": str(p.get("name") or p.get("display_name") or "").strip(),
                "lat": lat,
                "lng": lng,
                "confidence": conf_f,
                "source": "deepseek",
            }
        )
    return {
        "queries": queries[:4],
        "places": places[:3],
        "note": str(data.get("note") or "").strip(),
    }


def deepseek_locate_hints(
    entity: str,
    document_text: str = "",
) -> dict[str, Any]:
    from model.openrouter import OpenRouterExtractor

    client = OpenRouterExtractor()
    if not client.ready:
        return {
            "ok": False,
            "error": "DeepSeek недоступен (нет OPENROUTER_API_KEY)",
            "queries": [],
            "places": [],
            "note": "",
            "raw": "",
        }
    name = (entity or "").strip()
    doc = _excerpt_around_entity(document_text or "", name, max_chars=2200) if name else ""
    if not doc:
        doc = _clip_text((document_text or "").strip(), 2200)
    user = (
        f"Сущность: {name or '(не указана)'}\n\n"
        f"Фрагмент документа:\n{doc or '(пусто)'}\n\n"
        "Верни JSON с queries и places для карты."
    )
    chat = client.chat(
        [
            {"role": "system", "content": GEOCODE_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.15,
        max_tokens=600,
    )
    if not chat.get("ok"):
        return {
            "ok": False,
            "error": chat.get("error") or "DeepSeek не ответил",
            "queries": [],
            "places": [],
            "note": "",
            "raw": chat.get("raw") or "",
        }
    parsed = _parse_locate_json(chat.get("raw") or "")
    return {
        "ok": True,
        "error": chat.get("error"),
        "queries": parsed["queries"],
        "places": parsed["places"],
        "note": parsed["note"],
        "raw": chat.get("raw") or "",
        "model": chat.get("model"),
    }


def locate_on_map(
    *,
    entity: str = "",
    query: str = "",
    document_text: str = "",
    use_deepseek: bool = True,
    limit: int = 5,
) -> dict[str, Any]:
    """DeepSeek hints + Nominatim search; returns ranked map hits."""
    entity = (entity or "").strip()
    query = (query or "").strip()
    seed = query or entity
    deep: dict[str, Any] = {
        "ok": False,
        "queries": [],
        "places": [],
        "note": "",
        "error": None,
    }
    if use_deepseek and (entity or query or document_text):
        deep = deepseek_locate_hints(entity or query, document_text)

    search_queries: list[str] = []
    for q in deep.get("queries") or []:
        if q and q not in search_queries:
            search_queries.append(q)
    if seed and seed not in search_queries:
        search_queries.insert(0, seed)
    if entity and entity != seed and entity not in search_queries:
        search_queries.append(entity)

    map_hits: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    for q in search_queries[:4]:
        for hit in nominatim_search(q, limit=limit):
            key = (round(hit["lat"], 4), round(hit["lng"], 4))
            if key in seen:
                continue
            seen.add(key)
            map_hits.append(hit)

    # DeepSeek direct places (lower priority unless high confidence)
    for p in deep.get("places") or []:
        key = (round(p["lat"], 4), round(p["lng"], 4))
        if key in seen:
            continue
        if float(p.get("confidence") or 0) < 0.55 and map_hits:
            continue
        seen.add(key)
        map_hits.append(
            {
                "display_name": p.get("display_name") or entity or query or "Точка",
                "lat": p["lat"],
                "lng": p["lng"],
                "source": "deepseek",
                "confidence": p.get("confidence"),
                "query": seed,
            }
        )

    return {
        "entity": entity,
        "query": seed,
        "queries_used": search_queries,
        "deepseek": {
            "ok": bool(deep.get("ok")),
            "note": deep.get("note") or "",
            "error": deep.get("error"),
            "model": deep.get("model"),
        },
        "results": map_hits[: max(1, limit)],
    }


GEO_GRAPH_SYSTEM = """Ты выделяешь ГЕОГРАФИЧЕСКИЙ слой графа знаний Grafix: только места и события, привязанные к местам.

ЗАДАЧА:
По тексту и списку троек верни локации для карты/глобуса.

ФОРМАТ (строго JSON):
{"locations":[{"entity":"Имя","kind":"place","queries":["поиск OSM"],"note":"кратко"}]}

kind:
- "place" — город, страна, завод, регион, объект на карте
- "event" — событие в месте (дефолт, война, открытие завода); в queries укажи место события

ПРАВИЛА:
1. Только сущности с реальной геопривязкой. Людей/абстракции без места — пропускай.
2. entity — каноническое имя как в графе (из троек), если возможно.
3. queries — 1–3 запроса для Nominatim (лучше с страной).
4. Не больше 18 локаций. JSON полный, без markdown.
5. Если географии нет: {"locations":[]}
"""


def _parse_geo_graph_json(raw: str) -> list[dict[str, Any]]:
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    items = data.get("locations") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        entity = str(item.get("entity") or item.get("name") or "").strip()
        if not entity:
            continue
        kind = str(item.get("kind") or "place").strip().lower()
        if kind not in {"place", "event"}:
            kind = "place"
        queries: list[str] = []
        for q in item.get("queries") or []:
            s = str(q or "").strip()
            if s and s not in queries:
                queries.append(s)
        if entity not in queries:
            queries.insert(0, entity)
        note = str(item.get("note") or "").strip()
        lat = lng = None
        try:
            if item.get("lat") is not None and item.get("lng") is not None:
                lat = float(item.get("lat"))
                lng = float(item.get("lng"))
                if not (-90 <= lat <= 90 and -180 <= lng <= 180):
                    lat = lng = None
        except (TypeError, ValueError):
            lat = lng = None
        out.append(
            {
                "entity": entity,
                "kind": kind,
                "queries": queries[:3],
                "note": note,
                "lat": lat,
                "lng": lng,
            }
        )
    return out[:18]


def _resolve_location_coords(loc: dict[str, Any]) -> dict[str, Any] | None:
    if loc.get("lat") is not None and loc.get("lng") is not None:
        return {
            "lat": float(loc["lat"]),
            "lng": float(loc["lng"]),
            "display_name": loc.get("note") or loc.get("entity") or "",
            "source": "deepseek",
        }
    for q in loc.get("queries") or []:
        hits = nominatim_search(q, limit=1)
        if hits:
            h = hits[0]
            return {
                "lat": h["lat"],
                "lng": h["lng"],
                "display_name": h.get("display_name") or "",
                "source": "nominatim",
                "query": q,
            }
    return None


def build_geo_graph(
    text: str,
    triples: list[dict[str, Any]] | None = None,
    engine: str | None = "deepseek-v4",
    *,
    reasoning: bool = False,
) -> dict[str, Any]:
    """DeepSeek: places/events → Nominatim coords → markers + geo links from triples."""
    from model.openrouter import is_deepseek_engine, openrouter_for_engine

    eng = (engine or "deepseek-v4").strip().lower()
    if not is_deepseek_engine(eng):
        eng = "deepseek-v4"
    client = openrouter_for_engine(eng)
    if not client.ready:
        return {
            "ok": False,
            "error": "DeepSeek недоступен (нет OPENROUTER_API_KEY)",
            "markers": [],
            "links": [],
            "raw": "",
            "engine": eng,
        }

    triples = triples or []
    triple_lines = []
    for t in triples[:60]:
        s = str((t or {}).get("subject") or "").strip()
        r = str((t or {}).get("relation") or "").strip()
        o = str((t or {}).get("object") or "").strip()
        if s and o:
            triple_lines.append(f"{s} —[{r}]→ {o}")
    doc = _clip_text((text or "").strip(), 5500)
    user = (
        "Выдели места и события с геопривязкой для карты и глобуса.\n\n"
        f"Тройки графа:\n" + ("\n".join(triple_lines) if triple_lines else "(нет)") + "\n\n"
        f"Текст:\n{doc or '(пусто)'}\n\n"
        'Ответ — только JSON: {"locations":[...]}'
    )
    chat = client.chat(
        [
            {"role": "system", "content": GEO_GRAPH_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=1800,
        reasoning=bool(reasoning),
    )
    if not chat.get("ok"):
        return {
            "ok": False,
            "error": chat.get("error") or "DeepSeek не ответил",
            "markers": [],
            "links": [],
            "raw": chat.get("raw") or "",
            "engine": eng,
            "model": chat.get("model"),
        }

    locations = _parse_geo_graph_json(chat.get("raw") or "")
    markers: list[dict[str, Any]] = []
    by_entity: dict[str, dict[str, Any]] = {}
    for loc in locations:
        resolved = _resolve_location_coords(loc)
        if not resolved:
            continue
        entity = loc["entity"]
        kind = loc["kind"]
        note_bits = [x for x in (loc.get("note"), resolved.get("display_name")) if x]
        mid = f"geo:{kind}:{entity}:{resolved['lat']:.4f}:{resolved['lng']:.4f}"
        marker = {
            "id": mid,
            "entity": entity,
            "lat": resolved["lat"],
            "lng": resolved["lng"],
            "note": " · ".join(note_bits)[:240],
            "kind": kind,
            "auto": True,
            "source": resolved.get("source") or "geo",
        }
        markers.append(marker)
        by_entity[entity.lower()] = marker

    links: list[dict[str, Any]] = []
    seen_links: set[tuple[str, str, str]] = set()
    for t in triples:
        s = str((t or {}).get("subject") or "").strip()
        o = str((t or {}).get("object") or "").strip()
        r = str((t or {}).get("relation") or "").strip() or "связан_с"
        kind = str((t or {}).get("kind") or "explicit")
        a = by_entity.get(s.lower())
        b = by_entity.get(o.lower())
        if not a or not b or a["id"] == b["id"]:
            continue
        key = (a["id"], b["id"], r)
        if key in seen_links:
            continue
        seen_links.add(key)
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

    return {
        "ok": bool(markers),
        "error": None if markers else "Не удалось привязать места к координатам",
        "markers": markers,
        "links": links,
        "raw": chat.get("raw") or "",
        "model": chat.get("model"),
        "engine": eng,
        "locations_found": len(locations),
    }
