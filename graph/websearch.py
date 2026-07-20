"""Web search via DuckDuckGo (no API key) for Gemma prompt augmentation."""

from __future__ import annotations

from typing import Any


def search_web(query: str, max_results: int = 5) -> dict[str, Any]:
    """
    Return {results: [{title, url, snippet}], error: str|None, query: str}.
    Never raises — empty results + error on failure.
    """
    q = (query or "").strip()
    if not q:
        return {"results": [], "error": "Пустой поисковый запрос", "query": q}

    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS  # type: ignore
    except ImportError:
        return {
            "results": [],
            "error": "Нужен пакет ddgs (или duckduckgo-search)",
            "query": q,
        }

    results: list[dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(q, max_results=max_results))
        for item in raw[:max_results]:
            title = (item.get("title") or "").strip()
            url = (item.get("href") or item.get("link") or "").strip()
            snippet = (item.get("body") or item.get("snippet") or "").strip()
            if not (title or snippet or url):
                continue
            results.append({"title": title, "url": url, "snippet": snippet})
    except Exception as e:
        return {"results": [], "error": str(e), "query": q}

    return {"results": results, "error": None, "query": q}


def format_results_for_prompt(results: list[dict[str, str]]) -> str:
    if not results:
        return "(результатов поиска нет)"
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or "Без названия"
        url = r.get("url") or ""
        snippet = r.get("snippet") or ""
        lines.append(f"{i}. {title}\n   URL: {url}\n   {snippet}")
    return "\n".join(lines)
