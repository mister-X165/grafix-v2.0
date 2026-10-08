"""Дедупликация источников (ТЗ §67) и независимость (§24)."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from app.core.models import SearchResult, SourceMeta
from app.security.urlguard import canonicalize_url


def result_key(r: SearchResult) -> str:
    if r.doi:
        return "doi:" + r.doi.lower()
    if r.pmid:
        return "pmid:" + r.pmid
    return canonicalize_url(r.url)


def dedup_results(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    out: list[SearchResult] = []
    for r in results:
        k = result_key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def source_key(s: SourceMeta) -> str:
    if s.doi:
        return "doi:" + s.doi.lower()
    if s.pmid:
        return "pmid:" + s.pmid
    return canonicalize_url(s.url)


def dedup_sources(sources: list[SourceMeta]) -> list[SourceMeta]:
    """Убрать точные/canonical дубликаты и зеркала; при конфликте — тот, что получен."""
    by_key: dict[str, SourceMeta] = {}
    for s in sources:
        k = source_key(s)
        old = by_key.get(k)
        if old is None or (s.fetched and not old.fetched):
            by_key[k] = s
    return list(by_key.values())


# -------------------------------------------------- independence (§24) -----

_TEXT_SHINGLE_N = 8


def shingles(text: str, n: int = _TEXT_SHINGLE_N) -> set[str]:
    words = re.sub(r"[^a-zа-яё0-9\s]", " ", (text or "").lower()).split()
    return {" ".join(words[i:i + n]) for i in range(0, max(0, len(words) - n + 1), 2)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cluster_independent(sources: list[SourceMeta]) -> list[SourceMeta]:
    """Простая кластеризация репостов: очень похожий текст или same-domain+title →
    один independent_cluster_key. Внутри кластера независимым считается первый."""
    tagged: list[tuple[SourceMeta, set[str]] | None] = []
    clusters: list[list[SourceMeta]] = []
    for s in sources:
        sh = shingles(s.body_text[:6000]) if s.body_text else set()
        placed = False
        for cl in clusters:
            head = cl[0]
            same_host = _host(head.url) == _host(s.url)
            near_dup = bool(sh) and bool(head.body_text) and jaccard(
                sh, shingles(head.body_text[:6000])) > 0.6
            same_title = bool(s.title) and s.title.strip().lower() == head.title.strip().lower()
            if same_host and same_title or near_dup:
                cl.append(s)
                s.is_derivative = True
                s.original_source_url = head.url
                placed = True
                break
        if not placed:
            clusters.append([s])
    for cid, cl in enumerate(clusters):
        key = f"cluster_{cid}"
        for s in cl:
            s.independent_cluster_key = key
    return sources


def _host(url: str) -> str:
    h = (urlparse(url).hostname or "").lower()
    return h.removeprefix("www.")
