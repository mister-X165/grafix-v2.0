"""Упрощённый source graph (ТЗ §68): original vs derivative, cites.

Интерфейс рассчитан на расширение до полноценного графа
(source → cites → source, source → derived_from → source).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.models import SourceMeta


@dataclass
class SourceGraph:
    nodes: dict[str, str] = field(default_factory=dict)          # source_id -> url
    derived_from: dict[str, str] = field(default_factory=dict)   # id -> id оригинала
    cites: dict[str, list[str]] = field(default_factory=dict)    # id -> [ids]

    def build(self, sources: list[SourceMeta]) -> "SourceGraph":
        by_url = {}
        for s in sources:
            self.nodes[s.id] = s.url
            by_url.setdefault(s.url, s.id)
            if s.original_source_url and s.original_source_url in by_url:
                self.derived_from[s.id] = by_url[s.original_source_url]
        # citations: DOI-пересечения между текстами источников (упрощённо)
        dois = {s.doi.lower(): s.id for s in sources if s.doi}
        for s in sources:
            refs = []
            body = (s.body_text or "")[:20000].lower()
            for doi, sid in dois.items():
                if sid != s.id and doi in body:
                    refs.append(sid)
            if refs:
                self.cites[s.id] = refs
        return self

    def independent_count(self) -> int:
        """Число корней независимости (кластеров), а не число сайтов (§24)."""
        roots = set()
        for nid in self.nodes:
            cur, seen = nid, set()
            while cur in self.derived_from and cur not in seen:
                seen.add(cur)
                cur = self.derived_from[cur]
            roots.add(cur)
        return len(roots)
