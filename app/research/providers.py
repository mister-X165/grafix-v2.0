"""Search providers (ТЗ §19, §64–§66).

Единый интерфейс SearchProvider; реализации: PubMed (E-utilities), Europe PMC,
Crossref, web (DuckDuckGo без ключа — паттерн graph/websearch.py из Grafix),
поиск по доверенным доменам. Ни один метод не бросает наружу: ошибка провайдера
фиксируется в SearchResult.error/логе и не останавливает pipeline (§50).

API-ключи — только из окружения (cfg.api_key), никогда в коде (§19).
"""

from __future__ import annotations

import asyncio
import logging
import urllib.parse
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from app.config import AppConfig, get_config
from app.core.models import AnalysisMode, SearchResult
from app.security.urlguard import validate_url

log = logging.getLogger(__name__)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FactCheckLocal/1.0"


class SearchProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def search(self, query: str, *, max_results: int = 8,
                     mode: AnalysisMode = AnalysisMode.RESEARCH) -> list[SearchResult]:
        """Всегда возвращает список (возможно пустой). Никогда не бросает."""

    def _safe_url(self, url: str) -> Optional[str]:
        try:
            return validate_url(url)
        except Exception as e:
            log.warning("%s: отброшен небезопасный URL %r (%s)", self.name, url[:80], e)
            return None


class PubmedProvider(SearchProvider):
    """NCBI E-utilities esearch/esummary. Ключ опционален: NCBI_API_KEY."""

    name = "pubmed"
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, cfg: Optional[AppConfig] = None) -> None:
        self.cfg = cfg or get_config()

    async def search(self, query: str, *, max_results: int = 8,
                     mode: AnalysisMode = AnalysisMode.RESEARCH) -> list[SearchResult]:
        params = {"db": "pubmed", "retmode": "json", "retmax": max_results,
                  "term": query}
        key = self.cfg.api_key("NCBI_API_KEY")
        if key:
            params["api_key"] = key
        try:
            async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": UA}) as c:
                r = await c.get(f"{self.BASE}/esearch.fcgi", params=params)
                r.raise_for_status()
                ids = r.json().get("esearchresult", {}).get("idlist", [])[:max_results]
                if not ids:
                    return []
                r2 = await c.get(f"{self.BASE}/esummary.fcgi",
                                 params={"db": "pubmed", "retmode": "json", "id": ",".join(ids)})
                r2.raise_for_status()
                res = r2.json().get("result", {})
                out: list[SearchResult] = []
                for uid in res.get("uids", []):
                    item = res.get(uid, {})
                    doi = next((a["value"] for a in item.get("articleids", [])
                                if a.get("idtype") == "doi"), "")
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"
                    out.append(SearchResult(
                        title=item.get("title", "")[:300], url=url, provider=self.name,
                        publication_date=item.get("pubdate", ""), pmid=uid, doi=doi,
                        source_type="journal_article",
                        metadata={"authors": "; ".join(a.get("name", "") for a in item.get("authors", [])[:8])},
                    ))
                return out
        except Exception as e:
            # graceful degradation (§50): PubMed упал — продолжаем без него
            log.warning("PubMed search failed (%s): %s", query[:60], e)
            return []


class EuropePMCProvider(SearchProvider):
    name = "europepmc"
    API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    async def search(self, query: str, *, max_results: int = 8,
                     mode: AnalysisMode = AnalysisMode.RESEARCH) -> list[SearchResult]:
        params = {"query": query, "format": "json", "pageSize": max_results,
                  "resultType": "core"}
        try:
            async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": UA}) as c:
                r = await c.get(self.API, params=params)
                r.raise_for_status()
                out: list[SearchResult] = []
                for it in r.json().get("resultList", {}).get("result", [])[:max_results]:
                    pmcid = it.get("pmcid", "")
                    pubmid = it.get("pmid", "")
                    doi = it.get("doi", "")
                    url = (f"https://europepmc.org/article/PMC/{pmcid}" if pmcid else
                           (f"https://pubmed.ncbi.nlm.nih.gov/{pubmid}/" if pubmid else
                            (f"https://doi.org/{doi}" if doi else "")))
                    if not url:
                        continue
                    st = (it.get("pubtype") or "").lower()
                    src_type = ("systematic_review" if "systematic" in st or "meta" in st
                                else "journal_article")
                    out.append(SearchResult(
                        title=it.get("title", "")[:300], url=url, provider=self.name,
                        publication_date=it.get("pubYear", ""), doi=doi, pmid=pubmid,
                        source_type=src_type,
                        snippet=(it.get("abstractText") or "")[:600],
                        metadata={"journal": it.get("journalTitle", "")},
                    ))
                return out
        except Exception as e:
            log.warning("EuropePMC search failed: %s", e)
            return []


class CrossrefProvider(SearchProvider):
    name = "crossref"
    API = "https://api.crossref.org/works"

    def __init__(self, cfg: Optional[AppConfig] = None) -> None:
        self.cfg = cfg or get_config()

    async def search(self, query: str, *, max_results: int = 8,
                     mode: AnalysisMode = AnalysisMode.RESEARCH) -> list[SearchResult]:
        params = {"query.bibliographic": query, "rows": max_results}
        mailto = self.cfg.api_key("CROSSREF_MAILTO")   # вежливый пул, опционально
        if mailto:
            params["mailto"] = mailto
        try:
            async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": UA}) as c:
                r = await c.get(self.API, params=params)
                r.raise_for_status()
                out: list[SearchResult] = []
                for it in r.json().get("message", {}).get("items", [])[:max_results]:
                    doi = it.get("DOI", "")
                    title = (it.get("title") or [""])[0][:300]
                    year = ""
                    parts = (it.get("published", {}) or {}).get("date-parts", [[]])
                    if parts and parts[0]:
                        year = str(parts[0][0])
                    url = f"https://doi.org/{doi}" if doi else (it.get("URL") or "")
                    if not url:
                        continue
                    out.append(SearchResult(
                        title=title, url=url, provider=self.name, doi=doi,
                        publication_date=year,
                        source_type=({"review" if "review" in (it.get("type") or "") else
                                      "journal_article"}),
                        metadata={"container": (it.get("container-title") or [""])[0],
                                  "publisher": it.get("publisher", "")},
                    ))
                return out
        except Exception as e:
            log.warning("Crossref search failed: %s", e)
            return []


class WebSearchProvider(SearchProvider):
    """DuckDuckGo HTML (без ключей). Паттерн grafix/graph/websearch.py: never raises."""

    name = "web"

    async def search(self, query: str, *, max_results: int = 8,
                     mode: AnalysisMode = AnalysisMode.RESEARCH) -> list[SearchResult]:
        try:
            async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": UA},
                                         follow_redirects=True) as c:
                r = await c.post("https://html.duckduckgo.com/html/",
                                 data={"q": query, "kl": "wt-wt"})
                if r.status_code != 200:
                    return []
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(r.text, "lxml")
                out: list[SearchResult] = []
                for res in soup.select(".result, .web-result")[:max_results * 2]:
                    a = res.select_one("a.result__a, a")
                    if not a or not a.get("href"):
                        continue
                    href = _ddg_unwrap(a["href"])
                    sn = res.select_one(".result__snippet")
                    url = self._safe_url(href)
                    if not url:
                        continue
                    out.append(SearchResult(title=a.get_text(strip=True)[:300], url=url,
                                            snippet=sn.get_text(" ", strip=True)[:500] if sn else "",
                                            provider=self.name))
                    if len(out) >= max_results:
                        break
                return out
        except Exception as e:
            log.warning("Web search failed: %s", e)
            return []


def _ddg_unwrap(href: str) -> str:
    try:
        q = urllib.parse.urlparse(href).query
        p = urllib.parse.parse_qs(q)
        if "uddg" in p:
            return urllib.parse.unquote(p["uddg"][0])
    except Exception:
        pass
    return href


class TrustedDomainProvider(SearchProvider):
    """CONTROLLED MODE: запрос ограничен site: доверенных доменов (§7.1)."""

    name = "trusted"

    def __init__(self, inner: WebSearchProvider, domains: set[str]) -> None:
        self.inner = inner
        self.domains = domains

    async def search(self, query: str, *, max_results: int = 8,
                     mode: AnalysisMode = AnalysisMode.RESEARCH) -> list[SearchResult]:
        dom_list = sorted(self.domains)
        chunks = [dom_list[i:i + 6] for i in range(0, len(dom_list), 6)] or [dom_list]
        results: list[SearchResult] = []
        seen: set[str] = set()
        for chunk in chunks[:3]:
            site_ops = " OR ".join(f"site:{d}" for d in chunk)
            rs = await self.inner.search(f"{query} ({site_ops})", max_results=max_results)
            for r in rs:
                host = httpx.URL(r.url).host.lower().removeprefix("www.")
                if any(host == d or host.endswith("." + d) for d in self.domains):
                    if r.url not in seen:
                        seen.add(r.url)
                        r.provider = self.name
                        results.append(r)
        return results[:max_results]


def build_providers(cfg: AppConfig, mode: AnalysisMode) -> list[SearchProvider]:
    """Набор провайдеров под режим. Никаких фиктивных результатов: если API
    недоступен — провайдер вернёт пустой список, и это попадёт в limitations."""
    providers: list[SearchProvider] = []
    if mode == AnalysisMode.CONTROLLED:
        providers.append(TrustedDomainProvider(WebSearchProvider(), cfg.sources.all_trusted_domains()))
        if cfg.research.pubmed_enabled:
            providers.append(PubmedProvider(cfg))
        if cfg.research.europepmc_enabled:
            providers.append(EuropePMCProvider())
        return providers
    if cfg.research.pubmed_enabled:
        providers.append(PubmedProvider(cfg))
    if cfg.research.europepmc_enabled:
        providers.append(EuropePMCProvider())
    if cfg.research.crossref_enabled:
        providers.append(CrossrefProvider(cfg))
    if cfg.research.web_search_enabled:
        providers.append(WebSearchProvider())
    return providers
