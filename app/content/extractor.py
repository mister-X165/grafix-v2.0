"""Извлечение содержимого из HTML/PDF (ТЗ §63).

Приоритет: trafilatura → BeautifulSoup → грубый текст. Извлекаются title,
author, date, основной текст, metadata, canonical URL, ссылки на источники.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.content.fetcher import FetchResult, extract_pdf_text
from app.core.models import ContentNature, Document
from app.security.urlguard import canonicalize_url

log = logging.getLogger(__name__)


def _trafilatura_extract(html: str, url: str) -> Optional[dict]:
    try:
        import trafilatura
    except ImportError:  # controlled dependency loading (§51)
        log.debug("trafilatura not installed")
        return None
    try:
        meta = trafilatura.extract_metadata(html)
        body = trafilatura.extract(
            html, url=url, include_comments=False, include_tables=True,
            favor_precision=True,
        )
    except Exception:
        log.warning("trafilatura failed for %s", url, exc_info=True)
        return None
    if not body or len(body.strip()) < 80:
        return None
    d: dict = {"text": body}
    if meta is not None:
        d["title"] = getattr(meta, "title", "") or ""
        d["author"] = getattr(meta, "author", "") or ""
        d["date"] = getattr(meta, "date", "") or ""
        d["sitename"] = getattr(meta, "sitename", "") or ""
    return d


def _bs_extract(html: str) -> dict:
    from bs4 import BeautifulSoup  # обязательная зависимость

    # Выбор parser'а с graceful degradation (§51): lxml быстрее, html.parser — stdlib fallback.
    parser = "lxml"
    try:
        import lxml  # noqa: F401
    except ImportError:
        log.warning("lxml not installed — falling back to html.parser")
        parser = "html.parser"
    try:
        soup = BeautifulSoup(html, parser)
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "form", "nav", "footer", "header", "aside"]):
        tag.decompose()
    title = (soup.title.get_text(strip=True) if soup.title else "")
    main = soup.find("article") or soup.find(main=True) or soup.body or soup
    text = re.sub(r"\n{3,}", "\n\n", main.get_text("\n").strip())

    author = _meta(soup, ["author", "citation_author", "article:author", "og:article:author"])
    date = _meta(soup, ["article:published_time", "publication_date", "date",
                        "parsely-pub-date", "dc.date", "og:article:published_time"])
    canonical = _meta(soup, ["og:url"]) or ""
    if not canonical:
        link = soup.find("link", rel="canonical")
        if link and link.get("href"):
            canonical = link["href"]
    refs = [a.get("href", "") for a in soup.find_all("a", href=True)][:200]
    return {"text": text, "title": title, "author": author, "date": date,
            "canonical": canonical, "refs": refs}


def _meta(soup, names: list[str]) -> str:
    for n in names:
        t = soup.find("meta", attrs={"name": n}) or soup.find("meta", attrs={"property": n})
        if t and t.get("content"):
            return str(t["content"]).strip()
    return ""


_AD_RE = re.compile(
    r"(купить|заказать со скидкой|реклама|спонсорский материал|партнёрский материал|"
    r"advertorial|sponsored content|affiliate link|order now)", re.IGNORECASE)


def detect_content_nature(text: str, url: str = "") -> ContentNature:
    """Эвристика реклама/информация (§41). Точную классификацию дополняет LLM."""
    blob = (url + "\n" + text[:4000]).lower()
    if "/buy" in blob or "?aff" in blob or "utm_source=affiliate" in blob:
        return ContentNature.AFFILIATE_CONTENT
    hits = len(_AD_RE.findall(blob))
    if hits >= 2:
        return ContentNature.ADVERTISEMENT
    if hits == 1:
        return ContentNature.SPONSORED_CONTENT
    return ContentNature.INFORMATION


def extract_document(fetch: FetchResult, cfg) -> Document:
    """Построить Document из результата загрузки. Никогда не бросает — error в metadata."""
    doc = Document(url=fetch.url, canonical_url=canonicalize_url(fetch.url))
    if fetch.is_pdf:
        text, err = extract_pdf_text(fetch.raw, cfg.limits.max_pdf_pages)
        if err:
            doc.metadata["extract_error"] = err
            return doc
        doc.text = text
        doc.title = (text.splitlines() or [""])[0][:200]
    else:
        # Декодируем raw, если text не заполнен (например, FetchResult построен вручную/из кэша)
        html = fetch.text
        if not html and fetch.raw:
            enc = "utf-8"
            ct = (fetch.content_type or "").lower()
            m = re.search(r"charset=([^\s;]+)", ct)
            if m:
                enc = m.group(1).strip("'\"")
            try:
                html = fetch.raw.decode(enc, errors="replace")
            except LookupError:
                html = fetch.raw.decode("utf-8", errors="replace")
        data = _trafilatura_extract(html, fetch.url) if html else {}
        if not data:
            data = _bs_extract(html) if html else {}
        doc.text = data.get("text", "")
        doc.title = data.get("title", "") or ""
        doc.author = data.get("author", "") or ""
        doc.published = data.get("date", "") or ""
        doc.canonical_url = data.get("canonical") or doc.canonical_url
        doc.references = [r for r in data.get("refs", []) if r.startswith("http")]
    doc.content_nature = detect_content_nature(doc.text, fetch.url)
    return doc
