"""Безопасная загрузка веб-материалов (ТЗ §30–§32, §61–§63).

- httpx async, streaming с ограничением размера;
- редиректы контролируются вручную и каждый проходит SSRF-валидацию;
- JavaScript НЕ исполняется; PDF читается PyMuPDF (опционально);
- Playwright — только при явном включении в конфиге и недоступности HTTP.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import mimetypes
from dataclasses import dataclass, field
from typing import Optional

import httpx

from app.config import AppConfig, get_config
from app.security.urlguard import UrlValidationError, validate_url

log = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FactCheckLocal/1.0 (research tool)"


@dataclass
class FetchResult:
    url: str                      # финальный URL после редиректов
    status_code: int
    content_type: str = ""
    raw: bytes = b""
    text: str = ""                # декодированный HTML/текст (не для PDF)
    is_pdf: bool = False
    from_cache: bool = False
    error: str = ""               # пустая строка = успех
    headers: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error and 200 <= self.status_code < 400


class RequestBudget:
    """Ограничитель числа запросов на один анализ (§32)."""

    def __init__(self, max_requests: int) -> None:
        self.max_requests = max_requests
        self.used = 0

    def take(self) -> None:
        if self.used >= self.max_requests:
            raise BudgetExceeded("Превышен лимит сетевых запросов анализа")
        self.used += 1

    @property
    def remaining(self) -> int:
        return max(0, self.max_requests - self.used)


class BudgetExceeded(RuntimeError):
    pass


class HttpFetcher:
    """Async fetcher c SSRF-guard и resource limits."""

    def __init__(self, cfg: Optional[AppConfig] = None, budget: Optional[RequestBudget] = None) -> None:
        self.cfg = cfg or get_config()
        self.budget = budget or RequestBudget(self.cfg.limits.max_requests_per_analysis)
        self._sem = asyncio.Semaphore(self.cfg.limits.fetch_concurrency)

    # ------------------------------------------------------------- cache ---
    def _cache_get(self, url: str) -> Optional[FetchResult]:
        try:
            from app.storage.db import Database

            row = Database.instance(self.cfg).cache_lookup(url)
        except Exception:  # pragma: no cover - кэш не критичен
            log.debug("cache lookup failed", exc_info=True)
            return None
        if not row:
            return None
        res = FetchResult(url=row["url"], status_code=200, content_type=row["content_type"],
                          raw=row["body"], from_cache=True)
        if not res.is_pdf:
            res.text = _decode(res.raw, row["content_type"])
        return res

    def _cache_put(self, res: FetchResult) -> None:
        try:
            from app.storage.db import Database

            Database.instance(self.cfg).cache_store(
                url=res.url, content_hash=content_hash(res.raw), body=res.raw,
                content_type=res.content_type,
            )
        except Exception:  # pragma: no cover
            log.debug("cache store failed", exc_info=True)

    # -------------------------------------------------------------- fetch --
    async def fetch(self, url: str, *, use_cache: bool = True) -> FetchResult:
        async with self._sem:
            return await self._fetch_inner(url, use_cache=use_cache)

    async def _fetch_inner(self, url: str, *, use_cache: bool) -> FetchResult:
        try:
            current = validate_url(url)
        except UrlValidationError as e:
            return FetchResult(url=url, status_code=0, error=f"URL отклонён: {e}")

        if use_cache:
            cached = self._cache_get(current)
            if cached is not None:
                log.info("cache hit: %s", current)
                return cached

        limits = self.cfg.limits
        client = httpx.AsyncClient(
            follow_redirects=False,          # редиректы под нашим контролем
            timeout=httpx.Timeout(limits.request_timeout_s),
            headers={"User-Agent": USER_AGENT, "Accept-Language": "ru,en;q=0.8"},
        )
        try:
            redirects = 0
            while True:
                self.budget.take()
                try:
                    async with client.stream("GET", current) as resp:
                        if resp.status_code in (301, 302, 303, 307, 308):
                            redirects += 1
                            if redirects > limits.max_redirects:
                                return FetchResult(url=current, status_code=resp.status_code,
                                                   error="Слишком много редиректов")
                            loc = resp.headers.get("location", "")
                            if not loc:
                                return FetchResult(url=current, status_code=resp.status_code,
                                                   error="Редирект без Location")
                            next_url = httpx.URL(current).join(loc)
                            try:
                                current = validate_url(str(next_url))
                            except UrlValidationError as e:
                                return FetchResult(url=str(next_url), status_code=resp.status_code,
                                                   error=f"Редирект запрещён: {e}")
                            continue

                        if resp.status_code >= 400:
                            return FetchResult(url=current, status_code=resp.status_code,
                                               error=f"HTTP {resp.status_code}")

                        ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
                        declared = resp.headers.get("content-length")
                        if declared and int(declared) > limits.max_download_size_bytes:
                            return FetchResult(url=current, status_code=resp.status_code,
                                               error="Файл превышает лимит размера загрузки")
                        chunks: list[bytes] = []
                        total = 0
                        async for chunk in resp.aiter_bytes():
                            total += len(chunk)
                            if total > limits.max_download_size_bytes:
                                return FetchResult(url=current, status_code=resp.status_code,
                                                   error="Размер загрузки превысил лимит")
                            chunks.append(chunk)
                        body = b"".join(chunks)
                        ct = ctype or mimetypes.guess_type(current)[0] or "text/html"
                        res = FetchResult(url=current, status_code=resp.status_code,
                                          content_type=ct, raw=body,
                                          headers=dict(resp.headers))
                        res.is_pdf = ct == "application/pdf" or body[:5] == b"%PDF-"
                        if not res.is_pdf:
                            res.text = _decode(body, ct)
                        if ct.startswith("text/") or res.is_pdf or "json" in ct:
                            self._cache_put(res)
                        return res
                except httpx.TimeoutException:
                    return FetchResult(url=current, status_code=0, error="Таймаут запроса")
                except httpx.HTTPError as e:
                    return FetchResult(url=current, status_code=0, error=f"Ошибка сети: {type(e).__name__}")
                except BudgetExceeded as e:
                    return FetchResult(url=current, status_code=0, error=str(e))
        finally:
            await client.aclose()

    # ------------------------------------------------------------ offline --
    def check_connectivity(self) -> bool:
        """Быстрая проверка интернета (для OFFLINE MODE, §49)."""
        try:
            r = httpx.get("https://www.google.com/generate_204", timeout=5.0,
                          headers={"User-Agent": USER_AGENT})
            return r.status_code < 500
        except Exception:
            try:
                r = httpx.get("https://pubmed.ncbi.nlm.nih.gov/", timeout=5.0,
                              headers={"User-Agent": USER_AGENT})
                return r.status_code < 500
            except Exception:
                return False


def _decode(raw: bytes, content_type: str) -> str:
    for enc in ("utf-8", "cp1251", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_pdf_text(raw: bytes, max_pages: int) -> tuple[str, str]:
    """Извлечь текст из PDF. Возвращает (text, error). PyMuPDF — опциональная зависимость."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return "", "Для чтения PDF установите pymupdf (pip install pymupdf)"
    try:
        doc = fitz.open(stream=raw, filetype="pdf")
    except Exception as e:
        return "", f"Не удалось открыть PDF: {type(e).__name__}"
    try:
        parts: list[str] = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            parts.append(page.get_text("text"))
        return "\n".join(parts), ""
    finally:
        doc.close()
