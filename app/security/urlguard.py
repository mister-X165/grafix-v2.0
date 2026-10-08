"""SSRF protection and URL validation (ТЗ §31, §71, security_model.md).

Правила:
- разрешены только http/https;
- hostname резолвится через DNS, все IP проверяются на private/loopback/
  link-local (cloud metadata)/reserved/multicast;
- повторы для каждого redirect (вызывается policy перенаправлений);
- лимит redirect'ов и длины.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

ALLOWED_SCHEMES = {"http", "https"}

# hostname-ловушки, которые не должны уходить в DNS как «безопасные»
_BLOCKED_HOSTNAMES = {
    "localhost", "localhost.localdomain", "metadata", "metadata.google.internal",
    "ip6-localhost", "ip6-loopback",
}

# Внутренние/служебные доменные имена (ТЗ §31: internal hostnames)
_BLOCKED_HOST_SUFFIXES = (".localdomain", ".local", ".internal", ".corp", ".lan", ".home.arpa")

_URL_RE = re.compile(r"^https?://[^\s<>\"']+$", re.IGNORECASE)


class UrlValidationError(ValueError):
    """Безопасное для показа пользователю сообщение об ошибке URL."""


def _is_blocked_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True  # непарсимый IP — блокируем
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local      # 169.254.0.0/16 — cloud metadata endpoints
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def resolve_hosts(hostname: str) -> list[str]:
    """Все IP хоста. Бросает UrlValidationError при невозможности резолва."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        raise UrlValidationError(f"Не удалось определить адрес хоста: {hostname}") from e
    ips = sorted({i[4][0] for i in infos})
    if not ips:
        raise UrlValidationError(f"Хост не найден: {hostname}")
    return ips


def validate_url(url: str, *, allow_resolution: bool = True) -> str:
    """Проверить URL на безопасность. Возвращает канонизированную строку URL.

    :param allow_resolution: выключается в юнит-тестах / офлайн-режиме,
        чтобы не зависеть от DNS.
    """
    url = (url or "").strip()
    if not url:
        raise UrlValidationError("Пустой URL")
    if len(url) > 2048:
        raise UrlValidationError("URL слишком длинный")
    if not _URL_RE.match(url):
        raise UrlValidationError("URL должен начинаться с http:// или https://")

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise UrlValidationError(f"Неподдерживаемая схема: {scheme}")
    host = (parsed.hostname or "").lower().strip(".")
    if not host:
        raise UrlValidationError("В URL отсутствует хост")
    if "@" in (parsed.netloc or ""):
        raise UrlValidationError("URL с учётными данными запрещён")
    if host in _BLOCKED_HOSTNAMES:
        raise UrlValidationError("Локальные адреса запрещены")
    if host.endswith(_BLOCKED_HOST_SUFFIXES):
        raise UrlValidationError("Внутренние имена хостов запрещены")

    # IP-литералы (включая IPv6 в скобках и decimal/octal обфускацию через int)
    literal = host.strip("[]")
    try:
        ipaddress.ip_address(literal)
        is_literal = True
    except ValueError:
        is_literal = False
    if is_literal:
        if _is_blocked_ip(literal):
            raise UrlValidationError("Приватные/служебные IP-адреса запрещены")
        return url

    if not re.match(r"^[a-z0-9]([a-z0-9\-\.]*[a-z0-9])?$", host):
        raise UrlValidationError("Некорректное имя хоста")
    if "." not in host:
        raise UrlValidationError("Внутренние имена хостов запрещены")

    if allow_resolution:
        for ip in resolve_hosts(host):
            # IPv6-mapped IPv4
            v = ip
            if v.lower().startswith("::ffff:"):
                v = v[7:]
            if _is_blocked_ip(v):
                raise UrlValidationError(
                    f"Хост указывает на запрещённый адрес ({v}): {host}"
                )
    return url


@dataclass
class RedirectGuard:
    """Ограничитель редиректов: каждый новый location проходит validate_url."""

    max_redirects: int

    def check(self, next_url: str) -> str:
        return validate_url(next_url)


def is_same_origin(a: str, b: str) -> bool:
    pa, pb = urlparse(a), urlparse(b)
    return (pa.hostname or "") == (pb.hostname or "")


def canonicalize_url(url: str) -> str:
    """Убрать utm/поисковый шум, trailing slash, lower-case host — для дедупликации (§67)."""
    p = urlparse(url.strip())
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = re.sub(r"/+$", "", p.path or "")
    query = "&".join(
        kv for kv in (p.query or "").split("&")
        if kv and not kv.lower().startswith(("utm_", "gclid", "fbclid", "ref="))
    )
    frag = ""
    return f"{p.scheme.lower()}://{host}{path}" + (f"?{query}" if query else "") + frag
