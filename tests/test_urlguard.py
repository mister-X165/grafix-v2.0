"""Unit-тесты URL-валидатора и SSRF-защиты (ТЗ §31, §72 security)."""

from __future__ import annotations

import pytest

from app.security.urlguard import UrlValidationError, validate_url


@pytest.mark.parametrize("url", [
    "http://localhost/x",
    "http://127.0.0.1/x",
    "http://0.0.0.0/x",
    "http://[::1]/x",
    "http://10.0.0.5/x",
    "http://192.168.1.1/x",
    "http://172.16.0.1/x",
    "http://169.254.169.254/latest/meta-data/",   # cloud metadata endpoint
    "http://metadata.google.internal/x",
])
def test_blocks_private_and_metadata(url):
    with pytest.raises(UrlValidationError):
        validate_url(url, allow_resolution=False)


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "gopher://evil/x",
    "javascript:alert(1)",
    "ftp://example.com/x",
    "",
])
def test_blocks_bad_schemes(url):
    with pytest.raises(UrlValidationError):
        validate_url(url, allow_resolution=False)


def test_blocks_credentials_and_internal_names():
    with pytest.raises(UrlValidationError):
        validate_url("https://user:pass@example.com/", allow_resolution=False)
    for host in ("localhost.localdomain", "internal.corp", "db.internal"):
        with pytest.raises(UrlValidationError):
            validate_url(f"http://{host}/", allow_resolution=False)


def test_blocks_too_long_and_garbage():
    with pytest.raises(UrlValidationError):
        validate_url("http://a" * 3000, allow_resolution=False)
    with pytest.raises(UrlValidationError):
        validate_url("not a url at all", allow_resolution=False)


def test_allows_public_https():
    out = validate_url("https://www.who.int/news-room/x", allow_resolution=False)
    assert out.startswith("https://www.who.int")
