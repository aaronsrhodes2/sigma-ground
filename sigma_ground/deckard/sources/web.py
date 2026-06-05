"""Cached HTTP JSON fetch (stdlib only) — for the free factual APIs.

Deckard stays zero-dependency: no httpx, just ``urllib`` plus a small persistent
on-disk JSON cache so repeated lookups don't re-hit the network. Any failure
(network down, non-JSON, timeout) returns None — callers degrade gracefully.

Tests monkeypatch :func:`get_json`, so they never touch disk or network.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
import time
import urllib.parse
import urllib.request

_CACHE_DIR = pathlib.Path(tempfile.gettempdir()) / "deckard_web_cache"
# Polite identification: a real client name + a contact URL. Wikimedia's API
# etiquette requires a meaningful User-Agent with a way to reach the operator.
_UA = ("Deckard/0.2 (sigma-ground physics shape researcher; "
       "https://github.com/aaronsrhodes2/sigma-ground)")
# Be a good citizen: at most ~1 request/second per host, issued sequentially
# (Wikimedia asks for serial requests; the cache means we rarely hit the net
# twice for the same URL anyway). Override via env for offline tests.
_MIN_INTERVAL_S = float(os.environ.get("DECKARD_FETCH_MIN_INTERVAL_S", "1.0"))
_last_hit: dict = {}


def _cache_path(url: str, suffix: str = "json") -> pathlib.Path:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return _CACHE_DIR / f"{h}.{suffix}"


def _rate_limit(url: str) -> None:
    """Sleep just enough to keep per-host requests below the polite rate."""
    host = urllib.parse.urlsplit(url).netloc
    prev = _last_hit.get(host)
    if prev is not None:
        wait = _MIN_INTERVAL_S - (time.monotonic() - prev)
        if wait > 0:
            time.sleep(wait)
    _last_hit[host] = time.monotonic()


def get_json(url: str, timeout: float = 10.0):
    """GET ``url`` and parse JSON, with a persistent on-disk cache.

    Returns the parsed object, or None on any error.
    """
    cp = _cache_path(url)
    try:
        if cp.is_file():
            return json.loads(cp.read_text(encoding="utf-8"))
    except Exception:
        pass
    try:
        _rate_limit(url)                       # polite: throttle per host
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        return None
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass
    return data


def get_text(url: str, timeout: float = 30.0):
    """GET ``url`` as text (e.g. NDJSON), with the same polite, cached discipline
    as :func:`get_json` — rate-limited, identifies itself, cached on disk so a
    bulk file (like a Quick Draw category) is fetched at most once. Returns the
    body string, or None on any error.
    """
    cp = _cache_path(url, "txt")
    try:
        if cp.is_file():
            return cp.read_text(encoding="utf-8")
    except Exception:
        pass
    try:
        _rate_limit(url)
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = r.read().decode("utf-8")
    except Exception:
        return None
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cp.write_text(text, encoding="utf-8")
    except Exception:
        pass
    return text


__all__ = ["get_json", "get_text"]
