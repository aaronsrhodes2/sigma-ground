"""Cached HTTP JSON fetch (stdlib only) — for the free factual APIs.

Deckard stays zero-dependency: no httpx, just ``urllib`` plus a small persistent
on-disk JSON cache so repeated lookups don't re-hit the network. Any failure
(network down, non-JSON, timeout) returns None — callers degrade gracefully.

Tests monkeypatch :func:`get_json`, so they never touch disk or network.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
import urllib.request

_CACHE_DIR = pathlib.Path(tempfile.gettempdir()) / "deckard_web_cache"
_UA = "Deckard/0.1 (sigma-ground; physics shape researcher)"


def _cache_path(url: str) -> pathlib.Path:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return _CACHE_DIR / f"{h}.json"


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


__all__ = ["get_json"]
