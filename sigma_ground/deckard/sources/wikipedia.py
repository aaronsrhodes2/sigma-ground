"""Free factual prose — a cached Wikipedia/MediaWiki REST summary.

Gives the researcher a short, attributable description of an object (its typical
real proportions + what it is made of) to GROUND the shape it proposes, instead
of relying on the model's recall alone. Same stdlib + on-disk-cache discipline as
the rest of ``sources``: any failure (offline, no article, non-JSON) returns None
and the caller degrades to the bare name — research still runs.
"""
from __future__ import annotations

import urllib.parse

from . import web

# The REST summary endpoint returns a small JSON with a plain-text ``extract``.
_REST = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_NOT_FOUND = "https://mediawiki.org/wiki/HyperSwitch/errors/not_found"


def summary(name: str, max_chars: int = 1200) -> str | None:
    """A plain-text Wikipedia summary for ``name`` (≤ ``max_chars``), or None."""
    title = urllib.parse.quote((name or "").strip().replace(" ", "_"))
    if not title:
        return None
    d = web.get_json(_REST.format(title=title))
    if not isinstance(d, dict) or d.get("type") == _NOT_FOUND:
        return None
    text = (d.get("extract") or "").strip()
    return text[:max_chars] or None


__all__ = ["summary"]
