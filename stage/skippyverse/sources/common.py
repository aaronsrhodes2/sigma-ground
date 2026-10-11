"""Shared helpers for the source adapters. Nothing here touches the network."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from skippyverse import cards as C  # noqa: E402
from skippyverse.taxonomy import guess_category  # noqa: E402

UNIT_M = {"mm": 0.001, "millimeter": 0.001, "millimeters": 0.001, "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
          "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
          "in": 0.0254, "inch": 0.0254, "inches": 0.0254, "ft": 0.3048, "foot": 0.3048, "feet": 0.3048}      # definitional conversions


class AdapterFormatError(ValueError):
    """The raw metadata does not look like the format this adapter expects."""


class NetworkNotAllowed(RuntimeError):
    """Live fetching was requested without --allow-network (or at all: no adapter fetches in this build)."""


def to_m(value: float, unit: str) -> float:
    try:
        return float(value) * UNIT_M[unit.strip().lower().rstrip(".")]
    except KeyError:
        raise AdapterFormatError(f"unknown length unit {unit!r}") from None


_NUM = r"(\d+(?:\.\d+)?)"
_DIMS = re.compile(_NUM + r"\s*(?:x|×|by)\s*" + _NUM + r"(?:\s*(?:x|×|by)\s*" + _NUM + r")?\s*(mm|cm|m|in|inches|inch|ft|feet)\b", re.I)


def parse_dims_text(text: str) -> tuple[float, float, float] | None:
    """'H x W x D: 10 x 20 x 30 cm' -> (0.10, 0.20, 0.30) metres in the order written; None if there are not three numbers + a unit."""
    m = _DIMS.search(text or "")
    if not m or m.group(3) is None:
        return None
    a, b, c, unit = m.groups()
    return to_m(a, unit), to_m(b, unit), to_m(c, unit)


def label_order_to_whd(dims: tuple[float, float, float], text: str) -> tuple[float, float, float]:
    """Map dims written in the order a label names ('H x W x D', 'H x L x W') to (w, d, h). `D` is the depth when present, else the
    second horizontal letter; without a recognisable three-letter label the numbers are taken as W x D x H as written."""
    m = re.search(r"\b([HWDL])\s*x\s*([HWDL])\s*x\s*([HWDL])\b", text or "", re.I)
    if not m:
        return dims
    letters = [g.upper() for g in m.groups()]
    if sorted(letters).count("H") != 1:
        return dims
    hi = letters.index("H")
    h = dims[hi]
    rest = [(letters[i], dims[i]) for i in range(3) if i != hi]
    di = next((i for i, (k, _) in enumerate(rest) if k == "D"), 1)
    d = rest[di][1]
    w = rest[1 - di][1]
    return w, d, h


def bbox_to_size(bbox, up: str, basis: str, source: str) -> dict:
    """[x, y, z] extents in METRES with `up` ('y' for glTF, 'z' for Z-up tools) -> {w, d, h, basis, source}. w = x, d = the other
    horizontal axis, h = the up axis."""
    x, y, z = (float(v) for v in bbox)
    if up == "y":
        w, d, h = x, z, y
    elif up == "z":
        w, d, h = x, y, z
    else:
        raise AdapterFormatError(f"up must be 'y' or 'z', got {up!r}")
    if min(w, d, h) <= 0:
        raise AdapterFormatError(f"non-positive bounding box {bbox!r}")
    return {"w": w, "d": d, "h": h, "basis": basis, "source": source}


def found_card(*, source: str, ext_id: str, name: str, tags=(), category: str | None = None, size: dict | None, license_raw: str,
               author: str | None, url: str | None, style: str = "low-poly", blocky: bool = False, extra_flags=(),
               aliases=(), fetched: str | None = None) -> dict:
    """Build a `found` card from one listing entry. The card's id embeds the source and the source's own id."""
    nm = name.strip()
    # "Sofa 01" is "sofa": a trailing variant number is not part of the name people say (the full title stays as an alias)
    short = re.sub(r"[\s_-]*\d+$", "", nm).strip() or nm
    aliases = [*aliases, *([nm.lower()] if short.lower() != nm.lower() else [])]
    nm = short
    card = C.make_card(
        id=f"found.{C.slug(source)}.{C.slug(ext_id)}", name=nm.lower(), aliases=aliases,
        category=category or guess_category(nm + " " + " ".join(tags)), tags=list(tags), size=size,
        mass=C.prior_mass(size) if size else None,
        visual={"class": "found", "style": style, "blocky": blocky, "lods": []},
        provenance={"source": source, "url": url, "author": author, "license": license_raw, "fetched": fetched},
        flags=["front_unknown", *extra_flags])
    # an unmeasured mesh has no known front: unknown fronts are treated as symmetric (design section 4)
    C.assert_valid(card)
    return card


def filter_servable(cards: list[dict], include_disallowed: bool) -> tuple[list[dict], list[dict]]:
    """(kept, skipped). By default a card whose licence gate says no is skipped, never ingested."""
    kept, skipped = [], []
    for c in cards:
        (kept if (c["redistributable"] or include_disallowed) else skipped).append(c)
    return kept, skipped
