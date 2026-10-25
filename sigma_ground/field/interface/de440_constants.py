"""The one place DE440's own constants live for both the predictor and Mentat's body lookup.

Data: fixtures/de440_header_constants.json, written by scripts/make_de440_constants.py from the DE440 ASCII header (the constants JPL's integration used; Park et al. 2021, AJ 161:105). Jupiter's and Saturn's
fields are not in that header; they come from the Juno and Cassini papers named in the file's `other_sources`, with what was and was not read from each.

Used by: rolling_shootout.restated_predictors (the `restated_hdr` label), mcp/tools/astronomy.py (solar_system_body returns these fields), scripts/constants_audit.py (ours against DE440, and why).
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

_FILE = Path(__file__).parent / "fixtures" / "de440_header_constants.json"


@functools.lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(_FILE.read_text(encoding="utf-8"))


def body(name: str) -> dict | None:
    """Field constants of a body as a flat dict (gm_km3_s2, reference_radius_km, j2, ...) with a 'source' string, or None if neither the DE440 header nor the cited papers carry it."""
    doc = load()
    key = name.strip().lower()
    if key in ("sun", "earth", "moon"):
        d = {k: v for k, v in doc[key].items() if k != "header_names"}
        if key == "moon":
            g = d.pop("gravity")
            d["gravity"] = g
            d["j2"], d["j3"], d["j4"] = g["J2M"], g["J3M"], g["J4M"]
        d["source"] = "JPL DE440 header (header.440; Park et al. 2021)"
        return d
    for n, d in doc["other_sources"].items():
        if n.lower() == key:
            out = {k: v for k, v in d.items() if k not in ("citation", "read_2026-10-23")}
            out["source"] = d["citation"]
            return out
    return None


def gm_km3_s2(name: str) -> float | None:
    return load()["gm_km3_s2"].get(name)


def predictor_zonal_overrides(bodies: tuple[str, ...] = ("Sun", "Earth", "Moon", "Jupiter", "Saturn")) -> tuple[tuple[str, str, float], ...]:
    """(body, attribute, value) triples for Predictor.zonal_overrides, from the header (Sun, Earth, Moon) and the cited papers (Jupiter, Saturn).

    Attributes are those BodyParams has: radius_km, j2, j3, j4. The header's higher terms (Earth J5, the Moon's C21 ... C66) have no slot there and are not applied.
    """
    out: list[tuple[str, str, float]] = []
    for name in bodies:
        d = body(name)
        if d is None:
            continue
        out.append((name, "radius_km", float(d["reference_radius_km"])))
        for a in ("j2", "j3", "j4"):
            if a in d:
                out.append((name, a, float(d[a])))
    return tuple(out)
