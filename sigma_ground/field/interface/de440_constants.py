"""The one place DE440's own constants live for both the predictor and Mentat's body lookup.

Data: fixtures/de440_header_constants.json holds the header's constants EXACTLY as JPL wrote them (`raw`, au^3/day^2 and so on) plus the Jupiter and Saturn field values from the Juno and Cassini papers.
Everything derived from them is CALCULATED here when asked, never stored: GM in km^3/s^2 (header value x AU^3 / 86400^2), the Earth and Moon GM (split from the Earth-Moon barycentre GM with the header's
mass ratio EMRAT), the field blocks. If an equation or a convention changes, one function changes and every consumer follows (the Captain, 2026-10-26: keep calculations as calculations).

Used by: rolling_shootout.restated_predictors (the `restated_hdr` label), mcp/tools/astronomy.py (solar_system_body and body_constants), field/interface/body_data.py, scripts/constants_audit.py.
"""
from __future__ import annotations

import functools
import json
import re
from pathlib import Path

_FILE = Path(__file__).parent / "fixtures" / "de440_header_constants.json"
_SOURCE = "JPL DE440 header (header.440; Park et al. 2021)"
_SYSTEM_GM = {"Mars system": "GM4", "Jupiter system": "GM5", "Saturn system": "GM6", "Uranus system": "GM7", "Neptune system": "GM8", "Pluto system": "GM9"}


@functools.lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(_FILE.read_text(encoding="utf-8"))


def _gm_conversion() -> float:
    """au^3/day^2 -> km^3/s^2, with the header's own AU (km)."""
    au = load()["raw"]["AU"]
    return au ** 3 / 86400.0 ** 2


def gm_km3_s2(name: str) -> float | None:
    """GM in km^3/s^2, calculated from the header values. Names: Sun, Mercury, Venus, Earth, Moon, 'Earth-Moon barycentre', and 'Mars system' ... 'Pluto system'."""
    c = load()["raw"]
    conv = _gm_conversion()
    gmb = c["GMB"] * conv
    emrat = c["EMRAT"]
    table = {"Sun": lambda: c["GMS"] * conv, "Mercury": lambda: c["GM1"] * conv, "Venus": lambda: c["GM2"] * conv, "Earth": lambda: gmb * emrat / (1.0 + emrat), "Moon": lambda: gmb / (1.0 + emrat),
             "Earth-Moon barycentre": lambda: gmb}
    table.update({k: (lambda key=v: c[key] * conv) for k, v in _SYSTEM_GM.items()})
    f = table.get(name)
    return f() if f else None


def body(name: str) -> dict | None:
    """Field constants of a body as a flat dict (gm_km3_s2, reference_radius_km, j2, ...) with a 'source' string, or None if neither the DE440 header nor the cited papers carry it. Calculated on each call."""
    doc = load()
    c = doc["raw"]
    key = name.strip().lower()
    if key == "sun":
        d = {"gm_km3_s2": gm_km3_s2("Sun"), "j2": c["J2SUN"], "j3": c["J3SUN"], "j4": c["J4SUN"], "reference_radius_km": c["ASUN"], "gm_rate_per_yr": c["GMSDOT"]}
    elif key == "earth":
        d = {"gm_km3_s2": gm_km3_s2("Earth"), "reference_radius_km": c["RE"], "j2": c["J2E"], "j3": c["J3E"], "j4": c["J4E"], "j5": c["J5E"], "j2_rate_per_yr": c["J2EDOT"], "j2_t2": c["J2ET2"],
             "k2": [c["K2E0"], c["K2E1"], c["K2E2"]], "tidal_time_delay_days": [c["TAUE0"], c["TAUE1"], c["TAUE2"]]}
    elif key == "moon":
        gravity = {k: c[k] for k in sorted(c) if re.fullmatch(r"(J[2-6]|[CS][2-6][0-6])M", k)}
        d = {"gm_km3_s2": gm_km3_s2("Moon"), "reference_radius_km": c["AM"], "k2": c["K2M"], "tidal_time_delay_days": c["TAUM"], "libration_lbet": c["LBET"], "libration_lgam": c["LGAM"],
             "core_oblateness": c["COBLAT"], "gravity": gravity, "j2": gravity["J2M"], "j3": gravity["J3M"], "j4": gravity["J4M"]}
    else:
        for n, v in doc["other_sources"].items():
            if n.lower() == key:
                out = {k: x for k, x in v.items() if k not in ("citation", "read_2026-10-23")}
                out["source"] = v["citation"]
                return out
        return None
    d["source"] = _SOURCE
    return d


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
