"""Tree allometry -- DBH -> total height / crown radius, cited to two
independently-verified, real peer-reviewed/USDA papers for ONE species
(loblolly pine, Pinus taeda). Parallel to sources.composition_of /
sources.sizes.typical_size_of: a small hand-curated local table, matched by
whole-word containment + an explicit alias map, returning Fact-wrapped
values with (source, license, confidence) provenance -- never a silent
guess.

Curated data ONLY: this module is NOT auto-scraped and does not grow by
pattern -- each new species needs its own independently-fetched-and-verified
source pair (the same discipline used for loblolly pine here), not an
extrapolation from this one species' shape.
"""
from __future__ import annotations

import math
import re

from ..schema import Fact

CM_PER_IN = 2.54
M_PER_FT = 0.3048

# ── Loblolly pine (Pinus taeda) -- height, Chapman-Richards ─────────────────
# HT(m) = 1.37 + b1 * (1 - exp(-b2 * DBH_cm)) ** b3
# Bragg, D. 2008. "A Comparison of Pine Height Models for the Crossett
# Experimental Forest." J. Arkansas Acad. Sci. 62:24-31. Table 2, loblolly
# pine row: n=415, R2=97.56%, RMSE=1.83m. Author affiliation: USDA Forest
# Service, Southern Research Station -- a US government work (public domain).
_LOBLOLLY_HEIGHT = {
    "b1": 41.9641, "b2": 0.0247, "b3": 1.1496,
    "dbh_min_cm": 0.3, "dbh_max_cm": 108.2,
    "source": "Bragg 2008, J. Arkansas Acad. Sci. 62:24-31 -- loblolly pine "
              "Chapman-Richards height model (n=415, R2=97.56%, RMSE=1.83m)",
    "license": "US government work (USDA Forest Service SRS) -- public domain",
    "confidence": 0.85,
}

# ── Loblolly pine -- crown diameter, linear (quadratic term not significant) ─
# LCW(ft) = b0 + b1 * DBH_in   [crown_radius_m = 0.5 * LCW_ft * M_PER_FT]
# Bechtold, W.A. 2003. "Crown-Diameter Prediction Models for 87 Species of
# Stand-Grown Trees in the Eastern United States." South. J. Appl. For.
# 27(4):269-278. Table 3, loblolly pine row: n=5099, R2=0.66, RMSE=3.1ft,
# CV=22%. USDA Forest Service, Southern Research Station -- public domain.
_LOBLOLLY_CROWN = {
    "b0": 2.5689, "b1": 1.3817,
    "dbh_min_in": 5.0, "dbh_max_in": 34.6,
    "source": "Bechtold 2003, South. J. Appl. For. 27(4):269-278 -- loblolly "
              "pine linear crown-diameter model (n=5099, R2=0.66)",
    "license": "US government work (USDA Forest Service SRS) -- public domain",
    "confidence": 0.75,
}

_SPECIES = {
    "loblolly pine": {
        "height": _LOBLOLLY_HEIGHT, "crown": _LOBLOLLY_CROWN,
        # 25cm sits inside BOTH fitted ranges (12.7-87.9cm crown /
        # 0.3-108.2cm height) -- interpolation, not extrapolation, for the
        # un-sized default case. This DEFAULT is a Choice (research.py
        # _tree_spec), not part of either cited equation.
        "default_dbh_cm": 25.0,
    },
}
_ALIASES = {
    "loblolly": "loblolly pine", "pinus taeda": "loblolly pine",
    "loblolly pine tree": "loblolly pine",
}


def _words(s: str) -> set:
    return {w for w in re.split(r"[^a-z0-9]+", s.lower()) if w}


def _resolve(species: str) -> str | None:
    key = species.strip().lower()
    if key in _SPECIES:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    kw = _words(key)
    best, best_len = None, 0
    for name in _SPECIES:
        nw = _words(name)
        if nw <= kw and len(nw) > best_len:
            best, best_len = name, len(nw)
    for alias, canon in _ALIASES.items():
        aw = _words(alias)
        if aw <= kw and len(aw) > best_len:
            best, best_len = canon, len(aw)
    return best


def species_default_dbh_cm(species: str) -> float | None:
    sp = _resolve(species)
    return _SPECIES[sp]["default_dbh_cm"] if sp else None


def height_of(species: str, dbh_cm: float) -> Fact | None:
    """Total tree height (m), Chapman-Richards, cited -- or None if the
    species isn't curated. Flags (lower confidence + a note) when dbh_cm
    falls outside the paper's fitted range (extrapolation)."""
    sp = _resolve(species)
    if sp is None:
        return None
    d = _SPECIES[sp]["height"]
    dbh_cm = max(0.0, float(dbh_cm))
    ht_m = 1.37 + d["b1"] * (1.0 - math.exp(-d["b2"] * dbh_cm)) ** d["b3"]
    extrapolated = not (d["dbh_min_cm"] <= dbh_cm <= d["dbh_max_cm"])
    conf = round(d["confidence"] * (0.6 if extrapolated else 1.0), 2)
    note = d["source"]
    if extrapolated:
        note += (f" [DBH={dbh_cm:.1f}cm outside the fitted "
                 f"{d['dbh_min_cm']}-{d['dbh_max_cm']}cm range -- "
                 "extrapolated; the paper's own Fig.4 notes Chapman-Richards "
                 "stays reasonable extrapolated toward 200cm, unlike the "
                 "rejected Curtis-Arney form]")
    return Fact(round(ht_m, 3), note, d["license"], conf)


def crown_radius_of(species: str, dbh_cm: float) -> Fact | None:
    """Crown RADIUS (m) = half the linear crown-diameter model, cited -- or
    None if the species isn't curated. Same extrapolation flagging as
    height_of."""
    sp = _resolve(species)
    if sp is None:
        return None
    d = _SPECIES[sp]["crown"]
    dbh_in = max(0.0, float(dbh_cm)) / CM_PER_IN
    lcw_ft = max(d["b0"] + d["b1"] * dbh_in, 0.1)   # never non-positive
    radius_m = 0.5 * lcw_ft * M_PER_FT
    extrapolated = not (d["dbh_min_in"] <= dbh_in <= d["dbh_max_in"])
    conf = round(d["confidence"] * (0.6 if extrapolated else 1.0), 2)
    note = d["source"]
    if extrapolated:
        note += (f" [DBH={dbh_cm:.1f}cm ({dbh_in:.1f}in) outside the fitted "
                 f"{d['dbh_min_in']}-{d['dbh_max_in']}in range -- "
                 "extrapolated]")
    return Fact(round(radius_m, 4), note, d["license"], conf)


def validate_tree_mass(species: str, dbh_cm: float, height_m: float,
                       trunk_mass_kg: float, wood_density_kg_m3: float
                       ) -> dict:
    """A geometric self-consistency + plausibility check for a compiled tree
    construct. NOT an independent biomass citation -- see the returned note.

      1. height self-consistency: recompute height_of(species, dbh_cm) and
         compare to the construct's own height_m (should match to float
         precision) -- catches a wiring bug (stale dbh, unit error), not
         biology.
      2. trunk-mass upper bound: an untapered cylinder at DBH radius run the
         full tree height is a geometric UPPER BOUND on true stem volume
         (real boles taper monotonically above breast height, by definition)
         -- trunk_mass_kg must not exceed
         wood_density_kg_m3 * pi * (dbh_cm/200)**2 * height_m. A violation
         means an arithmetic/unit bug, not a biology surprise (true by
         construction for any real, non-cylindrical tree).
    """
    checks = []
    h_fact = height_of(species, dbh_cm)
    h_ok = h_fact is not None and abs(h_fact.value - height_m) <= 0.01 * max(height_m, 1e-6)
    checks.append({"name": "height self-consistency", "passed": bool(h_ok),
                   "detail": f"recomputed={h_fact.value if h_fact else None}, "
                             f"construct={height_m}"})

    dbh_m = max(0.0, float(dbh_cm)) / 100.0
    upper_bound_kg = wood_density_kg_m3 * math.pi * (dbh_m / 2.0) ** 2 * height_m
    b_ok = 0.0 < trunk_mass_kg <= upper_bound_kg * 1.001   # float slack
    checks.append({"name": "trunk-mass upper bound (untapered cylinder)",
                   "passed": bool(b_ok),
                   "detail": f"trunk_mass_kg={trunk_mass_kg:.2f}, "
                             f"upper_bound_kg={upper_bound_kg:.2f}"})

    return {
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "needs_independent_biomass_citation": True,
        "note": ("Self-consistency only: no cited biomass equation backs "
                "the ABSOLUTE trunk/canopy mass numbers -- NEITHER "
                "verified source (Bechtold 2003 crown, Bragg 2008 height) "
                "publishes one for loblolly pine. The closest real "
                "candidate: Jenkins, Chojnacky, Heath & Birdsey 2003, "
                "'National-scale biomass estimators for United States tree "
                "species,' Forest Science 49(1):12-35 (a pseudo-taxonomic "
                "'pines' aboveground-biomass-vs-DBH equation) -- named here "
                "as the next real source to FETCH AND VERIFY the same way "
                "Bechtold/Bragg were (read the actual paper, check the "
                "actual table). Its coefficients are deliberately NOT "
                "hardcoded here because they were not independently "
                "verified this session -- do not trust them from memory."),
    }


__all__ = ["height_of", "crown_radius_of", "species_default_dbh_cm",
          "validate_tree_mass", "CM_PER_IN", "M_PER_FT"]
