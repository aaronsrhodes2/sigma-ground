"""Deckard's self-audit — how much of a generated object is GROUNDED vs guessed,
and whether the result is physically sane.

Deckard can generate *any nameable object* (the model decomposes it into
primitives, the compiler turns that into validated matter) — but most arbitrary
objects are dimensionally ungrounded: the proportions are the model's estimate,
so the mass is a guess, not a measurement. Deckard's promise is "never present a
guess as measured", and the audit makes that explicit per object:

  * a **groundedness** score (how many of its dims/densities are cited vs estimated),
  * the resulting size/mass, and
  * **warnings** for implausible (too big/heavy) or inconsistent (failed self-check,
    non-mating joints) results.

A `verified` object is trustworthy; an `estimated` one is generated and
self-consistent but rests on guessed proportions; a `suspect` one should not be
trusted as-is. The audit *describes* confidence — it never silently "fixes" a
value.
"""
from __future__ import annotations

_HUMAN_MIN_M, _HUMAN_MAX_M = 5e-4, 12.0     # below/above this is not a human-scale object
_SIZE_VERIFY_M = 3.0                          # larger than most handheld/furniture -> verify
_MASS_VERIFY_KG = 50.0                        # heavier than most nameable objects -> verify


def audit(spec, construct) -> dict:
    """Confidence + plausibility report for a researched/compiled object."""
    # -- groundedness, from the spec's provenance Facts --
    dim_facts, dens_facts = [], []
    for p in getattr(spec, "parts", []):
        dim_facts += list(p.dims.values())
        dens_facts.append(p.density)
    for L in getattr(spec, "layers", []):
        dens_facts.append(L.density)
        dim_facts.append(L.thickness)
    dim_facts += list(getattr(spec, "geometry", {}).values())

    dims_total, dims_cited = len(dim_facts), sum(1 for f in dim_facts if not f.estimated)
    dens_total, dens_cited = len(dens_facts), sum(1 for f in dens_facts if not f.estimated)
    total = (dims_total + dens_total) or 1
    groundedness = (dims_cited + dens_cited) / total

    # -- plausibility / integrity, from the compiled construct --
    (x0, x1), (y0, y1), (z0, z1) = construct.bbox
    ext = max(x1 - x0, y1 - y0, z1 - z0)
    mass = construct.mass_kg
    v = construct.validation or {}
    warnings = []
    if ext > _HUMAN_MAX_M or ext < _HUMAN_MIN_M:
        warnings.append(f"non-human scale: {ext:.3g} m")
    elif ext > _SIZE_VERIFY_M:
        warnings.append(f"large for a handheld object: {ext:.2f} m — verify proportions")
    if mass > _MASS_VERIFY_KG:
        warnings.append(f"heavy: {mass:.1f} kg — verify proportions")
    if not v.get("passed", True):
        warnings.append(f"self-check failed (mass residual {v.get('mass_residual', 0) * 100:.1f}%)")
    if v.get("unmated"):
        warnings.append(f"{len(v['unmated'])} declared mate(s) form no real interface")

    if dims_cited > 0 and dens_cited == dens_total and not warnings:
        verdict = "verified"
    elif not warnings:
        verdict = "estimated"     # generated + self-consistent, but proportions are guesses
    else:
        verdict = "suspect"       # implausible or inconsistent — do not trust as-is

    return {
        "verdict": verdict,
        "groundedness": round(groundedness, 3),
        "dimensions_grounded": f"{dims_cited}/{dims_total}",
        "density_grounded": f"{dens_cited}/{dens_total}",
        "size_m": round(ext, 4),
        "mass_kg": round(mass, 6),
        "warnings": warnings,
    }


def render(a: dict) -> str:
    """Human-readable one-block audit."""
    mark = {"verified": "OK", "estimated": "~~", "suspect": "!!"}.get(a["verdict"], "??")
    lines = [
        f"[{mark}] {a['verdict'].upper()}  groundedness {a['groundedness']:.0%} "
        f"(dims {a['dimensions_grounded']}, density {a['density_grounded']})",
        f"      size {a['size_m']} m, mass {a['mass_kg'] * 1000:.1f} g",
    ]
    for w in a["warnings"]:
        lines.append(f"      ! {w}")
    return "\n".join(lines)


__all__ = ["audit", "render"]
