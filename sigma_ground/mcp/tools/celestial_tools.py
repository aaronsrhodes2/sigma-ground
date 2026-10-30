"""Two-body celestial mechanics from DE440 states: orbital elements, Hill radius, sphere of influence, Roche limit, Lagrange points, synodic period, Hohmann transfer, GR perihelion advance, the integration-step
check, and the measured skill of the N-body model by body. The computation is field/interface/celestial.py and field/interface/skill.py; this file only asks and words the answer.

A question the data cannot answer comes back as a declined result with the reason (value None), never a guess."""
from __future__ import annotations

from sigma_ground.mcp.provenance import ToolResult

_ORBIT_SOURCE = "sigma-ground celestial: JPL DE440 kernel (planets, Moon) and DE440 r8 snapshots (moons); GM from the DE440 header and satellite ephemerides"
_OSCULATING = ("Osculating (instantaneous two-body) elements in the ICRF equatorial frame with mu = GM central + GM body; not mean elements, so a planet's value moves by a fraction of a percent with the other planets' pull. "
               "Mars to Pluto are planet-system barycentres, the Earth is the Earth-Moon barycentre, a moon's elements are those of the nearest annual snapshot (the epoch is in the answer).")


def _run(fn, inputs: dict, *, units: str, source: str = _ORBIT_SOURCE, notes: str = _OSCULATING) -> ToolResult:
    from sigma_ground.field.interface import celestial as C, where_when as ww
    try:
        jd = ww.parse_when(inputs.get("when") or "now")[0] if "when" in inputs else None
        out = fn(C, jd)
    except (C.CelestialError, ww.WhereWhenError) as e:
        return ToolResult(value=None, source=source, notes=str(e), inputs=inputs)
    out = {k: (float(v) if hasattr(v, "item") else v) for k, v in out.items() if k not in ("r_km", "v_km_s")}
    extra = out.get("note") or ""
    return ToolResult(value=out, units=units, source=source, provenance_tag="DERIVED", notes=(notes + (" " + extra if extra else "")).strip(), inputs=inputs)


def orbital_elements(body_name: str, central_body: str = "sun", when: str = "now") -> ToolResult:
    """Osculating orbital elements of a planet or moon about a central body at a date: semi-major axis, eccentricity, inclination, period, periapsis and apoapsis, from JPL DE440 states.

    Planets and the Moon at any date from the DE440 kernel (the Earth as the Earth-Moon barycentre); moons from the nearest annual DE440 snapshot, the epoch stated. Small bodies are in small_body.
    """
    return _run(lambda C, jd: C.elements(body_name, central_body, jd), {"body_name": body_name, "central_body": central_body, "when": when}, units="km, degrees, days, km^3/s^2")


def hill_radius(body_name: str, central_body: str = "sun", when: str = "now") -> ToolResult:
    """Hill radius of a body: the distance within which its gravity, not the central body's tide, holds a satellite, a (1 - e) (m / 3M)^(1/3), with a and e from the DE440 state (also given at a)."""
    return _run(lambda C, jd: C.hill_radius(body_name, central_body, jd), {"body_name": body_name, "central_body": central_body, "when": when}, units="km")


def sphere_of_influence(body_name: str, central_body: str = "sun", when: str = "now") -> ToolResult:
    """Laplace sphere of influence a (m/M)^(2/5): inside it the body, not the central body, is the better centre for a two-body orbit (patched-conic mission planning); a from the DE440 state."""
    return _run(lambda C, jd: C.sphere_of_influence(body_name, central_body, jd), {"body_name": body_name, "central_body": central_body, "when": when}, units="km")


def roche_limit(body_name: str, central_body: str) -> ToolResult:
    """Roche limit of a satellite about its primary, rigid and fluid, from their radii and GMs: body_name is the satellite, central_body the primary, e.g. roche_limit('moon', 'earth')."""
    return _run(lambda C, jd: C.roche_limit(body_name, central_body), {"body_name": body_name, "central_body": central_body}, units="km",
                source="sigma-ground celestial: radius and GM of the bodies (body_constants); rigid form from tidal stretch = self-gravity, fluid form Roche 1849 (coefficient 2.44)",
                notes="Rigid: where the primary's tidal stretch across the satellite equals its surface gravity, R_s (2 M_p / M_s)^(1/3). Fluid: 2.44 R_s (M_p / M_s)^(1/3). "
                      "A satellite inside the limit is pulled apart only if its own strength is below the tide.")


def lagrange_points(body_name: str, central_body: str = "sun", when: str = "now") -> ToolResult:
    """The five Lagrange points of a body and its central body (L1, L2, L3 as km from the central body, L4 and L5 at 60 degrees), with whether L4 and L5 are stable (Routh's criterion), at the pair's separation on that date."""
    return _run(lambda C, jd: C.lagrange_points(body_name, central_body, jd), {"body_name": body_name, "central_body": central_body, "when": when}, units="km",
                notes="Circular restricted three-body problem at the pair's separation on that date (the real orbit is slightly eccentric). " + _OSCULATING)


def synodic_period(body_name: str, other_body: str, central_body: str = "sun", when: str = "now") -> ToolResult:
    """Mean time between successive alignments (conjunctions or oppositions) of two bodies about the same central body, 1/|1/P1 - 1/P2|, from their DE440 orbital periods."""
    return _run(lambda C, jd: C.synodic_period_days(body_name, other_body, central_body, jd), {"body_name": body_name, "other_body": other_body, "central_body": central_body, "when": when}, units="days")


def hohmann_transfer(body_name: str, other_body: str, central_body: str = "sun", when: str = "now") -> ToolResult:
    """Hohmann transfer between two bodies' orbits about a central body: the two burns, their total, and the transfer time, with the orbits taken as circular and coplanar at their distances on that date."""
    return _run(lambda C, jd: C.hohmann(body_name, other_body, central_body, jd), {"body_name": body_name, "other_body": other_body, "central_body": central_body, "when": when}, units="km/s, days, km",
                notes="Textbook idealisation: circular coplanar orbits at the distances the bodies have on that date; real transfers differ with eccentricity, inclination and launch window. " + _OSCULATING)


def perihelion_advance(body_name: str, central_body: str = "sun", when: str = "now") -> ToolResult:
    """General-relativistic advance of a body's perihelion, 6 pi mu / (c^2 a (1 - e^2)) per orbit and in arcseconds per century, from its DE440 orbit (the relativistic part only; the other planets add their own)."""
    return _run(lambda C, jd: C.gr_perihelion_advance(body_name, central_body, jd), {"body_name": body_name, "central_body": central_body, "when": when}, units="radians per orbit, arcseconds per century",
                notes="First post-Newtonian term only. The closed form equals the advance the sigma-ground 1PN engine integrates (test_celestial.py). " + _OSCULATING)


def step_check(central_body: str, step_s: float) -> ToolResult:
    """Does an integration step resolve every satellite of a body? Steps per orbit of each moon (or each planet, for the Sun) at that step, against the rule of 300 steps per shortest period.

    Measured on the real Jupiter system: Io at 71 steps per orbit is 118,000 km off after 5 yr, at 142 steps 7,610 km (error falls as step^4, order 3.95); the rule is shortest period / 300.
    """
    from sigma_ground.field.interface import celestial as C, where_when as ww
    inputs = {"central_body": central_body, "step_s": step_s}
    try:
        if not step_s > 0:
            raise C.CelestialError("the step must be positive")
        rows = C.step_rows(central_body, step_s, ww.parse_when("now")[0])
    except (C.CelestialError, ww.WhereWhenError) as e:
        return ToolResult(value=None, source=_ORBIT_SOURCE, notes=str(e), inputs=inputs)
    bad = [r["body"] for r in rows if not r["adequate"]]
    value = {"central_body": central_body, "step_s": step_s, "rows": rows, "unresolved": bad, "all_adequate": not bad,
             "shortest_period_days": rows[0]["period_days"] if rows else None,
             "step_that_resolves_all_s": rows[0]["period_days"] * 86400.0 / C.STEPS_PER_ORBIT_TARGET if rows else None}
    msg = (f"{len(bad)} of {len(rows)} bodies are not resolved by this step: {', '.join(bad)}." if bad else f"All {len(rows)} bodies get at least {C.STEPS_PER_ORBIT_TARGET} steps per orbit.")
    return ToolResult(value=value, units="days, steps", source="sigma-ground celestial: osculating periods from DE440 states; rule shortest period / 300 (CANDIDATE_LEDGER C14), convergence order measured in C19",
                      provenance_tag="DERIVED", notes=msg + " " + _OSCULATING, inputs=inputs)


def body_skill(body_name: str) -> ToolResult:
    """How well can this model predict a body, and can it at all? The class (predictable, loses orbital phase within a year, measurements disagree, not scored), the measured error against JPL DE440 at +1, +3, +5, +10 and +20 years, and the converged error for fast moons.

    From the full-system shoot-out (21 windows), the converged moon models and the forecast's own hindcast; the numbers and their sources are in the answer.
    """
    from sigma_ground.field.interface import skill
    inputs = {"body_name": body_name}
    try:
        d = skill.body_skill(body_name)
    except skill.SkillError as e:
        return ToolResult(value=None, source="sigma-ground body_skill_table (measured against JPL DE440)", notes=str(e), inputs=inputs)
    return ToolResult(value=d, units="km (position error against JPL DE440), days", source="sigma-ground body_skill_table: measured against JPL DE440 by scripts/make_body_skill_table.py; sources and hashes in its metadata",
                      provenance_tag="DERIVED", notes=d["summary"], inputs=inputs)
