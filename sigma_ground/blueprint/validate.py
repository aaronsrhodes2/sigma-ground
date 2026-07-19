"""Deterministic cross-checks on a MechanismSpec — plain arithmetic, never an
LLM call (Golden Rule: exact analytic checks over a source's own claims, not
a plausibility guess). Two jobs:

  1. Catch transcription errors (a mistyped tooth count, a center distance
     that doesn't match the stated module).
  2. Prove internal consistency the SOURCE ITSELF usually offers for free —
     a horology textbook's worked example typically states a result (beats
     per hour, turns per hour) two different ways; if our transcribed teeth/
     leaf counts don't reproduce BOTH, the transcription is wrong somewhere.

Every gap where a check can't run (module absent, no independently-stated
period to check against) is reported, not silently skipped — mirrors the
project's ``# PHYSICS_GAP:`` convention, here as ``# GEOMETRY_GAP:``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .schema import MechanismSpec


@dataclass
class ValidationReport:
    ok: bool = True
    errors: list = field(default_factory=list)     # hard failures — a physically impossible spec
    warnings: list = field(default_factory=list)    # unusual but not wrong
    gaps: list = field(default_factory=list)         # "# GEOMETRY_GAP: ..." — data absent, check skipped

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_gap(self, msg: str) -> None:
        self.gaps.append(f"# GEOMETRY_GAP: {msg}")


_STANDARD_PRESSURE_ANGLES_DEG = (14.5, 20.0, 25.0)


def validate(spec: MechanismSpec) -> ValidationReport:
    r = ValidationReport()
    by_name = {g.name: g for g in spec.gears}

    for g in spec.gears:
        teeth = g.teeth.value
        if not isinstance(teeth, int) or teeth <= 0:
            r.add_error(f"{g.name}: teeth/leaves must be a positive integer, got {teeth!r}")
        if not g.teeth.estimated and not g.teeth.quote.strip():
            r.add_error(f"{g.name}: cites source {g.teeth.source!r} but carries no quote — "
                       "a citation without a verbatim quote is unverifiable")
        if g.module_mm is None:
            r.add_gap(f"{g.name}: no module cited — physical size/geometry cannot be "
                     "derived yet (tooth/leaf counts and ratios are still valid)")
        if g.pressure_angle_deg is not None:
            pa = g.pressure_angle_deg.value
            if not any(abs(pa - std) < 0.05 for std in _STANDARD_PRESSURE_ANGLES_DEG):
                r.add_warning(f"{g.name}: pressure angle {pa}° is outside the standard set "
                             f"{_STANDARD_PRESSURE_ANGLES_DEG}° — unusual, not necessarily wrong "
                             "(period gearing legitimately varies)")

    for m in spec.meshes:
        ga, gb = by_name.get(m.a), by_name.get(m.b)
        if ga is None:
            r.add_error(f"mesh references unknown gear '{m.a}'")
            continue
        if gb is None:
            r.add_error(f"mesh references unknown gear '{m.b}'")
            continue
        if ga.module_mm is None or gb.module_mm is None:
            r.add_gap(f"mesh {m.a}->{m.b}: module absent on at least one gear — "
                     "center-distance consistency check skipped")
            continue
        ma, mb = ga.module_mm.value, gb.module_mm.value
        if abs(ma - mb) > 1e-6:
            r.add_error(f"mesh {m.a}->{m.b}: modules disagree ({ma} vs {mb}) — "
                       "meshing gears must share a module")
            continue
        if m.center_distance_mm is None:
            r.add_gap(f"mesh {m.a}->{m.b}: no center distance cited — "
                     "cannot cross-check against module*(Na+Nb)/2")
            continue
        expected = ma * (ga.teeth.value + gb.teeth.value) / 2.0
        actual = m.center_distance_mm.value
        if abs(actual - expected) > 0.02 * expected:            # 2% tolerance
            r.add_error(f"mesh {m.a}->{m.b}: stated center distance {actual} mm doesn't "
                       f"match module*(Na+Nb)/2 = {expected:.3f} mm (>2% off) — "
                       "likely a transcription error")

    if spec.escapement is not None:
        e = spec.escapement
        if e.lift_angle_deg is not None and e.escape_wheel_teeth is not None:
            pitch = 360.0 / e.escape_wheel_teeth.value
            if e.lift_angle_deg.value >= pitch:
                r.add_error(f"escapement lift angle {e.lift_angle_deg.value}° >= the escape "
                           f"wheel's own tooth pitch {pitch:.2f}° — mechanically impossible "
                           "(a tooth can't lift past the next tooth's position)")
        elif e.escape_wheel_teeth is not None:
            r.add_gap("escapement: no lift angle cited — cannot check against tooth pitch")

    return r


def cumulative_ratio(spec: MechanismSpec, mesh_pairs) -> float:
    """Product of a.teeth/b.teeth over an ORDERED sequence of (wheel_name,
    pinion_name) mesh pairs — the turns of the LAST pinion per one turn of
    the FIRST wheel. Deliberately NOT a flat alternating name path: a
    pinion and the wheel fixed to its own arbor turn together 1:1, and that
    fact isn't derivable from the gear list alone, so callers chain
    explicit (a, b) mesh pairs (e.g. ``[(m.a, m.b) for m in spec.meshes]``)
    rather than one long name sequence. This is the reusable arithmetic a
    mechanism's catalog-building script uses to cross-check a source's own
    independently-stated period/turns-per-hour result."""
    by_name = {g.name: g for g in spec.gears}
    ratio = 1.0
    for a, b in mesh_pairs:
        ratio *= by_name[a].teeth.value / by_name[b].teeth.value
    return ratio


__all__ = ["ValidationReport", "validate", "cumulative_ratio"]
