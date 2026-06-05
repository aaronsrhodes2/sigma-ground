"""Materia ↔ Deckard — Deckard's catalogued shapes as simulation bodies.

Materia is **tier 3**; Deckard is **tier 2**, so Materia may consume it. This is
the bridge: Materia asks *"what shapes do I have?"* (Deckard's frozen catalog),
loads one as validated matter (a ``Construct`` — mass, centre of mass, inertia,
queryable SDF geometry), and derives the bulk properties a simulation needs —
the mass, and the projected (frontal) area aerodynamic drag sees. So when a
scenario calls for a feather, Materia pulls the feather.

Nothing here re-derives matter: mass/CoM/inertia come from Deckard's validated
Construct, and the frontal area is rasterised from its real geometry. Materia
adds only the drag wiring (``engine.simulate_drag_run`` /
``analytic_terminal_velocity``) and the self-check.
"""
from __future__ import annotations

from .. import deckard
from .engine import (analytic_terminal_velocity, simulate_drag_run,
                     MateriaResult, MateriaStep)

_AXES = {"x": 0, "y": 1, "z": 2}
_RHO_SEA = 1.225   # kg/m³, sea-level air density


def available_shapes() -> list[str]:
    """Names of the shapes Deckard has catalogued and Materia can simulate."""
    return deckard.catalog.available()


def has_shape(name: str) -> bool:
    """True if Materia can pull ``name`` from the catalog without researching."""
    return deckard.catalog.lookup(name) is not None


def load_shape(name: str, resolution: int = 64):
    """Compile a catalogued shape to a validated ``Construct`` (mass/CoM/inertia/SDF).

    Deterministic (``allow_llm=False``): a catalog hit, else a flagged best-guess —
    never a network call inside a simulation.
    """
    return deckard.identify(name, resolution=resolution, allow_llm=False)


def projected_area_m2(construct, axis: str = "y", n: int = 64) -> float:
    """Silhouette (frontal) area perpendicular to ``axis`` — the area drag sees.

    Rasterise the construct's shadow from its real geometry: a cell in the
    projection plane counts once if any point along the axis column is inside the
    matter (``material_at`` not None).
    """
    a = _AXES[axis]
    bb = construct.bbox
    lo = (bb[0][0], bb[1][0], bb[2][0])
    hi = (bb[0][1], bb[1][1], bb[2][1])
    u, v = [i for i in (0, 1, 2) if i != a]
    du, dv, da = (hi[u] - lo[u]) / n, (hi[v] - lo[v]) / n, (hi[a] - lo[a]) / n
    cell = du * dv
    area = 0.0
    p = [0.0, 0.0, 0.0]
    for iu in range(n):
        p[u] = lo[u] + (iu + 0.5) * du
        for iv in range(n):
            p[v] = lo[v] + (iv + 0.5) * dv
            for ia in range(n):
                p[a] = lo[a] + (ia + 0.5) * da
                if construct.material_at(p[0], p[1], p[2]) is not None:
                    area += cell
                    break
    return area


def broadside_area_m2(construct, n: int = 48) -> tuple[float, str]:
    """Largest silhouette over the three axes (broadside) and which axis — the
    orientation a tumbling, fluttering object presents most of the time."""
    best_area, best_axis = 0.0, "y"
    for ax in ("x", "y", "z"):
        a = projected_area_m2(construct, ax, n)
        if a > best_area:
            best_area, best_axis = a, ax
    return best_area, best_axis


def drop(name: str, *, from_altitude_m: float = 5.0, cd: float = 1.2,
         resolution: int = 64) -> MateriaResult:
    """Drop a catalogued shape through still air and report how it falls.

    Mass and the frontal (broadside) area come from Deckard's validated
    ``Construct``; Materia adds only the drag integration. ``cd`` defaults to
    ~1.2 (a bluff plate — a feather is draggy, not a sphere). The self-check is
    the drag run's own energy budget (drag dissipation vs PE lost − KE gained),
    and we report the closed-form terminal velocity √(2mg/ρC_dA) alongside.
    """
    c = load_shape(name, resolution=resolution)
    area, axis = broadside_area_m2(c)
    v_term = analytic_terminal_velocity(c.mass_kg, area, _RHO_SEA, cd)
    run = simulate_drag_run(c.mass_kg, area, start_altitude_m=from_altitude_m,
                            orientation="vertical", cd_mode="fixed", cd_value=cd)
    v_final = run["final_speed_m_s"]
    residual = run["energy_residual"]
    steps = [
        MateriaStep("shape", name, "", "", "deckard.catalog"),
        MateriaStep("mass", c.mass_kg, "kg", "Σ ρ·V (validated)", "deckard.Construct"),
        MateriaStep("broadside area", area, "m²", f"silhouette ⟂ {axis}", "deckard geometry"),
        MateriaStep("drag coefficient", cd, "", "bluff plate (assumed)", "assumption"),
        MateriaStep("terminal velocity", v_term, "m/s", "√(2 m g / ρ C_d A)",
                    "materia.engine.analytic_terminal_velocity"),
        MateriaStep("impact speed", v_final, "m/s",
                    f"drag-integrated fall from {from_altitude_m} m",
                    "materia.engine.simulate_drag_run"),
        MateriaStep("fall time", run["fall_time_s"], "s", "", "simulate_drag_run"),
    ]
    return MateriaResult(
        name=f"drop · {name}",
        inputs={"shape": name, "from_altitude_m": from_altitude_m, "cd": cd},
        steps=steps,
        summary=(f"A {name} ({c.mass_kg * 1000:.2f} g, {area * 1e4:.1f} cm² broadside) "
                 f"settles to ~{v_term:.2f} m/s terminal velocity in still air "
                 f"(impact {v_final:.2f} m/s after {run['fall_time_s']:.2f} s)."),
        validation={"passed": bool(residual < 0.05),
                    "note": (f"drag energy balanced to {residual * 100:.1f}%; "
                             f"impact {v_final:.2f} m/s vs closed-form terminal "
                             f"{v_term:.2f} m/s.")},
        outputs={"mass_kg": c.mass_kg, "frontal_area_m2": area,
                 "terminal_velocity_m_s": v_term, "impact_speed_m_s": v_final,
                 "fall_time_s": run["fall_time_s"]},
    )


__all__ = ["available_shapes", "has_shape", "load_shape",
           "projected_area_m2", "broadside_area_m2", "drop"]
