"""Materia regression tests — the engine must equal the closed form.

These pin the proof-of-concept: a falling sphere has an exact analytic terminal
velocity, so the integrator can be checked against it to integration tolerance.
"""
import os
import sys

# Pin the canonical sigma-ground ahead of any stale worktree shadow.
_CANON = r"D:\Aaron\development\sigma-ground"
if os.path.isdir(_CANON) and _CANON not in sys.path:
    sys.path.insert(0, _CANON)

from sigma_ground.materia import (simulate_fall, analytic_terminal_velocity,
                                  terminal_velocity_drop, drag_heating_drop,
                                  high_altitude_descent, supersonic_projectile,
                                  drag_coefficient_mach)


def test_uniform_drop_matches_closed_form():
    """In uniform sea-level air the asymptote IS the closed-form v_t (<2%)."""
    s = simulate_fall("copper", 0.05, 8000.0, uniform=True)
    v_t = analytic_terminal_velocity(s["mass_kg"], s["area_m2"],
                                     s["sea_level_density_kg_m3"], cd=0.44)
    err = abs(s["impact_speed_m_s"] - v_t) / v_t
    assert err < 0.02, f"engine {s['impact_speed_m_s']:.2f} vs v_t {v_t:.2f} ({err*100:.1f}%)"


def test_altitude_drop_relaxes_to_terminal():
    """Realistic drop: impact near sea-level v_t, with an aloft overshoot."""
    s = simulate_fall("copper", 0.05, 10_000.0)
    v_t = analytic_terminal_velocity(s["mass_kg"], s["area_m2"],
                                     s["sea_level_density_kg_m3"], cd=0.44)
    err = abs(s["impact_speed_m_s"] - v_t) / v_t
    assert err < 0.15, f"impact {s['impact_speed_m_s']:.1f} vs v_t {v_t:.1f}"
    # The sphere must overshoot terminal velocity in the thin upper air…
    assert s["max_speed_m_s"] > s["impact_speed_m_s"]
    # …and that peak must occur aloft, not at the ground.
    assert s["max_speed_altitude_m"] > 100.0


def test_denser_material_hits_harder():
    """Terminal velocity is monotone in density: lead > copper > iron."""
    v = {mk: simulate_fall(mk, 0.05, 10_000.0)["impact_speed_m_s"]
         for mk in ("lead", "copper", "iron")}
    assert v["lead"] > v["copper"] > v["iron"], v


def test_scenario_self_validates():
    """The worked scenario reports its own PASS verdict."""
    res = terminal_velocity_drop("copper", 0.05, 10_000.0)
    assert res.validation["passed"], res.validation["note"]
    assert res.steps and res.steps[-1].units == "m/s"


# ── movement → thermal: drag heating ────────────────────────────────────
def test_drag_energy_conserved():
    """Drag dissipation from the energy budget and the force integral agree."""
    s = simulate_fall("iron", 0.05, 10_000.0)
    assert s["energy_residual"] < 0.03, s["energy_residual"]


def test_heating_monotone():
    """A denser sphere dissipates more energy over the same fall."""
    qb = {mk: simulate_fall(mk, 0.05, 10_000.0)["energy_budget"]["q_drag_budget_J"]
          for mk in ("lead", "iron")}
    assert qb["lead"] > qb["iron"], qb


def test_drag_heating_scenario_self_validates():
    """The drag-heating scenario reports its own PASS verdict and a ΔT > 0."""
    res = drag_heating_drop("iron", 0.05, 10_000.0)
    assert res.validation["passed"], res.validation["note"]
    assert res.steps[-1].units == "K" and res.steps[-1].value > 0


# ── Family D: advanced drag (Mach + high-altitude descent) ──────────────
def test_mach_drag_curve_shape():
    """C_d(Mach): subsonic plateau, transonic peak, supersonic decay."""
    assert drag_coefficient_mach(0.5) == drag_coefficient_mach(0.7)      # subsonic flat
    assert abs(drag_coefficient_mach(1.2) - 0.75) < 1e-9                 # transonic peak
    assert drag_coefficient_mach(2.5) < drag_coefficient_mach(1.2)       # supersonic decay
    assert drag_coefficient_mach(1.0) > drag_coefficient_mach(0.5)       # rise through transonic


def test_high_altitude_descent_supersonic_then_slows():
    """A stratosphere free-fall goes supersonic, then decelerates — self-validates."""
    res = high_altitude_descent()                      # 118 kg, 0.28 m², 35 km
    assert res.validation["passed"], res.validation["note"]
    assert res.validation["went_supersonic"] is True


def test_supersonic_projectile_self_validates():
    """The supersonic projectile conserves energy through the C_d(Mach) sweep."""
    res = supersonic_projectile()
    assert res.validation["passed"], res.validation["note"]
    assert res.validation["energy_residual"] < 0.05
