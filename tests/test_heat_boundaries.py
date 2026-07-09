"""diffuse_fvm boundary conditions from the atmosphere presets.

The theater's ``physics_env`` and the heat solver's boundary spec come from the
SAME preset table (``field.interface.atmosphere.ATMOSPHERES``), translated by
``materia.thermal_field.boundary_from_env``:

  air (STP/IRT) → Dirichlet: void held at ambient (flagged well-stirred stand-in)
  vacuum (ISM)  → radiative-only: no conduction to void; Stefan-Boltzmann per
                  exposed face, with an EXACT radiated-energy ledger.

numpy/scipy are the opt-in [shapes] deps.
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from sigma_ground.dynamics.fields.heat import diffuse_fvm, thermal_energy
from sigma_ground.field.constants import STEFAN_BOLTZMANN
from sigma_ground.field.interface.atmosphere import atmosphere_preset
from sigma_ground.field.interface.thermal import (
    thermal_conductivity, heat_capacity_volumetric,
)
from sigma_ground.materia.thermal_field import (
    material_grids, exposed_faces, boundary_from_env,
)

_K = thermal_conductivity("iron")
_RHOCP = heat_capacity_volumetric("iron")


def _hot_iron_cube(N=12, cube=8, T_hot=900.0, T_void=2.725):
    """An iron cube at T_hot padded in void — labels, mask, T grid."""
    pad = (N - cube) // 2
    label = np.zeros((N, N, N), dtype=np.int8)
    label[pad:pad + cube, pad:pad + cube, pad:pad + cube] = 1
    solid = label == 1
    T = np.full(label.shape, float(T_void))
    T[solid] = T_hot
    return label, solid, T


def test_exposed_faces_counts_the_cube_surface():
    _, solid, _ = _hot_iron_cube()
    n = exposed_faces(solid)
    assert n.sum() == 6 * 8 * 8                       # 6 faces × 8² cells each
    assert n.max() == 3                               # corners expose 3 faces
    assert (n[~solid] == 0).all()                     # void radiates nothing


def test_material_grids_ground_the_properties():
    label, solid, _ = _hot_iron_cube()
    k, rho_cp, eps = material_grids(label, ["air", "iron"])
    # material_grids freezes properties at its T_ref (293.15) — a REAL k(T)
    # difference from the 300 K module default, not a tolerance fudge.
    assert k[solid].min() == k[solid].max() == pytest.approx(
        thermal_conductivity("iron", T=293.15), rel=1e-6)
    assert rho_cp[solid].max() == pytest.approx(
        heat_capacity_volumetric("iron", T=293.15), rel=1e-6)
    assert (k[~solid] == 0).all() and (rho_cp[~solid] == 0).all()
    assert 0.01 <= eps[solid].min() <= eps[solid].max() <= 1.0


def test_air_dirichlet_regression_via_presets():
    """The legacy hot-cube-cools behavior, now through diffuse_fvm + the IRT
    preset: surface cools, cube sheds heat, the bath never moves."""
    label, solid, T = _hot_iron_cube(T_hot=600.0, T_void=293.15)
    k, rho_cp, _ = material_grids(label, ["air", "iron"])
    bc = boundary_from_env("IRT", solid)
    assert bc["ambient_T"] == atmosphere_preset("IRT")["temperature_k"]
    cube0 = float(T[solid].mean())
    Tf, info = diffuse_fvm(T, k, rho_cp, 0.005, total_time=120.0, **bc)
    assert info["radiated_J"] == 0.0                  # air lane: no radiative sink
    assert Tf[solid].max() < 600.0
    assert Tf[solid].mean() < cube0
    assert Tf[~solid].max() == pytest.approx(293.15)  # bath held (Dirichlet)


def test_ism_radiative_ledger_is_exact_and_void_inert():
    """Vacuum: E0 − E1 == radiated_J to roundoff (the ledger is bookkeeping of
    the same arrays), and the void never changes — no conduction into nothing."""
    dx = 0.005
    label, solid, T = _hot_iron_cube()
    k, rho_cp, eps = material_grids(label, ["air", "iron"])
    bc = boundary_from_env("ISM", solid, emissivity_grid=eps)
    assert bc["radiative"]["T_env"] == pytest.approx(2.725)
    E0 = thermal_energy(T, rho_cp, dx)
    Tf, info = diffuse_fvm(T, k, rho_cp, dx, total_time=300.0, **bc)
    E1 = thermal_energy(Tf, rho_cp, dx)
    assert info["radiated_J"] > 0.0
    assert abs((E0 - E1) - info["radiated_J"]) / info["radiated_J"] < 1e-9
    assert np.array_equal(Tf[~solid], T[~solid])      # void inert in vacuum
    assert Tf[solid].max() < 900.0                    # it cooled…
    assert Tf[solid].min() > 2.725                    # …toward, never past, the CMB


def test_ism_matches_lumped_capacitance_ode():
    """Small-Biot iron cube in vacuum obeys m·c·dT/dt = −ε·σ·A·(T⁴−T_env⁴).
    Integrate that ODE with the SAME ε and stair-step area the solver uses —
    the mean solid temperature must track it within 5%."""
    dx = 0.005
    total_time = 300.0
    label, solid, T = _hot_iron_cube()
    k, rho_cp, eps = material_grids(label, ["air", "iron"])
    bc = boundary_from_env("ISM", solid, emissivity_grid=eps)
    Tf, _ = diffuse_fvm(T, k, rho_cp, dx, total_time=total_time, **bc)

    A = float(exposed_faces(solid).sum()) * dx * dx   # stair-step surface (m²)
    mc = float(rho_cp[solid].sum()) * dx ** 3         # total heat capacity (J/K)
    e = float(eps[solid].max())                       # uniform ε over the cube
    T_env = 2.725
    Tl, n = 900.0, 6000                               # explicit ODE, small steps
    dt = total_time / n
    for _ in range(n):
        Tl -= dt * e * STEFAN_BOLTZMANN * A * (Tl ** 4 - T_env ** 4) / mc
    assert float(Tf[solid].mean()) == pytest.approx(Tl, rel=0.05)


def test_irt_cools_faster_than_ism_at_900k():
    """The physics headline the demo narrates: at ~900 K a cm-scale iron cube
    loses heat to a room's air (conduction into the pinned bath) far faster
    than it can radiate into the interstellar dark."""
    dx = 0.005
    total_time = 120.0
    label, solid, T0 = _hot_iron_cube(T_void=293.15)
    k, rho_cp, eps = material_grids(label, ["air", "iron"])
    T_irt, _ = diffuse_fvm(T0, k, rho_cp, dx, total_time=total_time,
                           **boundary_from_env("IRT", solid))
    label, solid, T0v = _hot_iron_cube(T_void=2.725)
    T_ism, _ = diffuse_fvm(T0v, k, rho_cp, dx, total_time=total_time,
                           **boundary_from_env("ISM", solid, emissivity_grid=eps))
    assert float(T_irt[solid].mean()) < float(T_ism[solid].mean())


def test_boundary_from_env_accepts_theater_physics_env():
    """A theater's physics_env dict IS a valid boundary spec — the loop the
    preset table exists to close."""
    from sigma_ground.radiance import theaters
    scene = theaters.stage({"bbox": [[-0.1, 0.1], [-0.1, 0.1], [0.0, 0.2]]},
                           "deep_space")
    _, solid, _ = _hot_iron_cube()
    bc = boundary_from_env(scene["physics_env"], solid)
    assert "radiative" in bc and bc["radiative"]["T_env"] == pytest.approx(2.725)
