"""Spatial drag-heating temperature field — windward heating + Fourier conduction.

A reentry body is HOT on its leading (windward) face and COLD on its trailing
side; a single bulk temperature can't express that, a per-cell field can. These
lock the contract: a real gradient, the windward face held at the cited heating
temperature, the trailing face near ambient, and the diffusivity grounded in the
material physics (α = k/ρcp).
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from sigma_ground.materia.thermal_field import windward_heating_field, ThermalField


def _field(windward_T=2000.0, spread=8.0, mat="iron", R=0.05, n=26):
    return windward_heating_field(R, mat, windward_T=windward_T,
                                  spread_time_s=spread, n=n)


def test_windward_hotter_than_trailing():
    f = _field()
    assert f.windward_T > f.trailing_T + 500.0          # a real, large gradient


def test_windward_holds_the_heating_temperature():
    f = _field(windward_T=2000.0)
    # the stagnation point is held at (≈) the cited heating temperature
    assert abs(f.windward_T - 2000.0) < 60.0


def test_trailing_stays_near_ambient():
    f = _field(windward_T=2000.0, spread=8.0)
    assert f.trailing_T < 400.0                         # trailing face stays cold


def test_directionality_follows_velocity():
    # leading face is along −velocity; flip the velocity → the hot point flips.
    down = windward_heating_field(0.05, "iron", windward_T=1800.0, spread_time_s=6.0,
                                  velocity_dir=(0.0, -1.0, 0.0), n=24)
    up = windward_heating_field(0.05, "iron", windward_T=1800.0, spread_time_s=6.0,
                                velocity_dir=(0.0, 1.0, 0.0), n=24)
    # sample just inside the bottom surface for each
    r = 0.045
    bottom_down = down.sample(0.0, -r, 0.0)
    bottom_up = up.sample(0.0, -r, 0.0)
    assert bottom_down > bottom_up + 500.0              # down-velocity heats the bottom


def test_no_heating_when_windward_is_ambient():
    f = _field(windward_T=288.15)                       # no overheat → no gradient
    assert abs(f.windward_T - f.trailing_T) < 30.0


def test_sample_outside_grid_is_ambient():
    f = _field()
    assert f.sample(10.0, 10.0, 10.0) == pytest.approx(f.T_ambient)


def test_field_carries_geometry():
    f = _field(R=0.07)
    assert isinstance(f, ThermalField)
    assert f.radius_m == pytest.approx(0.07)
    assert f.n == 26 and f.dx > 0


def test_alpha_grounded_for_two_metals():
    # copper conducts ~6× iron, so its rind spreads faster — the field is built
    # from real α, not a constant. Hotter interior penetration for copper.
    r = 0.03
    cu = windward_heating_field(0.05, "copper", windward_T=2000.0, spread_time_s=10.0, n=24)
    fe = windward_heating_field(0.05, "iron", windward_T=2000.0, spread_time_s=10.0, n=24)
    # interior point a bit behind the stagnation face
    assert cu.sample(0.0, -r, 0.0) >= fe.sample(0.0, -r, 0.0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
