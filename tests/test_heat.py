"""dynamics.fields.heat — FVM heat diffusion on a voxel field (first taste).

The LAW is pure (an explicit ∂T/∂t=α∇²T stencil); we GROUND it by building the
diffusivity α = k/(ρ·cp) from the real material thermal physics. An insulated
field must conserve energy while equalizing (Fourier); a hot body in ambient void
must cool toward ambient. numpy/scipy are the opt-in [shapes] deps.
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from sigma_ground.dynamics.fields.heat import diffuse, stable_dt, thermal_energy
from sigma_ground.field.interface.thermal import (
    thermal_conductivity, heat_capacity_volumetric,
)

# diffusivity α = k / (ρ·cp), straight from the physics — grounded, not invented.
_K = thermal_conductivity("iron")
_RHOCP = heat_capacity_volumetric("iron")
_ALPHA = _K / _RHOCP                                  # m²/s


def test_alpha_is_physical():
    assert _K > 0 and _RHOCP > 0
    assert 1e-6 < _ALPHA < 1e-3                       # iron ≈ 2×10⁻⁵ m²/s


def test_stable_dt():
    assert stable_dt(0.0, 0.005) == float("inf")
    dt = stable_dt(_ALPHA, 0.005)
    assert 0 < dt < 1.0


def test_insulated_conserves_energy_and_equalizes():
    dx = 0.005
    T = np.full((16, 16, 16), 300.0)
    T[:, :, 8:] = 400.0                               # half hot
    alpha = np.full(T.shape, _ALPHA)
    rhocp = np.full(T.shape, _RHOCP)
    E0 = thermal_energy(T, rhocp, dx)
    std0 = T.std()
    Tf = diffuse(T, alpha, dx, total_time=40.0)       # insulated (Neumann edges)
    E1 = thermal_energy(Tf, rhocp, dx)
    assert abs(E1 - E0) / E0 < 0.01                   # energy conserved
    assert abs(Tf.mean() - 350.0) < 0.5               # mean unchanged (uniform ρcp)
    assert Tf.std() < 0.5 * std0                      # field equalizing
    assert Tf.max() < 400.0 and Tf.min() > 300.0      # heat moved, monotone


def test_hot_body_cools_toward_ambient():
    dx = 0.005
    N = 22
    T = np.full((N, N, N), 300.0)
    solid = np.zeros((N, N, N), dtype=bool)
    solid[7:15, 7:15, 7:15] = True                    # an 8³ hot iron cube
    T[solid] = 600.0
    void = ~solid
    alpha = np.where(solid, _ALPHA, 0.0)
    cube0 = float(T[solid].mean())
    Tf = diffuse(T, alpha, dx, total_time=120.0, ambient_mask=void, ambient_T=300.0)
    assert Tf[solid].max() < 600.0                    # surface cooled
    assert Tf[solid].mean() < cube0                   # cube losing heat to ambient
    assert Tf[void].max() == pytest.approx(300.0)     # ambient held (Dirichlet)
