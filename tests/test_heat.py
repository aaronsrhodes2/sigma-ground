"""dynamics.fields.heat — FVM heat diffusion on a voxel field (first taste).

The LAW is pure (an explicit ∂T/∂t=α∇²T stencil); we GROUND it by building the
diffusivity α = k/(ρ·cp) from the real material thermal physics. An insulated
field must conserve energy while equalizing (Fourier); a hot body in ambient void
must cool toward ambient. numpy/scipy are the opt-in [shapes] deps.
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from sigma_ground.dynamics.fields.heat import (diffuse, diffuse_fvm, stable_dt,
                                               thermal_energy)
from sigma_ground.field.interface.thermal import (
    thermal_conductivity, heat_capacity_volumetric, thermal_diffusivity,
)

# diffusivity α = k / (ρ·cp), straight from the physics — grounded, not invented.
_K = thermal_conductivity("iron")
_RHOCP = heat_capacity_volumetric("iron")
_ALPHA = _K / _RHOCP                                  # m²/s
_K_CU = thermal_conductivity("copper")
_RHOCP_CU = heat_capacity_volumetric("copper")


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


# ── diffuse_fvm: the conservative harmonic-mean face-flux solver ──────────

def test_thermal_diffusivity_helper_matches_hand_built_alpha():
    assert thermal_diffusivity("iron") == pytest.approx(_ALPHA, rel=1e-12)


def test_fvm_uniform_material_matches_legacy_diffuse():
    """With uniform k the harmonic mean equals k, so the face form must
    reproduce the legacy α·∇²T stencil (same dt, same zero-flux edges)."""
    dx = 0.005
    T = np.full((16, 16, 16), 300.0)
    T[:, :, 8:] = 400.0
    legacy = diffuse(T, np.full(T.shape, _ALPHA), dx, total_time=40.0)
    fvm, info = diffuse_fvm(T, np.full(T.shape, _K), np.full(T.shape, _RHOCP),
                            dx, total_time=40.0)
    assert info["substeps"] > 0
    assert np.allclose(fvm, legacy, rtol=1e-6, atol=1e-6)


def test_fvm_two_material_bar_conserves_energy_exactly():
    """iron|copper insulated bar: face antisymmetry ⇒ conservation to roundoff
    ACROSS the material interface — the thing the α·∇²T form cannot do."""
    dx = 0.005
    shape = (40, 6, 6)
    iron = np.zeros(shape, dtype=bool)
    iron[:20] = True                                  # left half iron, right copper
    k = np.where(iron, _K, _K_CU)
    rho_cp = np.where(iron, _RHOCP, _RHOCP_CU)
    T = np.where(iron, 500.0, 300.0)                  # hot iron end
    E0 = thermal_energy(T, rho_cp, dx)
    E_cu0 = float((rho_cp * T)[~iron].sum()) * dx ** 3
    Tf, _ = diffuse_fvm(T, k, rho_cp, dx, total_time=20.0)
    E1 = thermal_energy(Tf, rho_cp, dx)
    E_cu1 = float((rho_cp * Tf)[~iron].sum()) * dx ** 3
    assert abs(E1 - E0) / E0 < 1e-9                   # exact (roundoff) conservation
    assert E_cu1 > E_cu0                              # heat crossed the interface
    assert Tf.max() <= 500.0 + 1e-9 and Tf.min() >= 300.0 - 1e-9


def test_fvm_steady_interface_matches_series_resistance():
    """Ends pinned hot/cold across an iron|copper bar → the steady profile must
    carry the analytic series-resistance flux, continuous across the interface,
    with the interface temperature the resistor divider predicts. The kink in
    the profile (k_fe ≠ k_cu) is the harmonic-mean signature."""
    dx = 0.005
    n = 16
    shape = (n, 4, 4)
    iron = np.zeros(shape, dtype=bool)
    iron[:n // 2] = True                              # iron x∈[0,8), copper x∈[8,16)
    k = np.where(iron, _K, _K_CU)
    rho_cp = np.where(iron, _RHOCP, _RHOCP_CU)
    T_h, T_c = 500.0, 300.0
    T = np.full(shape, T_c); T[iron] = T_h
    pin = np.zeros(shape, dtype=bool)
    pin[0], pin[-1] = True, True                      # Dirichlet end planes
    T_pin = np.zeros(shape); T_pin[0], T_pin[-1] = T_h, T_c
    Tf, _ = diffuse_fvm(T, k, rho_cp, dx, total_time=400.0,
                        ambient_mask=pin, ambient_T=T_pin)
    prof = Tf.mean(axis=(1, 2))                       # quasi-1D (uniform in y,z)
    # analytic: q = ΔT / (L_fe/k_fe + L_cu/k_cu), lengths pinned-center→interface
    L_half = (n / 2 - 0.5) * dx                       # 7.5 cells each side
    q_ana = (T_h - T_c) / (L_half / _K + L_half / _K_CU)
    q_fe = _K * (prof[3] - prof[4]) / dx              # discrete flux, iron interior
    q_cu = _K_CU * (prof[11] - prof[12]) / dx         # discrete flux, copper interior
    assert q_fe == pytest.approx(q_ana, rel=0.02)
    assert q_cu == pytest.approx(q_ana, rel=0.02)
    assert q_fe == pytest.approx(q_cu, rel=0.02)      # flux CONTINUITY at the interface
    # interface temperature: extrapolate half a cell from each side — both sides
    # must meet at the resistor-divider value T_h − q·L_fe/k_fe
    T_i_ana = T_h - q_ana * L_half / _K
    T_i_fe = prof[7] - 0.5 * dx * q_fe / _K
    T_i_cu = prof[8] + 0.5 * dx * q_cu / _K_CU
    assert T_i_fe == pytest.approx(T_i_ana, rel=0.02)
    assert T_i_cu == pytest.approx(T_i_ana, rel=0.02)
