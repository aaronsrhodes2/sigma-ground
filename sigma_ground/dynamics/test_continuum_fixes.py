"""Regression sentinels for the 2026-10-10 continuum fixes (fix queue track D).

Every check compares the engine with an INDEPENDENT ground truth, not with the
code's own formula:

  D1 / A14  Boussinesq buoyancy of a lone parcel = α_V ΔT g with the literature
            VOLUMETRIC expansion coefficient (Wikipedia 'Thermal expansion',
            table at 20 °C: iron 35.4, copper 51, aluminium 69 × 10⁻⁶ K⁻¹).
            Before the fix the engine used the linear α_L: ratio 0.325–0.337.
            Under eos="ideal" a lone parcel falls freely (a = g exactly): the
            ideal-gas pressure already carries thermal buoyancy. Before: 0.99950 g.
  D1 / A15  SPH conduction against the heat equation ∂T/∂t = κ/(ρc_p) ∇²T for a
            sin(kx) profile on a cubic lattice (before: 1.2 × 10⁻⁴ of the
            Fourier rate), its 1/ρ scaling, exact pairwise heat conservation for
            neighbours of unequal density (before: 25 % imbalance), and the
            series-resistance law at an iron|copper contact (resistances of the
            two half-gaps add; before: the arithmetic mean, 1.61× the series flow).
"""
import math
import unittest

from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.continuum import (
    ContinuumParcel, ContinuumScene, continuum_step,
    _heat_conduction, _sph_density_sum, material_properties,
)
from sigma_ground.dynamics.fluid.kernel import smoothing_length

G_STD = 9.80665          # standard gravity (exact by definition, CGPM 1901)
T0 = 300.0
# Volumetric thermal expansion coefficients at 20 °C, Wikipedia 'Thermal expansion'
# (table "Thermal expansion coefficients at 20 °C", α_V column), verified 2026-10-10.
ALPHA_V_LIT = {"iron": 35.4e-6, "copper": 51e-6, "aluminum": 69e-6}


def _lone_parcel_accel_y(material, T, **scene_kw):
    p = ContinuumParcel(material, radius=0.01, temperature=T, position=Vec3(0, 0, 0))
    scene = ContinuumScene([p], gravity=Vec3(0, -G_STD, 0), ground=False,
                           reference_temperature=T0, **scene_kw)
    continuum_step(scene, 1e-3)
    return p.velocity.y / 1e-3


class TestBuoyancyVolumetric(unittest.TestCase):
    def test_buoyancy_equals_literature_alpha_V(self):
        dT = 10.0
        for mat, alpha_v in ALPHA_V_LIT.items():
            with self.subTest(material=mat):
                a = _lone_parcel_accel_y(mat, T0 + dT) - _lone_parcel_accel_y(mat, T0)
                ratio = a / (alpha_v * dT * G_STD)
                # the cascade's α_L(300 K) differs from the 20 °C table by ≤ 3 %
                self.assertTrue(0.95 <= ratio <= 1.05, ratio)

    def test_ideal_gas_lone_parcel_falls_freely(self):
        a = _lone_parcel_accel_y("water_ice", T0 + 10.0, eos="ideal",
                                 ideal_molar_mass_kg_mol=0.028014, ideal_gamma=1.4)
        self.assertAlmostEqual(a / -G_STD, 1.0, delta=1e-12)


def _conduction_lattice_Q(material, nlam, mass_scale=1.0):
    """engine dT/dt ÷ heat-equation dT/dt for T = T0 + sin(kx) on a cubic lattice
    (spacing dx, h = engine default 1.2 dx), least-squares over the interior row."""
    dx, ny, nz = 0.01, 7, 7
    h = smoothing_length(dx ** 3)
    r = dx * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)          # (4/3)π r³ = dx³
    kw = 2 * math.pi / (nlam * dx)
    nx = nlam + 8
    ps, idx = [], {}
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                p = ContinuumParcel(material, radius=r, temperature=T0 + math.sin(kw * ix * dx),
                                    position=Vec3(ix * dx, iy * dx, iz * dx))
                p.mass *= mass_scale
                idx[(ix, iy, iz)] = len(ps)
                ps.append(p)
    for p in ps:
        p.update_properties()
    _sph_density_sum(ps, h)
    rate = _heat_conduction(ps, h)
    props = material_properties(material, T0)
    rho = props["density_kg_m3"] * mass_scale                  # mass per lattice volume
    D = props["thermal_conductivity_W_mK"] / (rho * props["specific_heat_J_kgK"])
    inner = [idx[(ix, 3, 3)] for ix in range(3, nx - 3)]
    fourier = [-D * kw * kw * math.sin(kw * ps[i].position.x) for i in inner]
    return sum(rate[i] * f for i, f in zip(inner, fourier)) / sum(f * f for f in fourier)


class TestConductionFourier(unittest.TestCase):
    def test_lattice_rate_matches_heat_equation(self):
        # k h = 0.47; the residual is SPH discretisation (0.91 at k h = 0.94, 0.957 here)
        q = _conduction_lattice_Q("iron", 16)
        self.assertTrue(0.92 <= q <= 1.03, q)

    def test_rate_scales_as_inverse_density(self):
        # D = κ/(ρ c_p): doubling every mass doubles ρ and halves the rate
        q1 = _conduction_lattice_Q("iron", 8, 1.0)
        q2 = _conduction_lattice_Q("iron", 8, 2.0)
        self.assertAlmostEqual(q2 / q1, 1.0, delta=1e-9)
        self.assertTrue(0.85 <= q1 <= 1.03, q1)


def _pair(mat_a, mat_b, Ta, Tb, mass_a=None, mass_b=None, h=0.012, sep=0.01):
    a = ContinuumParcel(mat_a, 0.006, temperature=Ta, position=Vec3(0, 0, 0))
    b = ContinuumParcel(mat_b, 0.006, temperature=Tb, position=Vec3(sep, 0, 0))
    if mass_a is not None:
        a.mass = mass_a
    if mass_b is not None:
        b.mass = mass_b
    for p in (a, b):
        p.update_properties()
    _sph_density_sum([a, b], h)
    return a, b, _heat_conduction([a, b], h)


class TestConductionConservesHeat(unittest.TestCase):
    def test_unequal_density_pair_exactly_antisymmetric(self):
        m = ContinuumParcel("iron", 0.006).mass
        for (Ta, Tb) in ((310.0, 300.0), (300.0, 310.0)):
            with self.subTest(hotter="a" if Ta > Tb else "b"):
                a, b, d = _pair("iron", "iron", Ta, Tb, mass_a=m, mass_b=2.0 * m)
                self.assertNotAlmostEqual(a.sph_density / b.sph_density, 1.0, places=2)
                qa = a.mass * a.specific_heat * d[0]
                qb = b.mass * b.specific_heat * d[1]
                self.assertNotEqual(qa, 0.0)
                self.assertLess(abs(qa + qb), 1e-13 * abs(qa))
                self.assertLess(qa * (Ta - Tb), 0.0)            # heat leaves the hotter parcel

    def test_material_interface_obeys_series_resistance(self):
        # Equal masses and geometry, so only κ differs between the three pairs:
        # the contact conductance must be the series value 1/G = ½(1/G_FeFe + 1/G_CuCu).
        m = ContinuumParcel("iron", 0.006).mass

        def flow(ma, mb):
            a, _, d = _pair(ma, mb, T0 + 0.5, T0 - 0.5, mass_a=m, mass_b=m)
            return -a.mass * a.specific_heat * d[0]

        g_fe, g_cu, g_fecu = flow("iron", "iron"), flow("copper", "copper"), flow("iron", "copper")
        series = 2.0 * g_fe * g_cu / (g_fe + g_cu)
        self.assertAlmostEqual(g_fecu / series, 1.0, delta=2e-3)


if __name__ == "__main__":
    unittest.main()
