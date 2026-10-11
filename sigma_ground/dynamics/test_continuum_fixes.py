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
  D6 / A16  cfl_timestep uses the EOS signal speed (isothermal c_s; ideal-gas
            sound speed, N₂ at 300 K = 353.2 m/s measured) and the force criterion
            0.25√(h/|a|) (Phantom eq. 73) with |a| the exact uniform g or the
            Newtonian self-gravity of the cloud; before: the cascade c_L for every
            EOS (5942× too small for a c_s = 1 m/s gas) and no force criterion.
            Conduction criterion: the heat equation's maximum principle (no new
            temperature extremum) holds at dt = cfl_timestep in a ~1 kg/m³
            iron-carrier gas; before: κΔt/(ρc_p h²) = 4.04, T 154–453 K after one
            step from 295–305 K, OverflowError at the third.
  D6 / A17  eos="ideal" with α_visc = 0 is time-reversible (Loschmidt echo returns
            the gas to its start to roundoff; before 0.22 R) and second order
            (self-convergence ratio ≈ 4; before 2.2).
"""
import math
import unittest

from sigma_ground.dynamics.vec import Vec3
import random

from sigma_ground.dynamics.continuum import (
    ContinuumParcel, ContinuumScene, continuum_step, cfl_timestep,
    _heat_conduction, _sph_density_sum, material_properties,
)
from sigma_ground.dynamics.seeding import seed_rotating_cloud, to_continuum, G_CODATA
from sigma_ground.field.constants import R_GAS
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


# ── D6: timestep and the ideal-gas temperature lag ───────────────────────
ZERO = Vec3(0.0, 0.0, 0.0)
M_N2, GAMMA_N2 = 0.028014, 1.4        # N₂ molar mass (kg/mol), diatomic γ


class TestCFLTimestep(unittest.TestCase):
    def _lone(self, gravity=ZERO, **kw):
        p = ContinuumParcel("iron", 0.01, temperature=300.0)
        return ContinuumScene([p], gravity=gravity, ground=False, **kw)

    def test_isothermal_signal_speed_is_c_s(self):
        sc = self._lone(eos="isothermal", sound_speed_m_s=1.0)   # P = ρ c_s²: signal speed c_s
        self.assertAlmostEqual(cfl_timestep(sc) / (0.3 * sc.h / 1.0), 1.0, delta=1e-12)

    def test_ideal_signal_speed_is_the_gas_sound_speed(self):
        sc = self._lone(eos="ideal", ideal_molar_mass_kg_mol=M_N2, ideal_gamma=GAMMA_N2)
        c_implied = 0.3 * sc.h / cfl_timestep(sc)
        # speed of sound in N₂ at 300 K, 1 bar: 353.2 m/s (knowledgedoor elements handbook)
        self.assertAlmostEqual(c_implied / 353.2, 1.0, delta=0.005)

    def test_force_criterion_under_strong_uniform_gravity(self):
        g = 1.0e4                                             # a lone parcel falls at exactly g
        sc = self._lone(gravity=Vec3(0.0, -g, 0.0), eos="isothermal", sound_speed_m_s=1.0)
        self.assertAlmostEqual(cfl_timestep(sc) / (0.25 * math.sqrt(sc.h / g)), 1.0, delta=1e-12)

    def _cloud(self):
        M, R, n = 1.0e12, 5000.0, 16
        bodies = seed_rotating_cloud(n, cloud_mass_kg=M, cloud_radius_m=R, omega_rad_s=0.0,
                                     particle_density_kg_m3=920.0, jitter_frac=0.0, seed=3)
        h = 1.5 * (4 / 3 * math.pi * R ** 3 / n) ** (1 / 3)
        return to_continuum(bodies, material_key="water_ice", G=G_CODATA, gravity_softening_m=0.02 * R,
                            eos="isothermal", sound_speed_m_s=1e-9, smoothing_h=h)

    def test_force_criterion_from_self_gravity_before_the_first_step(self):
        sc = self._cloud()
        ps = sc.continuum_parcels()
        eps2 = sc.gravity_softening_m ** 2
        a_max = 0.0                                           # Plummer-softened Newton, written here
        for i, pi_ in enumerate(ps):
            ax = ay = az = 0.0
            for j, pj in enumerate(ps):
                if i != j:
                    dx = pj.position.x - pi_.position.x
                    dy = pj.position.y - pi_.position.y
                    dz = pj.position.z - pi_.position.z
                    k = G_CODATA * pj.mass / (dx * dx + dy * dy + dz * dz + eps2) ** 1.5
                    ax, ay, az = ax + k * dx, ay + k * dy, az + k * dz
            a_max = max(a_max, math.sqrt(ax * ax + ay * ay + az * az))
        self.assertAlmostEqual(cfl_timestep(sc) / (0.25 * math.sqrt(sc.h / a_max)), 1.0, delta=1e-6)

    def test_cfl_timestep_leaves_the_run_unchanged(self):
        def run(probe):
            sc = self._cloud()
            sc.sound_speed_m_s = 0.03
            if probe:
                cfl_timestep(sc)
            for _ in range(5):
                continuum_step(sc, 500.0)
            return [(p.position.x, p.position.y, p.velocity.z, p.sph_density) for p in sc.continuum_parcels()]
        self.assertEqual(run(False), run(True))

    @staticmethod
    def _iron_lattice(mass_scale, n=5, dx=0.01, seed=0):
        rng = random.Random(seed)
        ps = []
        for ix in range(n):
            for iy in range(n):
                for iz in range(n):
                    p = ContinuumParcel("iron", dx * (3 / (4 * math.pi)) ** (1 / 3),
                                        temperature=300.0 + rng.uniform(-5.0, 5.0),
                                        position=Vec3(ix * dx, iy * dx, iz * dx))
                    p.mass *= mass_scale
                    ps.append(p)
        return ps

    def test_conduction_obeys_the_maximum_principle_at_the_cfl_step(self):
        # A ~1 kg/m³ gas with an iron carrier (κ, c_p of iron): conduction, not
        # sound, sets the time scale. Under eos="isothermal" T changes only by
        # conduction, and Fourier's law makes no new extremum (maximum principle
        # of the heat equation): every T must stay inside the initial range.
        ps = self._iron_lattice(1.0 / 7874.0)
        sc = ContinuumScene(ps, gravity=ZERO, ground=False, eos="isothermal", sound_speed_m_s=1.0)
        t_lo = min(p.temperature for p in ps)
        t_hi = max(p.temperature for p in ps)
        for step in range(20):
            continuum_step(sc, cfl_timestep(sc))
            temps = [p.temperature for p in ps]
            self.assertGreaterEqual(min(temps), t_lo - 1e-9, step)
            self.assertLessEqual(max(temps), t_hi + 1e-9, step)
        spread = max(temps) - min(temps)
        self.assertLess(spread, 0.5 * (t_hi - t_lo))          # and heat really flowed

    def test_conduction_criterion_does_not_bind_for_condensed_matter(self):
        sc = ContinuumScene(self._iron_lattice(1.0), gravity=Vec3(0.0, -G_STD, 0.0), ground=False)
        dt_with = cfl_timestep(sc)
        sc.heat_conduction = False
        self.assertEqual(dt_with, cfl_timestep(sc))


def _gas_ball(n=40, seed=2):
    rng = random.Random(seed)
    ps = []
    while len(ps) < n:
        x, y, z = [rng.uniform(-1, 1) for _ in range(3)]
        if x * x + y * y + z * z <= 1:
            ps.append(ContinuumParcel("water_ice", 0.05, temperature=300.0, position=Vec3(x, y, z), velocity=ZERO))
    h = 1.5 * (4 / 3 * math.pi / n) ** (1 / 3)
    sc = ContinuumScene(ps, gravity=ZERO, ground=False, smoothing_h=h, eos="ideal",
                        ideal_molar_mass_kg_mol=M_N2, ideal_gamma=GAMMA_N2)
    sc.heat_conduction = False
    return sc, ps


class TestIdealGasTimeCentred(unittest.TestCase):
    T_END_STEPS_AT_CFL = 40                     # 40 steps of 0.15 h/c_s

    def _t_end(self):
        sc, _ = _gas_ball()
        return self.T_END_STEPS_AT_CFL * 0.15 * sc.h / math.sqrt(GAMMA_N2 * R_GAS * 300.0 / M_N2)

    def test_loschmidt_echo_returns_to_the_start(self):
        sc, ps = _gas_ball()
        x0 = [(p.position.x, p.position.y, p.position.z, p.temperature) for p in ps]
        steps = self.T_END_STEPS_AT_CFL
        dt = self._t_end() / steps
        for _ in range(steps):
            continuum_step(sc, dt, alpha_visc=0.0)
        moved = max(abs(p.position.x - a[0]) for p, a in zip(ps, x0))
        self.assertGreater(moved, 0.05)                        # the gas really expanded
        for p in ps:
            p.velocity = Vec3(-p.velocity.x, -p.velocity.y, -p.velocity.z)
        for _ in range(steps):
            continuum_step(sc, dt, alpha_visc=0.0)
        rms = math.sqrt(sum((p.position.x - a[0]) ** 2 + (p.position.y - a[1]) ** 2 + (p.position.z - a[2]) ** 2
                            for p, a in zip(ps, x0)) / len(ps))
        self.assertLess(rms, 1e-10)                            # R = 1 m
        self.assertLess(max(abs(p.temperature - a[3]) for p, a in zip(ps, x0)), 1e-8)

    def test_second_order_in_dt(self):
        def end_state(steps):
            sc, ps = _gas_ball()
            dt = self._t_end() / steps
            for _ in range(steps):
                continuum_step(sc, dt, alpha_visc=0.0)
            return [(p.position.x, p.position.y, p.position.z) for p in ps]

        def rms(a, b):
            return math.sqrt(sum((u[k] - v[k]) ** 2 for u, v in zip(a, b) for k in range(3)) / len(a))

        s1, s2, s4 = end_state(20), end_state(40), end_state(80)
        ratio = rms(s1, s2) / rms(s2, s4)                      # 2^p for a p-th order scheme
        self.assertTrue(3.5 <= ratio <= 5.0, ratio)


if __name__ == "__main__":
    unittest.main()
