"""The SPH continuum bridge has a first law (KNOWN_GAPS 2026-10-10, closed the same day).

continuum_step books the work of the internal SPH forces (pressure + artificial
viscosity) as internal energy pair by pair, with the same pair forces the kick
applied and the step-averaged velocities:
    Δuᵢ = ¼ Δt Σ_{evaluations} Σⱼ gᵢⱼ·(v̄ᵢ − v̄ⱼ)
Since mᵢgᵢⱼ = −mⱼgⱼᵢ this conserves K + U_int EXACTLY (to roundoff) for those
forces. (A first implementation used the end-of-step rate with the pre-kick
velocity; a free-expanding gas then gained 24 % more kinetic energy than it lost
in heat — these tests would have caught it.)

  1. Isothermal pair collision: K + bath conserved to roundoff; viscous loss
     booked to the bath, not lost.
  2. Tait pair (strong tension — the dilute-Tait regime limit): K + U_int to
     roundoff; dynamics bit-identical with the switch on or off; ΔT bracketed
     by Δu/c_p(start..end).
  3. Ideal gas (eos="ideal"): K + U_int to roundoff and T = T₀ + u/c_v exactly.
  4. Self-gravitating isothermal cloud: K + U_grav(softened) + bath conserved
     to the velocity-Verlet tolerance (gravity is in the potential ledger).
  5. energy_equation = False restores the pre-fix behaviour.
"""
import math
import random
import unittest

from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.continuum import ContinuumParcel, ContinuumScene, continuum_step
from sigma_ground.dynamics.seeding import seed_rotating_cloud, to_continuum, G_CODATA
from sigma_ground.field.constants import R_GAS

ZERO = Vec3(0.0, 0.0, 0.0)


def _pair(material, sep, v, radius=0.01, T=300.0):
    a = ContinuumParcel(material, radius, temperature=T, position=Vec3(-sep / 2, 0, 0), velocity=Vec3(v, 0, 0), label="a")
    b = ContinuumParcel(material, radius, temperature=T, position=Vec3(sep / 2, 0, 0), velocity=Vec3(-v, 0, 0), label="b")
    return [a, b]


def _rel(e0, e1, scale):
    return abs(e1["total"] - e0["total"]) / scale


class TestIsothermalPairLedger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = ContinuumScene(_pair("water_ice", 0.08, 0.5), gravity=ZERO, ground=False,
                                   smoothing_h=0.05, eos="isothermal", sound_speed_m_s=1.0)
        cls.e0 = cls.scene.energy_ledger()
        for _ in range(400):
            continuum_step(cls.scene, 2e-4)
        cls.e1 = cls.scene.energy_ledger()

    def test_ledger_exact(self):
        self.assertLess(_rel(self.e0, self.e1, self.e0["kinetic"]), 1e-10)

    def test_dissipation_is_booked_not_lost(self):
        self.assertLess(self.e1["kinetic"], self.e0["kinetic"])
        self.assertGreater(self.e1["bath"], 0.0)

    def test_isothermal_temperature_is_held(self):
        self.assertEqual([p.temperature for p in self.scene.parcels], [300.0, 300.0])


class TestTaitPair(unittest.TestCase):
    def _run(self, on, steps=20, dt=1e-7):
        scene = ContinuumScene(_pair("iron", 0.018, 0.0), gravity=ZERO, ground=False, smoothing_h=0.012)
        scene.energy_equation = on
        e0 = scene.energy_ledger()
        for _ in range(steps):
            continuum_step(scene, dt)
        return scene, e0

    def test_ledger_exact(self):
        on, e0 = self._run(True)
        e1 = on.energy_ledger()
        self.assertGreater(abs(e1["internal"]), 0.0)
        self.assertLess(_rel(e0, e1, abs(e1["internal"])), 1e-10)

    def test_positions_velocities_bit_identical(self):
        on, _ = self._run(True)
        off, _ = self._run(False)
        self.assertEqual([(p.position.x, p.velocity.x) for p in on.parcels],
                         [(p.position.x, p.velocity.x) for p in off.parcels])

    def test_heat_bracketed_by_booked_internal_energy(self):
        probe = _pair("iron", 0.018, 0.0)[0]
        probe.update_properties()
        cp_start = probe.specific_heat
        on, _ = self._run(True)
        off, _ = self._run(False)
        for p_on, p_off in zip(on.parcels, off.parcels):
            dT = p_on.temperature - p_off.temperature
            u = p_on.internal_energy / p_on.mass
            lo, hi = sorted((u / cp_start, u / p_on.specific_heat))
            self.assertTrue(lo - 1e-9 * abs(dT) <= dT <= hi + 1e-9 * abs(dT), (lo, dT, hi))


class TestIdealGas(unittest.TestCase):
    def test_free_expansion_ledger_exact_and_T_is_u_over_cv(self):
        M, gam = 0.028014, 1.4
        rng = random.Random(2); ps = []
        while len(ps) < 40:
            x, y, z = [rng.uniform(-1, 1) for _ in range(3)]
            if x * x + y * y + z * z <= 1:
                ps.append(ContinuumParcel("water_ice", 0.05, temperature=300.0, position=Vec3(x, y, z), velocity=ZERO))
        h = 1.5 * (4 / 3 * math.pi / 40) ** (1 / 3)
        scene = ContinuumScene(ps, gravity=ZERO, ground=False, smoothing_h=h, eos="ideal",
                               ideal_molar_mass_kg_mol=M, ideal_gamma=gam)
        scene.heat_conduction = False
        cs = math.sqrt(gam * R_GAS * 300.0 / M)
        e0 = scene.energy_ledger()
        for _ in range(60):
            continuum_step(scene, 0.15 * h / cs)
        e1 = scene.energy_ledger()
        self.assertGreater(e1["kinetic"], 0.0)                         # it expanded
        self.assertLess(e1["internal"], 0.0)                            # ...and cooled
        self.assertLess(_rel(e0, e1, e1["kinetic"]), 1e-10)
        cv = R_GAS / (M * (gam - 1.0))
        for p in ps:
            self.assertAlmostEqual(p.temperature, 300.0 + p.internal_energy / (p.mass * cv), delta=1e-9)

    def test_ideal_requires_parameters(self):
        with self.assertRaises(ValueError):
            ContinuumScene(_pair("water_ice", 0.08, 0.0), gravity=ZERO, ground=False, eos="ideal")


class TestSelfGravityLedger(unittest.TestCase):
    def test_collapsing_cloud_conserves_total(self):
        M, R = 1.0e12, 5000.0
        bodies = seed_rotating_cloud(16, cloud_mass_kg=M, cloud_radius_m=R, omega_rad_s=0.0,
                                     particle_density_kg_m3=920.0, jitter_frac=0.0, seed=3)
        h = 1.5 * (4 / 3 * math.pi * R ** 3 / 16) ** (1 / 3)
        cs = 0.3 * math.sqrt(G_CODATA * M / R)
        scene = to_continuum(bodies, material_key="water_ice", G=G_CODATA, gravity_softening_m=0.02 * R,
                             eos="isothermal", sound_speed_m_s=cs, smoothing_h=h)
        t_ff = math.sqrt(3 * math.pi / (32 * G_CODATA * 3 * M / (4 * math.pi * R ** 3)))
        e0 = scene.energy_ledger()
        for _ in range(120):
            continuum_step(scene, t_ff / 400)
        e1 = scene.energy_ledger()
        self.assertGreater(e1["kinetic"], 0.0)
        self.assertLess(_rel(e0, e1, abs(e0["self_gravity"])), 2e-3)


class TestSwitchOff(unittest.TestCase):
    def test_energy_equation_false_is_legacy(self):
        scene = ContinuumScene(_pair("water_ice", 0.08, 0.5), gravity=ZERO, ground=False,
                               smoothing_h=0.05, eos="isothermal", sound_speed_m_s=1.0)
        scene.energy_equation = False
        for _ in range(200):
            continuum_step(scene, 2e-4)
        self.assertEqual(scene.energy_bath_J, 0.0)
        self.assertEqual(sum(p.internal_energy for p in scene.parcels), 0.0)


if __name__ == "__main__":
    unittest.main()
