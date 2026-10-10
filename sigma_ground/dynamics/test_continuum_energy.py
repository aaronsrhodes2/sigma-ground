"""The SPH continuum bridge has a first law (KNOWN_GAPS 2026-10-10, closed the same day).

continuum_step now books the energy-conserving SPH energy equation
    duᵢ/dt = ½ Σⱼ mⱼ (Pᵢ/ρᵢ² + Pⱼ/ρⱼ² + Πᵢⱼ) vᵢⱼ·∇ᵢWᵢⱼ
(PdV work + artificial-viscosity dissipation) with the same trapezoid as the
velocity kick. These tests pin:

  1. Ledger closure that CONVERGES — isothermal pair collision: K + bath is
     conserved up to an error that falls with dt (first order: the viscous
     force sees the step's lagged velocities), and the viscous loss is booked
     to the bath instead of vanishing (the Phase E "−387 % mechanical energy"
     symptom).
  2. Ledger closure with self-gravity — a collapsing isothermal cloud:
     K + U_grav(softened) + bath is conserved to integrator tolerance.
  3. The energy equation never changes the dynamics (positions and velocities
     are bit-identical with the switch on or off), and the heat it adds to a
     Tait parcel is exactly the booked internal energy over c_p.
  4. energy_equation = False restores the pre-fix behaviour (no heating, no bath).

Note on test 3's setup: two isolated Tait parcels sit below ρ₀, so the EOS puts
them in strong tension (the "dilute Tait" regime limit documented in
_pressure_from_eos) and they accelerate together; the PdV term converts that
work to (negative) heat correctly. Short runs only — the regime is unphysical
for long ones, which is the documented limit, not this feature.
"""
import math
import unittest

from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.continuum import ContinuumParcel, ContinuumScene, continuum_step
from sigma_ground.dynamics.seeding import seed_rotating_cloud, to_continuum, G_CODATA

ZERO = Vec3(0.0, 0.0, 0.0)


def _pair(material, sep, v, radius=0.01, T=300.0):
    a = ContinuumParcel(material, radius, temperature=T, position=Vec3(-sep / 2, 0, 0), velocity=Vec3(v, 0, 0), label="a")
    b = ContinuumParcel(material, radius, temperature=T, position=Vec3(sep / 2, 0, 0), velocity=Vec3(-v, 0, 0), label="b")
    return [a, b]


def _iso_pair_run(dt, t_end=0.08):
    scene = ContinuumScene(_pair("water_ice", 0.08, 0.5), gravity=ZERO, ground=False,
                           smoothing_h=0.05, eos="isothermal", sound_speed_m_s=1.0)
    e0 = scene.energy_ledger()
    for _ in range(int(round(t_end / dt))):
        continuum_step(scene, dt)
    return scene, e0, scene.energy_ledger()


class TestIsothermalPairLedger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coarse = _iso_pair_run(2e-4)
        cls.fine = _iso_pair_run(5e-5)

    def test_dissipation_is_booked_not_lost(self):
        _, e0, e1 = self.fine
        self.assertLess(e1["kinetic"], e0["kinetic"])                  # the collision dissipated
        self.assertGreater(e1["bath"], 0.0)                            # ...into the bath
        self.assertLess(abs(e1["total"] - e0["total"]), 1e-2 * e0["kinetic"])

    def test_ledger_error_converges_with_dt(self):
        err_c = abs(self.coarse[2]["total"] - self.coarse[1]["total"])
        err_f = abs(self.fine[2]["total"] - self.fine[1]["total"])
        self.assertGreater(err_c / err_f, 3.0)                         # 4× smaller dt → ≥ 3× smaller error

    def test_isothermal_temperature_is_held(self):
        scene = self.fine[0]
        self.assertEqual([p.temperature for p in scene.parcels], [300.0, 300.0])


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
        self.assertGreater(e1["kinetic"], 0.0)                          # it fell
        self.assertLess(abs(e1["total"] - e0["total"]), 2e-3 * abs(e0["self_gravity"]))


class TestDynamicsUnchangedHeatConsistent(unittest.TestCase):
    def _run(self, on, steps=20, dt=1e-7):
        ps = _pair("iron", 0.018, 0.0)
        scene = ContinuumScene(ps, gravity=ZERO, ground=False, smoothing_h=0.012)
        scene.energy_equation = on
        e0 = scene.energy_ledger()
        for _ in range(steps):
            continuum_step(scene, dt)
        return scene, e0

    def test_positions_velocities_bit_identical(self):
        on, _ = self._run(True)
        off, _ = self._run(False)
        self.assertEqual([(p.position.x, p.velocity.x) for p in on.parcels],
                         [(p.position.x, p.velocity.x) for p in off.parcels])

    def test_heat_equals_booked_internal_energy(self):
        probe = _pair("iron", 0.018, 0.0)[0]
        probe.update_properties()
        cp_start = probe.specific_heat                                   # c_p at 300 K
        on, e0 = self._run(True)
        off, _ = self._run(False)
        for p_on, p_off in zip(on.parcels, off.parcels):
            dT = p_on.temperature - p_off.temperature
            self.assertNotEqual(dT, 0.0)
            u = p_on.internal_energy / p_on.mass                          # J/kg booked by the first law
            cp_end = p_on.specific_heat
            lo, hi = sorted((u / cp_start, u / cp_end))
            # each step heats by du/c_p(T_step); c_p moves monotonically over the run, so ΔT is bracketed
            self.assertTrue(lo - 1e-9 * abs(dT) <= dT <= hi + 1e-9 * abs(dT), (lo, dT, hi))
        # ledger: first-order in dt (the end-of-step rate sees the pre-kick velocity) — so test convergence
        err_c = abs(on.energy_ledger()["total"] - e0["total"])
        fine, e0f = self._run(True, steps=80, dt=2.5e-8)
        err_f = abs(fine.energy_ledger()["total"] - e0f["total"])
        self.assertGreater(err_c / err_f, 3.0)


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
