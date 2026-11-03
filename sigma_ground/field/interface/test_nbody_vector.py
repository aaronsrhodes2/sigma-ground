"""The vectorised force routine (NBodySystem.force_backend = "vector", 2026-11-02) is the per-pair loop routine to rounding.

What is tested is the loop's answer: the stored predictors were all measured with the loops, so the vector path may only change the speed. Each
layer is compared on a unit system that exercises it (softening, gr_1pn, gr_2pn, the three zonals with a tilted pole and the reaction, the static
tide, coincident bodies, a massless body) and, when the DE440 kernel is installed, on the 11-body reduced model with every layer of the served
predictor, over a short Yoshida-6 and Forest-Ruth propagation (the integrators' state updates are part of the path).
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem, PhysicsToggles

KERNEL = Path("D:/datasets/jpl/de440s.bsp")
RTOL = 1e-12


def _both(system: NBodySystem):
    """(loop, vector) accelerations of the same system; the system is left on its previous backend."""
    keep = system.force_backend
    system.force_backend = "loop"
    a_loop = system.compute_accelerations()
    system.force_backend = "vector"
    a_vec = system.compute_accelerations()
    system.force_backend = keep
    return a_loop, a_vec


def _assert_same(test, a_loop, a_vec, where: str):
    scale = np.linalg.norm(a_loop, axis=1)[:, None]
    scale = np.where(scale > 0.0, scale, 1.0)
    rel = np.abs(a_vec - a_loop) / scale
    test.assertLess(float(rel.max()), RTOL, f"{where}: max relative difference {rel.max():.3e}")


def _unit_system(toggles: PhysicsToggles, softening: float = 0.0, coincident: bool = False, massless: bool = False, scale_m: float = 1.5e11) -> NBodySystem:
    rng = np.random.default_rng(7)
    bodies = []
    poles = [np.array([0.0, 0.0, 1.0]), np.array([0.3, -0.2, 0.93]), np.array([-0.5, 0.5, 0.7]), np.array([0.0, 1.0, 0.0]), np.array([0.1, 0.1, 0.99])]
    for k in range(5):
        p = rng.normal(size=3) * scale_m
        v = rng.normal(size=3) * 2e4
        gone = massless and k == 4                                             # a massless body carries no figure or tide (the loop divides by its mass otherwise)
        bodies.append(CelestialBody(mass_kg=[1.989e30, 5.97e24, 7.35e22, 6.4e23, 1.9e27][k] * (0.0 if gone else 1.0), position_m=p, velocity_m_s=v,
                                    radius_m=[6.96e8, 6.371e6, 1.737e6, 3.39e6, 6.99e7][k], love_number_k2=[0.03, 0.3, 0.024, 0.14, 0.0][k],
                                    j2=[2e-7, 1.083e-3, 2.03e-4, 1.96e-3, 0.0 if gone else 1.47e-2][k], j3=[0.0, -2.5e-6, 8.5e-6, 3.1e-5, 0.0][k], j4=[0.0, -1.6e-6, -1.0e-7, -1.5e-5, 0.0 if gone else -5.9e-4][k],
                                    pole_axis_unit=poles[k], field_radius_m=[0.0, 6.378e6, 1.738e6, 0.0, 0.0][k]))
    if coincident:
        bodies[3] = bodies[3].replace(position_m=bodies[1].position_m.copy())
    return NBodySystem(bodies, toggles=toggles, softening_m=softening)


class TestVectorEqualsLoop(unittest.TestCase):
    def test_newton_only(self):
        s = _unit_system(PhysicsToggles())
        _assert_same(self, *_both(s), "Newton")

    def test_newton_with_softening_and_coincident_bodies(self):                     # the loop skips r == 0 and adds the softening to r_sq
        for soft in (0.0, 1e6):
            s = _unit_system(PhysicsToggles(), softening=soft, coincident=True)
            _assert_same(self, *_both(s), f"Newton, softening {soft}, coincident pair")

    def test_eih(self):
        s = _unit_system(PhysicsToggles(eih_cross=True))
        a_loop, a_vec = _both(s)
        _assert_same(self, a_loop, a_vec, "EIH")
        s2 = _unit_system(PhysicsToggles())                                            # the 1PN part must be present, not masked away
        self.assertGreater(float(np.abs(a_vec - s2.compute_accelerations()).max()), 0.0)

    def test_gr_1pn_and_2pn(self):
        s = _unit_system(PhysicsToggles(gr_1pn=True, gr_2pn=True))
        _assert_same(self, *_both(s), "gr_1pn + gr_2pn")

    def test_zonals_with_reaction_and_tilted_poles(self):
        s = _unit_system(PhysicsToggles(j2_zonal=True, j3_zonal=True, j4_zonal=True))
        a_loop, a_vec = _both(s)
        _assert_same(self, a_loop, a_vec, "J2 + J3 + J4")
        m = np.array([b.mass_kg for b in s.bodies])
        self.assertLess(float(np.linalg.norm((m[:, None] * a_vec).sum(axis=0))) / float(np.linalg.norm(m[:, None] * a_vec, axis=1).max()), 1e-12, "zonal reaction conserves momentum")

    def test_zonals_with_a_massless_body(self):                                      # the reaction divides by m_j only where the loop does
        s = _unit_system(PhysicsToggles(j2_zonal=True, j3_zonal=True, j4_zonal=True), massless=True)
        a_loop, a_vec = _both(s)
        self.assertTrue(np.all(np.isfinite(a_vec)))
        _assert_same(self, a_loop, a_vec, "zonals, massless body")

    def test_static_tide(self):
        s = _unit_system(PhysicsToggles(tidal_force=True), scale_m=4e8)             # a close system: at 1 au the tide is below the double's resolution of the Newton term
        a_loop, a_vec = _both(s)
        _assert_same(self, a_loop, a_vec, "tide")
        s_off = _unit_system(PhysicsToggles(), scale_m=4e8)
        self.assertGreater(float(np.abs(a_vec - s_off.compute_accelerations()).max()), 0.0, "the tide is present")

    def test_everything_at_once(self):
        s = _unit_system(PhysicsToggles(eih_cross=True, gr_2pn=True, j2_zonal=True, j3_zonal=True, j4_zonal=True, tidal_force=True), softening=10.0)
        _assert_same(self, *_both(s), "all pair layers")

    def test_unknown_backend_is_refused(self):
        s = _unit_system(PhysicsToggles())
        s.force_backend = "cuda"
        with self.assertRaises(ValueError):
            s.compute_accelerations()


@unittest.skipUnless(KERNEL.exists(), "DE440s kernel not available")
class TestVectorOnTheReducedModel(unittest.TestCase):
    """The served model (every layer of restated_zon: EIH, zonals, tides, the Moon's C22, 409 prescribed sources, SRP) on both backends."""

    @classmethod
    def setUpClass(cls):
        from jplephem.spk import SPK
        cls.kernel = SPK.open(str(KERNEL))

    def _system(self, predictor: str):
        from sigma_ground.field.interface import reduced_model as rm
        return rm.build_system(self.kernel, 2460676.5, predictor, dt_days=0.1, horizon_days=10.0)[0]

    def test_one_evaluation_every_served_predictor(self):
        for pred in ("restated_zon", "restated_geo", "restated_hdr", "restated_tide", "restated_gmdot", "de440_stack"):
            _assert_same(self, *_both(self._system(pred)), pred)

    def test_short_propagation_both_integrators(self):                                # the integrators' state updates (moved()) are on the path too
        from sigma_ground.field.interface import reduced_model as rm
        for integ, dt in (("y6", 0.1), ("fr4", 0.025)):
            out = {}
            for backend in ("loop", "vector"):
                s = self._system("restated_zon")
                s.force_backend = backend
                d = rm.propagate_dense(s, 2460676.5, 2460676.5 + 1.0, dt, every=10 ** 9, integrator=integ)
                out[backend] = d["pos_km"][-1]
            diff = np.linalg.norm(out["vector"] - out["loop"], axis=1)
            self.assertLess(float(diff.max()), 1e-6, f"{integ}: after one day the two backends differ by {diff.max():.3e} km")   # 1 mm; measured 2026-11-02: below 1e-7 km


if __name__ == "__main__":
    unittest.main()
