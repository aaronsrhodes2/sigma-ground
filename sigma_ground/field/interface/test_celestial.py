"""celestial.py: two-body celestial mechanics from DE440 states, each result checked against an INDEPENDENT computation (a constructed orbit, the kernel's own motion, a numerical integration,
a force balance), not against the same formula written twice.

The planet tests are skipped where the JPL kernel is not installed; the moon routes use the snapshots in the repo and always run."""
from __future__ import annotations

import math
import unittest

import numpy as np

from sigma_ground.field.interface import celestial as C

try:
    from sigma_ground.field.interface import where_when as _ww
    _KERNEL = _ww._kernel() is not None
except Exception:                                                     # pragma: no cover
    _KERNEL = False
needs_kernel = unittest.skipUnless(_KERNEL, "the JPL DE440 kernel is not installed here")

JD_2026 = 2461000.5          # TDB, inside the kernel and within a year of the last snapshot
DAY = 86400.0


def _rot(i, node, argp):
    ci, si, cn, sn, cw, sw = math.cos(i), math.sin(i), math.cos(node), math.sin(node), math.cos(argp), math.sin(argp)
    return np.array([[cn * cw - sn * sw * ci, -cn * sw - sn * cw * ci, sn * si],
                     [sn * cw + cn * sw * ci, -sn * sw + cn * cw * ci, -cn * si],
                     [sw * si, cw * si, ci]])


def _state_from_elements(a, e, i, node, argp, nu, mu):
    """Textbook perifocal -> inertial construction, independent of celestial.elements_from_state."""
    p = a * (1 - e * e)
    r = p / (1 + e * math.cos(nu))
    rp = np.array([r * math.cos(nu), r * math.sin(nu), 0.0])
    vp = math.sqrt(mu / p) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])
    R = _rot(i, node, argp)
    return R @ rp, R @ vp


class TestElements(unittest.TestCase):
    def test_a_constructed_orbit_is_recovered(self):
        mu = 398600.4355
        for (a, e, i, node, argp, nu) in [(26560.0, 0.01, 0.96, 1.2, 0.7, 2.0), (42164.0, 0.3, 0.4, 5.5, 3.1, 0.1), (8000.0, 0.7, 2.5, 0.3, 6.0, 4.0)]:
            r, v = _state_from_elements(a, e, i, node, argp, nu, mu)
            el = C.elements_from_state(r, v, mu)
            self.assertAlmostEqual(el["a_km"] / a, 1.0, places=11)
            self.assertAlmostEqual(el["e"], e, places=11)
            self.assertAlmostEqual(el["i_deg"], math.degrees(i), places=8)
            self.assertAlmostEqual(el["node_deg"], math.degrees(node) % 360, places=8)
            self.assertAlmostEqual(el["argument_of_periapsis_deg"], math.degrees(argp) % 360, places=7)
            self.assertAlmostEqual(el["period_days"] * DAY, 2 * math.pi * math.sqrt(a ** 3 / mu), delta=1e-6)
            self.assertAlmostEqual(el["periapsis_km"], a * (1 - e), delta=1e-6 * a)

    def test_a_hyperbolic_state_has_no_period(self):
        mu = 1.0e5
        el = C.elements_from_state([7000.0, 0.0, 0.0], [0.0, 13.0, 0.0], mu)        # above escape speed sqrt(2 mu / r) = 5.35 km/s
        self.assertLess(el["a_km"], 0)
        self.assertNotIn("period_days", el)
        self.assertGreater(el["e"], 1)

    def test_small_bodies_and_a_body_about_itself_are_refused(self):
        with self.assertRaises(C.CelestialError):
            C.canonical("Ceres")                                                   # small bodies are in small_body
        with self.assertRaises(C.CelestialError):
            C.relative_state("Mars", "Mars", JD_2026)


class TestMoonsFromSnapshots(unittest.TestCase):
    def test_io_elements_and_the_step_counts_the_ledger_reports(self):
        el = C.elements("Io", "Jupiter", JD_2026)
        mu = el["mu_km3_s2"]
        self.assertAlmostEqual(el["period_days"], 2 * math.pi * math.sqrt(el["a_km"] ** 3 / mu) / DAY, places=12)
        self.assertAlmostEqual(el["period_days"], 1.769, delta=0.01)               # Io's sidereal period, textbook 1.769 d; osculating, so a 0.6 % band
        self.assertIn("snapshot", el["note"])                                      # the epoch used is stated
        rows = {r["body"]: r for r in C.step_rows("Jupiter", 0.0125 * DAY, JD_2026)}
        self.assertAlmostEqual(rows["Io"]["steps_per_orbit"], 142, delta=2)       # C19: "142 steps per orbit" at 0.0125 d
        self.assertFalse(rows["Io"]["adequate"])
        self.assertTrue(rows["Ganymede"]["adequate"])
        self.assertAlmostEqual(C.step_rows("Jupiter", 0.025 * DAY, JD_2026)[2]["steps_per_orbit"], 71, delta=1)   # C19: "Io at 71 steps per orbit"
        pers = [r["period_days"] for r in rows.values()]
        self.assertEqual(pers, sorted(pers))                                       # shortest first

    def test_a_date_far_from_the_snapshots_is_refused_for_moons(self):
        with self.assertRaises(C.CelestialError):
            C.elements("Io", "Jupiter", 2400000.0)

    def test_roche_rigid_is_the_root_of_tidal_stretch_equals_self_gravity(self):
        from scipy.optimize import brentq
        r = C.roche_limit("Moon", "Earth")
        gm_p, gm_s, rs = r["gm_primary"], r["gm_satellite"], r["satellite_radius_km"]
        d = brentq(lambda d: 2 * gm_p * rs / d ** 3 - gm_s / rs ** 2, rs, 1e8)      # the DEFINITION, solved numerically
        self.assertAlmostEqual(r["rigid_km"] / d, 1.0, places=10)
        self.assertAlmostEqual(r["fluid_km"] / r["rigid_km"], C.ROCHE_FLUID_COEFF / 2 ** (1 / 3), places=12)

    def test_roche_equals_the_engines_method(self):
        from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem
        r = C.roche_limit("Moon", "Earth")
        G = 6.6743e-11
        bodies = [CelestialBody(r["gm_primary"] * 1e9 / G, np.zeros(3), np.zeros(3), 6.371e6, 0.0),
                  CelestialBody(r["gm_satellite"] * 1e9 / G, np.array([3.8e8, 0.0, 0.0]), np.zeros(3), r["satellite_radius_km"] * 1e3, 0.0)]
        s = NBodySystem(bodies)
        self.assertAlmostEqual(s.roche_limit(0, 1) / 1e3 / r["rigid_km"], 1.0, places=9)
        self.assertAlmostEqual(s.roche_limit(0, 1, fluid=True) / 1e3 / r["fluid_km"], 1.0, places=9)

    def test_a_moon_with_no_published_mass_is_flagged_not_zeroed(self):
        el = C.elements("Pan", "Saturn", JD_2026)
        self.assertGreater(el["mu_km3_s2"], 0)
        self.assertIn("mass is not published", el["note"])

    def test_the_mass_ratio_of_io_about_jupiter_is_far_inside_the_stable_range(self):
        io = C.lagrange_points("Io", "Jupiter", JD_2026)
        self.assertTrue(io["L4_L5_stable"])
        self.assertLess(io["mass_ratio"], 1e-4)                                   # Io is 4.7e-5 of Jupiter + Io, far below the Routh limit 0.0385

    def test_pluto_charon_is_beyond_the_routh_limit(self):
        """Charon is half Pluto's size: m2/(m1+m2) is about 0.11, three times the limit 0.0385, so L4 and L5 are not stable for that pair."""
        lp = C.lagrange_points("Charon", "Pluto", JD_2026)
        self.assertGreater(lp["mass_ratio"], C.ROUTH_CRITICAL_MU)
        self.assertFalse(lp["L4_L5_stable"])


@needs_kernel
class TestPlanetsFromTheKernel(unittest.TestCase):
    def _mean_period_days(self, body, years=30):
        """Mean sidereal period from the kernel's own motion: slope of the unwrapped heliocentric longitude, independent of any osculating element."""
        from sigma_ground.field.interface import where_when as ww, reduced_model as rm
        k = ww._kernel()
        jds = np.arange(JD_2026 - years * 365.25, JD_2026, 5.0)
        lon = []
        for jd in jds:
            if body == "Earth":
                p, _ = k[0, 3].compute_and_differentiate(jd)                      # Earth-Moon barycentre
            else:
                p, _ = rm.kernel_state(k, body, jd)
            s, _ = k[0, 10].compute_and_differentiate(jd)
            d = np.asarray(p) - np.asarray(s)
            lon.append(math.atan2(d[1], d[0]))
        slope = np.polyfit(jds, np.unwrap(lon), 1)[0]                              # rad/day
        return 2 * math.pi / slope

    def test_osculating_periods_agree_with_the_mean_motion_in_the_kernel(self):
        for body, tol in (("Earth", 2e-3), ("Mars", 2e-3), ("Jupiter", 5e-3)):
            el = C.elements(body, "Sun", JD_2026)
            self.assertAlmostEqual(el["period_days"] / self._mean_period_days(body), 1.0, delta=tol, msg=body)

    def test_earth_about_the_sun_is_the_earth_moon_barycentre(self):
        el = C.elements("Earth", "Sun", JD_2026)
        self.assertIn("Earth-Moon barycentre", el["source"])
        self.assertAlmostEqual(el["period_days"], 365.25, delta=0.3)

    def test_hill_radius_and_sphere_of_influence_of_the_earth(self):
        h = C.hill_radius("Earth", "Sun", JD_2026)
        gm_s = 132712440041.93938
        m = h["mu_km3_s2"] - gm_s
        self.assertAlmostEqual(h["hill_radius_at_a_km"], h["a_km"] * (m / (3 * gm_s)) ** (1 / 3), delta=1e-6 * h["hill_radius_at_a_km"])   # arithmetic from the DE440 header
        self.assertAlmostEqual(h["hill_radius_at_a_km"] / 1e6, 1.5, delta=0.05)    # order check: the Earth's Hill sphere is about 1.5 million km
        s = C.sphere_of_influence("Earth", "Sun", JD_2026)
        self.assertAlmostEqual(s["sphere_of_influence_km"] / 1e6, 0.93, delta=0.05)
        self.assertLess(s["sphere_of_influence_km"], h["hill_radius_at_a_km"])

    def test_lagrange_points_are_equilibria_of_the_rotating_frame(self):
        """The returned positions must make the net force vanish in the frame rotating with the pair: checked with the full force, not the equation the solver used."""
        for body, central in (("Earth", "Sun"), ("Moon", "Earth"), ("Jupiter", "Sun")):
            lp = C.lagrange_points(body, central, JD_2026)
            gm_c = C.gm_of(lp["central"])[0]
            gm_b = lp["mu_km3_s2"] - gm_c
            sep = lp["separation_km"]
            n2 = lp["mu_km3_s2"] / sep ** 3                                         # angular rate squared of the pair
            xc, xb = -gm_b / lp["mu_km3_s2"] * sep, gm_c / lp["mu_km3_s2"] * sep      # barycentric x of central and body along the line

            def force(x, y):
                f = np.array([n2 * x, n2 * y])
                for gm, px in ((gm_c, xc), (gm_b, xb)):
                    d = np.array([x - px, y])
                    f -= gm * d / np.linalg.norm(d) ** 3
                return f
            scale = lp["mu_km3_s2"] / sep ** 2
            for key in ("L1", "L2", "L3"):
                x = lp[f"{key}_km_from_central"] + xc
                self.assertLess(np.linalg.norm(force(x, 0.0)) / scale, 1e-8, msg=f"{body}/{central} {key}")
            for s in (1, -1):
                x, y = sep * 0.5 + xc, s * sep * math.sqrt(3) / 2
                self.assertLess(np.linalg.norm(force(x, y)) / scale, 1e-8, msg=f"{body}/{central} L{4 if s > 0 else 5}")
            order = [lp["L1_km_from_central"], sep, lp["L2_km_from_central"]]
            self.assertEqual(order, sorted(order))
            self.assertTrue(lp["L4_L5_stable"])

    def test_synodic_period_equals_the_measured_interval_between_alignments(self):
        from sigma_ground.field.interface import where_when as ww, reduced_model as rm
        k = ww._kernel()
        jds = np.arange(JD_2026 - 40 * 365.25, JD_2026, 2.0)
        ang = []
        for jd in jds:
            pe, _ = k[0, 3].compute_and_differentiate(jd)
            pm, _ = rm.kernel_state(k, "Mars", jd)
            s, _ = k[0, 10].compute_and_differentiate(jd)
            de, dm = np.asarray(pe) - np.asarray(s), np.asarray(pm) - np.asarray(s)
            ang.append(math.atan2(de[1], de[0]) - math.atan2(dm[1], dm[0]))
        slope = np.polyfit(jds, np.unwrap(ang), 1)[0]                               # rad/day, relative angular rate
        measured = 2 * math.pi / abs(slope)
        res = C.synodic_period_days("Earth", "Mars", "Sun", JD_2026)
        self.assertAlmostEqual(res["synodic_days"] / measured, 1.0, delta=5e-3)

    def test_hohmann_burns_are_what_a_numerical_two_body_integration_needs(self):
        from scipy.integrate import solve_ivp
        h = C.hohmann("Earth", "Mars", "Sun", JD_2026)
        mu = C.gm_of("Sun")[0]
        r1, r2 = h["r_from_km"], h["r_to_km"]
        v0 = math.sqrt(mu / r1) + h["dv1_km_s"]                                     # tangential speed after the first burn

        def f(t, y):
            return np.concatenate([y[3:], -mu * y[:3] / np.linalg.norm(y[:3]) ** 3])
        sol = solve_ivp(f, (0, h["transfer_time_days"] * DAY), [r1, 0, 0, 0, v0, 0], method="DOP853", rtol=1e-12, atol=1e-9)
        y = sol.y[:, -1]
        self.assertAlmostEqual(np.linalg.norm(y[:3]) / r2, 1.0, delta=1e-7)         # arrives at the outer radius after the stated time
        self.assertAlmostEqual(math.sqrt(mu / r2) - np.linalg.norm(y[3:]), h["dv2_km_s"], delta=1e-6)   # and the second burn closes the gap to the circular speed
        self.assertAlmostEqual(h["dv_total_km_s"], h["dv1_km_s"] + h["dv2_km_s"], places=12)

    def test_gr_advance_closed_form_equals_what_the_engine_integrates_for_mercury(self):
        """The closed form here equals the perihelion advance the 1PN engine integrates (that engine value is pinned to the textbook one by test_gr_precession)."""
        from sigma_ground.field.interface import test_gr_precession as T
        from sigma_ground.field.interface.nbody import PhysicsToggles
        engine = T._perihelion_advance_per_orbit(PhysicsToggles(gr_1pn=True), n_orbits=2)
        mu_km3 = T._G * (T.M_SUN + T.M_MERCURY) / 1e9
        mine = C.gr_advance_per_orbit(mu_km3, T.A_MERCURY / 1e3, T.E_MERCURY)
        self.assertAlmostEqual(engine / mine, 1.0, delta=0.01)

    def test_gr_advance_of_mercury_from_de440_is_near_the_textbook_43_arcsec_per_century(self):
        el = C.gr_perihelion_advance("Mercury", "Sun", JD_2026)
        self.assertAlmostEqual(el["gr_advance_arcsec_per_century"], 43.0, delta=0.7)   # a sanity band around the textbook value; its source is to be read before the number is quoted


if __name__ == "__main__":
    unittest.main()
