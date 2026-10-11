"""Tests for local_library.interface.adapters (sg_ physics patch layer).

Verifies:
  - SGConstants are more precise than astropy/scipy defaults
  - SGAdapter injects sg_ methods on any object
  - SGPhysicsMixin sg_gm / sg_mass work without external libs
  - SGEphemeris tier routing: DE440 for past, forecast for future
  - sg_solve_ivp Forest-Ruth: energy-conserving harmonic oscillator
  - sg_nbody: circular orbit conserves energy / angular momentum
  - sg_odeint: API compatibility
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from sigma_ground.field.interface.adapters import (
    SGAdapter, SGConstants, SGEphemeris, SGPhysicsMixin,
    sg_nbody, sg_odeint, sg_solve_ivp,
)
from sigma_ground.field.interface.adapters.constants import (
    sg_patch_astropy, sg_patch_scipy,
)
from sigma_ground.field.constants import G as _G


# ══════════════════════════════════════════════════════════════════════════
# SGConstants
# ══════════════════════════════════════════════════════════════════════════

class TestSGConstants(unittest.TestCase):

    def test_G_matches_codata(self):
        """G should be the CODATA 2018 value from sigma_ground.field.constants."""
        self.assertAlmostEqual(SGConstants.G, _G, places=20)

    def test_GM_sun_de440(self):
        """GM_sun must match the DE440 value, 132712440041.279419 km³/s²
        (JPL SSD astro_par; Park et al. 2021, AJ 161, 105).  Until
        2026-10-10 this test pinned 132712440041.93938, the DE430 value."""
        expected = 132_712_440_041.279419 * 1e9
        self.assertAlmostEqual(SGConstants.GM_sun, expected, delta=1e6)

    def test_GM_sun_more_precise_than_astropy(self):
        """DE440 GM_sun should differ from CODATA 2014 by < 1 ppm."""
        delta_ppm = SGConstants.delta_GM_sun_ppm()
        self.assertLess(delta_ppm, 1.0,
                        f"GM_sun delta from CODATA 2014: {delta_ppm:.3f} ppm")

    def test_M_sun_derived_from_GM(self):
        """M_sun = GM_sun / G -- derived, not independent."""
        expected = SGConstants.GM_sun / SGConstants.G
        self.assertAlmostEqual(SGConstants.M_sun, expected, delta=1e10)

    def test_light_year_derived(self):
        """Light-year = c * yr -- both from sigma_ground.field.constants."""
        from sigma_ground.field.constants import C, YEAR_S
        expected = C * YEAR_S
        self.assertAlmostEqual(SGConstants.ly, expected, delta=1e3)

    def test_exact_h_and_parsec(self):
        """h is exact in the SI since 2019 (6.62607015e-34 J s); the parsec is
        648000/pi au (IAU 2015 Resolution B2).  Before: -6.1e-10, -1.6e-10."""
        self.assertEqual(SGConstants.h, 6.62607015e-34)
        self.assertAlmostEqual(SGConstants.pc / (648_000 / math.pi * 1.495978707e11),
                               1.0, delta=1e-15)

    def test_sg_alias(self):
        """sg_ aliases must equal their non-prefixed counterparts."""
        self.assertEqual(SGConstants.sg_G,      SGConstants.G)
        self.assertEqual(SGConstants.sg_GM_sun, SGConstants.GM_sun)
        self.assertEqual(SGConstants.sg_au,     SGConstants.au)

    def test_all_gm_dict(self):
        """all_gm() must include Sun, Earth, Jupiter."""
        gm = SGConstants.all_gm()
        for body in ("Sun", "Earth", "Jupiter"):
            self.assertIn(body, gm)
            self.assertGreater(gm[body], 0)

    def test_sg_patch_astropy_noop_when_missing(self):
        """sg_patch_astropy should raise ImportError if astropy absent."""
        try:
            import astropy  # noqa: F401
            patched = sg_patch_astropy()
            # If astropy is present, verify G was patched
            import astropy.constants as aconst
            # The patched G value should be close to SSBM G
            if hasattr(aconst.G, 'value'):
                self.assertAlmostEqual(aconst.G.value, SGConstants.G,
                                       delta=abs(SGConstants.G) * 1e-4)
        except ImportError:
            with self.assertRaises(ImportError):
                sg_patch_astropy()

    def test_sg_patch_scipy_noop_when_missing(self):
        """sg_patch_scipy should raise ImportError if scipy absent."""
        try:
            import scipy  # noqa: F401
            patched = sg_patch_scipy()
            import scipy.constants as sc
            self.assertAlmostEqual(sc.G, SGConstants.G, delta=abs(SGConstants.G)*1e-6)
        except ImportError:
            with self.assertRaises(ImportError):
                sg_patch_scipy()


# ══════════════════════════════════════════════════════════════════════════
# SGPhysicsMixin
# ══════════════════════════════════════════════════════════════════════════

class TestSGPhysicsMixin(unittest.TestCase):

    class _Dummy(SGPhysicsMixin):
        pass

    def setUp(self):
        self.obj = self._Dummy()

    def test_sg_gm_sun(self):
        gm = self.obj.sg_gm("Sun")
        self.assertAlmostEqual(gm, SGConstants.GM_sun, delta=1e6)

    def test_sg_gm_earth(self):
        gm = self.obj.sg_gm("Earth")
        self.assertGreater(gm, 3.9e14)
        self.assertLess(gm, 4.0e14)

    def test_sg_mass_sun(self):
        m = self.obj.sg_mass("Sun")
        self.assertGreater(m, 1.98e30)
        self.assertLess(m, 2.0e30)

    def test_sg_gm_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.obj.sg_gm("Pluto_X_9999")

    # -- sg_density (fix queue C4: every call raised ImportError) ----------

    def test_sg_density_room_temperature(self):
        """CRC Handbook room-temperature densities [LIT]: Fe 7874, Cu 8960."""
        self.assertAlmostEqual(self.obj.sg_density("Fe"), 7874.0, delta=0.005 * 7874)
        self.assertAlmostEqual(self.obj.sg_density("copper"), 8960.0, delta=0.005 * 8960)

    def test_sg_density_temperature_and_pressure(self):
        """Heating 293 -> 373 K lowers a metal's density by 3 alpha dT with
        alpha in 1e-5..2e-5 /K [LIT], i.e. 0.24-0.48 %; 1 GPa raises iron's by
        dP/K with K = 160-175 GPa [LIT], i.e. 0.57-0.63 %."""
        rho0 = self.obj.sg_density("iron")
        drop = 1.0 - self.obj.sg_density("iron", T_K=373.15) / rho0
        self.assertTrue(0.0024 < drop < 0.0048, drop)
        rise = self.obj.sg_density("iron", P_Pa=1e9 + 101_325.0) / rho0 - 1.0
        self.assertTrue(0.0057 < rise < 0.0063, rise)

    def test_sg_density_refuses_melted_ice_and_unknowns(self):
        """Ice above 273.15 K at 1 atm is not ice: no silent number."""
        with self.assertRaises(ValueError):
            self.obj.sg_density("water_ice", T_K=300.0)
        with self.assertRaises(KeyError):
            self.obj.sg_density("unobtainium")

    def test_sg_propagate_orbit_rejects_unknown_method(self):
        """Before: any string but 'forest_ruth' silently ran Verlet."""
        from sigma_ground.field.interface.nbody import CelestialBody
        bodies = [CelestialBody(1.989e30, np.zeros(3), np.zeros(3), 7e8, 0.5)]
        with self.assertRaises(ValueError):
            self.obj.sg_propagate_orbit(bodies, dt_s=3600.0, n_steps=1, method="rk4")

    def test_sg_propagate_orbit(self):
        """sg_propagate_orbit should return bodies list of same length."""
        from sigma_ground.field.interface.nbody import CelestialBody
        M = 1.989e30
        a = 1.496e11
        v = math.sqrt(_G * 2 * M / a)
        bodies = [
            CelestialBody(M, np.array([a/2, 0, 0.0]), np.array([0, v/2, 0.0]), 7e8, 0.5),
            CelestialBody(M, np.array([-a/2, 0, 0.0]), np.array([0, -v/2, 0.0]), 7e8, 0.5),
        ]
        result = self.obj.sg_propagate_orbit(bodies, dt_s=3600.0, n_steps=10)
        self.assertEqual(len(result), 2)


# ══════════════════════════════════════════════════════════════════════════
# SGAdapter
# ══════════════════════════════════════════════════════════════════════════

class TestSGAdapter(unittest.TestCase):

    def test_wraps_arbitrary_object(self):
        class _Ext:
            name = "external"
            def compute(self): return 42

        wrapped = SGAdapter(_Ext())
        self.assertEqual(wrapped.name, "external")
        self.assertEqual(wrapped.compute(), 42)

    def test_sg_methods_available(self):
        wrapped = SGAdapter(object())
        gm = wrapped.sg_gm("Sun")
        self.assertAlmostEqual(gm, SGConstants.GM_sun, delta=1e8)

    def test_lib_attribute(self):
        obj     = [1, 2, 3]
        wrapped = SGAdapter(obj)
        self.assertIs(wrapped.lib, obj)

    def test_unknown_sg_method_raises(self):
        wrapped = SGAdapter(object())
        with self.assertRaises(AttributeError):
            _ = wrapped.sg_nonexistent_method_xyz

    def test_copy_and_pickle(self):
        """Before: RecursionError (__getattr__ looked up self.lib on an
        instance rebuilt without __init__)."""
        import copy
        import pickle
        wrapped = SGAdapter([1, 2, 3])
        self.assertEqual(copy.copy(wrapped).lib, [1, 2, 3])
        self.assertEqual(copy.deepcopy(wrapped).lib, [1, 2, 3])
        self.assertEqual(pickle.loads(pickle.dumps(wrapped)).lib, [1, 2, 3])


# ══════════════════════════════════════════════════════════════════════════
# SGEphemeris
# ══════════════════════════════════════════════════════════════════════════

class TestSGEphemeris(unittest.TestCase):

    def setUp(self):
        self.eph = SGEphemeris()

    def test_available_bodies_includes_major_planets(self):
        bodies = self.eph.sg_available_bodies()
        for name in ("Sun", "Earth", "Mars", "Jupiter"):
            self.assertIn(name, bodies, f"{name} missing from available_bodies")

    def test_de440_tier_past_epoch(self):
        """J2025.0 (JD 2460676.25) is a DE440 snapshot -- tier should be de440.

        (Until 2026-10-10 this used JD 2460310.5, half a day before the
        J2024.0 snapshot, which the tier served only because it snapped to
        snapshots up to 5 days away.)"""
        jd_j2025 = 2460676.25
        tier = self.eph.sg_tier("Earth", jd_j2025)
        self.assertEqual(tier, "de440",
                         f"Expected de440 tier for J2025, got {tier!r}")

    def test_forecast_tier_future_epoch(self):
        """A stored forecast epoch (month 180, ~J2040) -- tier should be forecast."""
        jd_m180 = self.eph._load_forecast()["predictions"][180]["jd"]
        tier = self.eph.sg_tier("Earth", jd_m180)
        self.assertEqual(tier, "forecast",
                         f"Expected forecast at a stored epoch, got {tier!r}")

    def test_position_shape(self):
        """sg_position should return (3,) ndarray."""
        pos = self.eph.sg_position("Earth", 2460310.5)
        self.assertEqual(pos.shape, (3,))
        self.assertTrue(np.all(np.isfinite(pos)))

    def test_heliocentric_earth_is_1au(self):
        """Earth heliocentric distance at J2025 should be ~1 AU."""
        jd    = 2460310.5
        dist  = self.eph.sg_distance_au("Earth", jd)
        self.assertAlmostEqual(dist, 1.0, delta=0.05,
                               msg=f"Earth at {dist:.3f} AU (expected ~1 AU)")

    def test_position_with_uncertainty_past(self):
        """At a DE440 snapshot epoch the served value is exact: error 0."""
        jd   = 2460676.25  # J2025.0 -- in DE440
        pos, err_au = self.eph.sg_position_with_uncertainty("Earth", jd)
        self.assertLess(err_au, 0.05,
                        f"SSBM vs Kepler disagreement at J2025: {err_au:.4f} AU")

    def test_heliocentric_sun_is_zero(self):
        """Sun's heliocentric position should be (0,0,0) by definition."""
        hel = self.eph.sg_heliocentric("Sun", 2460310.5)
        dist_km = float(np.linalg.norm(hel))
        self.assertLess(dist_km, 1e6,  # within 1M km of origin
                        f"Sun heliocentric offset {dist_km:.0f} km")


class TestSGEphemerisContract(unittest.TestCase):
    """Regression tests for the 2026-10-10 audit (fix queue C2, A44-A46).

    Ground truth: the DE440 fixture snapshots, and the de440s kernel read
    directly with jplephem (not through query_jplephem); kernel tests skip
    when the kernel is not installed.
    """

    JD_J2010 = 2455197.5           # DE440 snapshot epoch
    JD_2012 = 2456110.5            # 2012-07-02, between snapshots
    _CHAINS = {"Sun": [(0, 10)], "Earth": [(0, 3), (3, 399)],
               "Moon": [(0, 3), (3, 301)], "Mars": [(0, 4)],
               "Jupiter": [(0, 5)]}

    @classmethod
    def setUpClass(cls):
        import json
        from pathlib import Path
        fx = Path(__file__).parent.parent / "fixtures"
        with open(fx / "de440_state_vectors.json") as f:
            cls.fix = json.load(f)
        with open(fx / "ssbm_future_predictions.json") as f:
            cls.fc = json.load(f)
        cls.kernel = None
        try:
            from jplephem.spk import SPK
            from sigma_ground.field.interface.adapters._jplephem_bridge import (
                _KERNEL_DIR,
            )
            path = _KERNEL_DIR / "de440s.bsp"
            if path.exists():
                cls.kernel = SPK.open(str(path))
        except ImportError:
            pass

    def _snap(self, key, name):
        for b in self.fix["snapshots"][key]["bodies"]:
            if b["name"] == name:
                sv = b["state_vector"]
                return np.array([sv["x_km"], sv["y_km"], sv["z_km"]])
        raise KeyError(name)

    def _truth(self, name, jd):
        if self.kernel is None:
            self.skipTest("de440s kernel not installed")
        return sum(np.asarray(self.kernel[a, b].compute(jd))
                   for a, b in self._CHAINS[name])

    # -- A44: the Kepler tier evaluates anchor and Sun at the query epoch --

    def test_kepler_tier_anchor_and_sun_at_query_epoch(self):
        """Before: the anchor was the J2026 snapshot for every epoch (Earth
        5e5 km off, Titan 2e9 km off at J2010; the Moon 2 AU off)."""
        eph = SGEphemeris(use_kernel=False)
        for name, tol in (("Earth", 5e4), ("Mars", 5e4), ("Sun", 2e3)):
            err = np.linalg.norm(eph._kepler_position(name, self.JD_J2010)
                                 - self._snap("j2010", name))
            self.assertLess(err, tol, f"{name}: {err:.4g} km")
        rel = (eph._kepler_position("Titan", self.JD_J2010)
               - eph._kepler_position("Saturn", self.JD_J2010))
        rel_true = self._snap("j2010", "Titan") - self._snap("j2010", "Saturn")
        self.assertLess(np.linalg.norm(rel - rel_true), 1e4)

    def test_kepler_tier_off_snapshot_vs_kernel(self):
        """Earth on 2012-07-02 from the Kepler tier: before 5.4e5 km off."""
        truth_e = self._truth("Earth", self.JD_2012)
        truth_s = self._truth("Sun", self.JD_2012)
        eph = SGEphemeris(use_kernel=False)
        self.assertEqual(eph.sg_tier("Earth", self.JD_2012), "kepler")
        self.assertLess(np.linalg.norm(eph.sg_position("Earth", self.JD_2012)
                                       - truth_e), 2e4)
        self.assertLess(np.linalg.norm(eph.sg_heliocentric("Earth", self.JD_2012)
                                       - (truth_e - truth_s)), 2e4)
        self.assertLess(np.linalg.norm(eph.sg_position("Sun", self.JD_2012)
                                       - truth_s), 2e3)

    # -- A45: the forecast tier serves only its stored epochs --------------

    def test_forecast_served_only_at_stored_epochs(self):
        """Before: month 0.5 was served 4e7 km off; so were epochs 15 d
        outside the forecast."""
        eph = SGEphemeris(use_kernel=False)
        meta = self.fc["metadata"]
        start, md = meta["start_jd"], meta["month_days"]
        last = self.fc["predictions"][-1]["jd"]
        for jd in (start + 0.5 * md, start + 119.5 * md, start - 15.0,
                   last + 15.0):
            self.assertIsNone(eph._forecast_position("Earth", jd), jd)
            self.assertEqual(eph.sg_tier("Earth", jd), "kepler", jd)
        stored = self.fc["predictions"][120]["jd"]
        self.assertEqual(eph.sg_tier("Earth", stored), "forecast")

    def test_forecast_heliocentric_is_the_stored_vector(self):
        """Before: a Sun from another date was subtracted (1.47e6 km)."""
        entry = self.fc["predictions"][240]
        hel = SGEphemeris().sg_heliocentric("Jupiter", entry["jd"])
        np.testing.assert_allclose(hel, entry["ssbm"]["Jupiter"], atol=1e-6)

    def test_forecast_sun_from_its_own_planets(self):
        """The forecast tier's Sun (barycentre of the forecast's planets)
        vs DE440 at month 240: before 1.47e6 km (J2025 Sun frozen)."""
        entry = self.fc["predictions"][240]
        jd = entry["jd"]
        eph = SGEphemeris()
        sun = eph._forecast_position("Mercury", jd) - np.array(entry["ssbm"]["Mercury"])
        self.assertLess(np.linalg.norm(sun - self._truth("Sun", jd)), 5e3)

    # -- A46: the uncertainty belongs to the tier that served -------------

    def test_uncertainty_zero_for_exact_values(self):
        """Before: 0.0085 AU for the exact DE440 Earth, nan for the Sun."""
        for name in ("Earth", "Moon", "Sun"):
            _, err = SGEphemeris().sg_position_with_uncertainty(name, self.JD_J2010)
            self.assertEqual(err, 0.0, name)

    def test_uncertainty_kepler_tier_is_not_zero(self):
        """Before: exactly 0 for the Moon on 2012-07-02, then 2.0 AU off."""
        _, err = SGEphemeris(use_kernel=False).sg_position_with_uncertainty(
            "Moon", self.JD_2012)
        self.assertTrue(math.isnan(err), err)

    def test_uncertainty_forecast_tier_is_measured(self):
        jd = self.fc["predictions"][120]["jd"]
        truth = self._truth("Earth", jd)
        pos, err = SGEphemeris().sg_position_with_uncertainty("Earth", jd)
        self.assertAlmostEqual(err, np.linalg.norm(pos - truth) / 1.495978707e8,
                               delta=1e-12)
        self.assertGreater(err, 0.0)

    # -- query_jplephem Earth and Moon; tier order; names ------------------

    def test_query_jplephem_earth_and_moon(self):
        """Before: None for both (kernel[0, 399] does not exist)."""
        from sigma_ground.field.interface.adapters._jplephem_bridge import (
            query_jplephem,
        )
        self._truth("Sun", self.JD_J2010)  # skip without kernel
        earth = query_jplephem("Earth", self.JD_J2010)
        moon = query_jplephem("moon", self.JD_J2010)
        self.assertIsNotNone(earth)
        self.assertIsNotNone(moon)
        self.assertLess(np.linalg.norm(earth - self._snap("j2010", "Earth")), 1.0)
        # Earth-Moon distance between lunar perigee and apogee extremes
        # (356 375 - 406 720 km; Meeus, Astronomical Algorithms, ch. 50)
        self.assertTrue(3.56e5 < np.linalg.norm(moon - earth) < 4.07e5)

    def test_naif_codes_kept_for_long_horizon_sweep(self):
        """scripts/run_long_horizon_sweep.py imports _NAIF_CODES; C2 had
        removed it (ImportError at import).  Values: NAIF integer IDs (NAIF
        "NAIF IDs" required reading): barycentres 1-9, Sun 10, Earth 399,
        Moon 301."""
        from sigma_ground.field.interface.adapters._jplephem_bridge import (
            _NAIF_CODES, _load_kernel,  # noqa: F401 -- the script's import
        )
        self.assertEqual(_NAIF_CODES, {
            "Sun": 10, "Mercury": 1, "Venus": 2, "Earth": 399, "Moon": 301,
            "Mars": 4, "Jupiter": 5, "Saturn": 6, "Uranus": 7, "Neptune": 8,
            "Pluto": 9})

    def test_de440_tier_only_at_snapshot_epochs(self):
        """Before: J2010 + 1 d returned the J2010 snapshot (2.6e6 km off)."""
        eph = SGEphemeris(use_kernel=False)
        self.assertNotEqual(eph.sg_tier("Earth", self.JD_J2010 + 1.0), "de440")
        self.assertEqual(eph.sg_tier("Earth", self.JD_J2010), "de440")
        truth = self._truth("Earth", self.JD_J2010 + 1.0)
        pos = SGEphemeris().sg_position("Earth", self.JD_J2010 + 1.0)
        self.assertLess(np.linalg.norm(pos - truth), 1.0)

    def test_kernel_ranks_above_kepler(self):
        """Before: Mars on 2012-07-02 came from Kepler (5.3e5 km off)."""
        truth = self._truth("Mars", self.JD_2012)
        eph = SGEphemeris()
        self.assertEqual(eph.sg_tier("Mars", self.JD_2012), "jplephem")
        self.assertLess(np.linalg.norm(eph.sg_position("Mars", self.JD_2012) - truth), 1.0)

    # -- C2 review: kernel tier planet centres and satellites on them -----

    # JPL SSD planetary satellite mean elements, https://ssd.jpl.nasa.gov/
    # sats/elem/ (checked 2026-10-10): a (km), e; MAR099, NEP097, PLU060,
    # JUP365, SAT441.
    _MEAN_ELEMENTS = {"Charon": ("Pluto", 19600.0, 0.000),
                      "Triton": ("Neptune", 354800.0, 0.000),
                      "Phobos": ("Mars", 9375.0, 0.015),
                      "Io": ("Jupiter", 421800.0, 0.004),
                      "Titan": ("Saturn", 1221900.0, 0.029)}

    def test_kernel_on_satellite_sits_on_its_planet(self):
        """Kernel on, 2012-07-02: |moon - planet| must lie in a(1-e)..a(1+e)
        (1 % allowed for mean vs osculating elements and the table's
        rounding).  Before: the planet came from the kernel and the moon
        from a Kepler planet: Charon-Pluto 1.78e6 km, Triton-Neptune
        1.90e6, Phobos-Mars 3.34e4, Io-Jupiter 5.45e5, Titan-Saturn 1.57e6."""
        self._truth("Sun", self.JD_2012)  # skip without kernel
        eph = SGEphemeris()
        for moon, (planet, a, e) in self._MEAN_ELEMENTS.items():
            self.assertEqual(eph.sg_tier(planet, self.JD_2012), "jplephem")
            for fn in (eph.sg_position, eph.sg_heliocentric):
                sep = float(np.linalg.norm(fn(moon, self.JD_2012)
                                           - fn(planet, self.JD_2012)))
                self.assertTrue(0.99 * a * (1 - e) <= sep <= 1.01 * a * (1 + e),
                                f"{moon}-{planet} {fn.__name__}: {sep:.6g} km")

    def test_kernel_tier_gives_planet_centres(self):
        """de440s holds system barycentres for Mars..Pluto; the kernel tier
        must give the planet centre.  Truth: the fixture's Horizons planet
        centre at J2010, and Horizons' own barycentre->centre offsets at
        JD 2457024.0 (0.25 d off the J2015 snapshot, so also between the
        moon fits' samples).  Before: Pluto 2131 km off at J2010; offsets
        off by Jupiter 180, Saturn 275, Neptune 74, Pluto 2131 km."""
        from sigma_ground.field.interface.adapters._jplephem_bridge import (
            query_jplephem,
        )
        self._truth("Sun", self.JD_J2010)  # skip without kernel
        eph = SGEphemeris()
        pluto = eph._jplephem_position("Pluto", self.JD_J2010)
        self.assertLess(np.linalg.norm(pluto - self._snap("j2010", "Pluto")), 1.0)
        offsets = self.fix["bary_to_center_offsets"]
        jd = offsets["epochs"]["j2015"]["jd_tdb"]
        for planet, tol in (("Jupiter", 1.0), ("Saturn", 20.0),
                            ("Neptune", 1.0), ("Pluto", 1.0)):
            rec = offsets["bodies"][planet]["j2015"]
            truth = query_jplephem(planet, jd) + np.array(
                [rec["dx_km"], rec["dy_km"], rec["dz_km"]])
            err = float(np.linalg.norm(eph._jplephem_position(planet, jd) - truth))
            self.assertLess(err, tol, f"{planet}: {err:.4g} km")

    def test_kernel_tier_uncertainty_covers_centre_error(self):
        """Before: exactly 0.0 for the kernel's system barycentres (Pluto
        2131 km, Uranus 507 km from the planet centre at J2010)."""
        self._truth("Sun", self.JD_J2010)  # skip without kernel
        eph = SGEphemeris()
        for planet in ("Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"):
            self.assertEqual(eph.sg_tier(planet, self.JD_2012), "jplephem")
            _, err_au = eph.sg_position_with_uncertainty(planet, self.JD_2012)
            miss = float(np.linalg.norm(eph._jplephem_position(planet, self.JD_J2010)
                                        - self._snap("j2010", planet)))
            self.assertGreaterEqual(err_au * 1.495978707e8, miss, planet)
            if planet in ("Saturn", "Uranus", "Pluto"):
                self.assertGreater(err_au, 0.0, planet)
        for name in ("Earth", "Moon", "Sun"):
            _, err_au = eph.sg_position_with_uncertainty(name, self.JD_2012)
            self.assertEqual(err_au, 0.0, name)

    def test_available_bodies_are_servable_off_snapshot(self):
        eph = SGEphemeris(use_kernel=False)
        bodies = eph.sg_available_bodies()
        self.assertNotIn("Mimas", bodies)
        self.assertIn("Titan", bodies)
        self.assertEqual(eph.sg_position("Titan", self.JD_2012).shape, (3,))
        np.testing.assert_array_equal(eph.sg_position("mars", self.JD_J2010),
                                      self._snap("j2010", "Mars"))


# ══════════════════════════════════════════════════════════════════════════
# sg_solve_ivp (Forest-Ruth)
# ══════════════════════════════════════════════════════════════════════════

class TestSGSolveIVP(unittest.TestCase):
    """Harmonic oscillator: H = p²/2 + q²/2  (exact analytical solution)."""

    @staticmethod
    def _sho(t, y):
        """Harmonic oscillator: dq/dt = p, dp/dt = -q."""
        return np.array([y[1], -y[0]])

    def test_solution_shape(self):
        sol = sg_solve_ivp(self._sho, (0, 2 * math.pi), np.array([1.0, 0.0]),
                           method="forest_ruth", dt=0.01)
        self.assertEqual(sol.y.shape[0], 2)
        self.assertGreater(sol.y.shape[1], 1)

    def test_period_correct(self):
        """After one period (T=2pi), q should return to 1.0."""
        T   = 2 * math.pi
        sol = sg_solve_ivp(self._sho, (0, T), np.array([1.0, 0.0]),
                           method="forest_ruth", dt=0.01)
        q_final = sol.y[0, -1]
        self.assertAlmostEqual(q_final, 1.0, delta=1e-3)

    def test_energy_conserved_fr4(self):
        """FR4 should conserve H = (q²+p²)/2 to < 1e-6 over 100 periods."""
        T   = 2 * math.pi
        sol = sg_solve_ivp(self._sho, (0, 100 * T), np.array([1.0, 0.0]),
                           method="forest_ruth", dt=0.05)
        H0 = 0.5 * (sol.y[0, 0]**2 + sol.y[1, 0]**2)
        Hf = 0.5 * (sol.y[0, -1]**2 + sol.y[1, -1]**2)
        drift = abs(Hf - H0) / abs(H0)
        self.assertLess(drift, 1e-6, f"FR4 energy drift {drift:.2e} over 100 periods")

    def test_fr4_better_than_rk4_energy(self):
        """FR4 should conserve energy better than RK4 over 100 periods."""
        T  = 2 * math.pi
        y0 = np.array([1.0, 0.0])

        def H(y): return 0.5 * (y[0]**2 + y[1]**2)

        sol_fr4 = sg_solve_ivp(self._sho, (0, 100*T), y0,
                               method="forest_ruth", dt=0.05)
        sol_rk4 = sg_solve_ivp(self._sho, (0, 100*T), y0,
                               method="rk4", dt=0.05)

        drift_fr4 = abs(H(sol_fr4.y[:, -1]) - H(y0)) / H(y0)
        drift_rk4 = abs(H(sol_rk4.y[:, -1]) - H(y0)) / H(y0)
        self.assertLess(drift_fr4, drift_rk4,
                        f"FR4 drift {drift_fr4:.2e} not < RK4 {drift_rk4:.2e}")

    def test_verlet_api(self):
        """Verlet method should also run without error."""
        sol = sg_solve_ivp(self._sho, (0, math.pi), np.array([1.0, 0.0]),
                           method="verlet", dt=0.01)
        self.assertTrue(sol.success)


class TestSGSolveIVPContract(unittest.TestCase):
    """Regression tests for the 2026-10-10 audit (fix queue C1).

    Ground truth is closed-form: the driven oscillator q'' = -q + cos 2t with
    q(0) = q'(0) = 0 has q(t) = (cos 2t - cos t) / (1 - 4); the free oscillator
    has q = cos t; y' = y^2, y(0) = 1 blows up at t = 1 (y = 1/(1 - t)).
    """

    @staticmethod
    def _sho(t, y):
        return np.array([y[1], -y[0]])

    def test_fr4_fourth_order_on_driven_oscillator(self):
        """Forest-Ruth substeps must advance time: order 4 on a driven system.

        Before the fix every drift and kick saw the step-start time and the
        error fell only 2x per halving (order 0.98)."""
        def driven(t, y):
            return np.array([y[1], -y[0] + math.cos(2.0 * t)])

        def exact(t):
            return (math.cos(2.0 * t) - math.cos(t)) / (1.0 - 4.0)

        T = 20.0
        errs = []
        for n in (400, 800, 1600):
            sol = sg_solve_ivp(driven, (0.0, T), np.array([0.0, 0.0]),
                               method="forest_ruth", dt=T / n)
            errs.append(abs(sol.y[0, -1] - exact(T)))
        for e1, e2 in zip(errs, errs[1:]):
            self.assertGreater(math.log2(e1 / e2), 3.8,
                               f"FR4 order on driven oscillator: errs {errs}")
        self.assertLess(errs[-1], 1e-7)

    def test_odd_length_state_rejected_by_symplectic_methods(self):
        """A [q, p] splitting cannot integrate an odd-length state."""
        for method in ("forest_ruth", "verlet"):
            with self.assertRaises(ValueError):
                sg_solve_ivp(lambda t, y: -y, (0.0, 5.0), np.array([1.0]),
                             method=method)
        sol = sg_solve_ivp(lambda t, y: -y, (0.0, 5.0), np.array([1.0]),
                           method="rk4", dt=5.0 / 400)
        self.assertAlmostEqual(sol.y[0, -1], math.exp(-5.0), delta=1e-8)

    def test_non_separable_split_warns(self):
        """Damping makes dp/dt depend on p: the symplectic default must say so."""
        def damped(t, y):
            return np.array([y[1], -y[0] - 0.2 * y[1]])

        with self.assertWarns(RuntimeWarning):
            sg_solve_ivp(damped, (0.0, 1.0), np.array([1.0, 0.0]), dt=0.01)
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error", RuntimeWarning)
            sg_solve_ivp(self._sho, (0.0, 1.0), np.array([1.0, 0.0]), dt=0.01)

    def test_list_returning_rhs(self):
        """A right-hand side returning a list (the scipy idiom) is accepted."""
        for method in ("forest_ruth", "verlet", "rk4"):
            sol = sg_solve_ivp(lambda t, y: [y[1], -y[0]], (0.0, 1.0),
                               np.array([1.0, 0.0]), method=method, dt=1e-3)
            self.assertAlmostEqual(sol.y[0, -1], math.cos(1.0), delta=1e-6)
            self.assertAlmostEqual(sol.y[1, -1], -math.sin(1.0), delta=1e-6)

    def test_include_gr_true_is_rejected(self):
        """include_gr=True used to be accepted and ignored."""
        with self.assertRaises(ValueError):
            sg_solve_ivp(self._sho, (0.0, 1.0), np.array([1.0, 0.0]),
                         dt=0.01, include_gr=True)

    def test_blow_up_reports_failure(self):
        """y' = y^2 from y(0) = 1 diverges at t = 1: success must be False."""
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            sol = sg_solve_ivp(lambda t, y: y * y, (0.0, 2.0), np.array([1.0]),
                               method="rk4", dt=0.01)
        self.assertFalse(sol.success)
        self.assertEqual(sol.status, -1)
        self.assertTrue(np.all(np.isfinite(sol.y)))
        self.assertLess(sol.t[-1], 1.05)

    def test_t_eval_returned_exactly(self):
        """sol.t equals t_eval, off-grid times included, values = cos t."""
        t_eval = np.array([1.0, 1.004, 2.0, 7.4])
        sol = sg_solve_ivp(self._sho, (0.0, 10.0), np.array([1.0, 0.0]),
                           dt=0.01, t_eval=t_eval)
        np.testing.assert_array_equal(sol.t, t_eval)
        np.testing.assert_allclose(sol.y[0], np.cos(t_eval), atol=1e-7)

    def test_t_eval_out_of_span_or_unsorted_raises(self):
        """As scipy: out-of-span or unsorted t_eval is an error, not a guess."""
        for bad in ([-1.0, 5.0, 12.0], [5.0, 1.0, 3.0]):
            with self.assertRaises(ValueError):
                sg_solve_ivp(self._sho, (0.0, 10.0), np.array([1.0, 0.0]),
                             dt=0.01, t_eval=np.array(bad))

    def test_backward_integration(self):
        """t_span = (10, 0) integrates backward to the initial condition."""
        y10 = np.array([math.cos(10.0), -math.sin(10.0)])
        sol = sg_solve_ivp(self._sho, (10.0, 0.0), y10, dt=0.01)
        self.assertTrue(sol.success)
        self.assertAlmostEqual(sol.t[-1], 0.0, delta=1e-9)
        np.testing.assert_allclose(sol.y[:, -1], [1.0, 0.0], atol=1e-7)

    def test_zero_span_returns_initial_state(self):
        sol = sg_solve_ivp(self._sho, (0.0, 0.0), np.array([1.0, 0.0]))
        self.assertTrue(sol.success)
        np.testing.assert_array_equal(sol.y[:, 0], [1.0, 0.0])

    def test_bad_dt_raises(self):
        for bad in (0.0, -0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                sg_solve_ivp(self._sho, (0.0, 1.0), np.array([1.0, 0.0]),
                             dt=bad)


# ══════════════════════════════════════════════════════════════════════════
# sg_nbody
# ══════════════════════════════════════════════════════════════════════════

class TestSGNbody(unittest.TestCase):

    def _circular_pair(self):
        from sigma_ground.field.interface.nbody import CelestialBody
        M = 1.989e30
        a = 1.496e11
        v = math.sqrt(_G * 2 * M / a) / 2
        b1 = CelestialBody(M, np.array([ a/2, 0, 0.0]), np.array([0,  v, 0.0]), 7e8, 0.5)
        b2 = CelestialBody(M, np.array([-a/2, 0, 0.0]), np.array([0, -v, 0.0]), 7e8, 0.5)
        return [b1, b2]

    def test_returns_bodies_list(self):
        bodies = self._circular_pair()
        result, times = sg_nbody(bodies, dt_s=3600.0, t_total_s=86400.0)
        self.assertEqual(len(result), 2)
        self.assertGreater(len(times), 0)

    def test_bodies_moved(self):
        from sigma_ground.field.interface.nbody import NBodySystem
        bodies = self._circular_pair()
        p0 = bodies[0].position_m.copy()
        final, _ = sg_nbody(bodies, dt_s=3600.0, t_total_s=86400.0)
        self.assertFalse(np.allclose(final[0].position_m, p0))

    def test_energy_conserved(self):
        from sigma_ground.field.interface.nbody import NBodySystem
        bodies_in = self._circular_pair()
        sys0 = NBodySystem(bodies_in)
        E0   = sys0.total_energy()

        period = 2 * math.pi * math.sqrt((1.496e11)**3 / (_G * 2 * 1.989e30))
        final, _ = sg_nbody(bodies_in, dt_s=period / 500,
                            t_total_s=2 * period, include_gr=False)

        sys1 = NBodySystem(final)
        E1   = sys1.total_energy()
        drift = abs(E1 - E0) / abs(E0)
        self.assertLess(drift, 0.05, f"sg_nbody energy drift {drift:.2e}")

    # -- fix queue C4 (audit 2026-10-10) ----------------------------------

    def test_ends_at_t_total(self):
        """Before: ceil() whole steps overshot (3 d steps over 10 d ended at
        12 d; 30 d steps over 1 d ended at 30 d)."""
        for dt_d, total_d in ((3.0, 10.0), (30.0, 1.0), (0.7, 7.0)):
            _, times = sg_nbody(self._circular_pair(), dt_s=dt_d * 86400.0,
                                t_total_s=total_d * 86400.0)
            self.assertAlmostEqual(times[-1] / 86400.0, total_d, delta=1e-9)

    def test_circular_orbit_closes_after_one_period(self):
        """Equal-mass circular binary returns to its start after
        T = 2 pi sqrt(a^3 / (G M_tot)) (closed form) even when dt does not
        divide T.  Before: 500 steps of T/499.5 overran by 0.1 % of an orbit
        (|dr| ~ 4.7e-3 of the orbit radius)."""
        M, a = 1.989e30, 1.496e11
        T = 2 * math.pi * math.sqrt(a ** 3 / (_G * 2 * M))
        bodies = self._circular_pair()
        r0 = bodies[0].position_m.copy()
        final, _ = sg_nbody(bodies, dt_s=T / 499.5, t_total_s=T)
        self.assertLess(np.linalg.norm(final[0].position_m - r0) / (a / 2), 1e-5)

    def test_rejects_unknown_method_and_bad_steps(self):
        """Before: 'rk4' silently ran Verlet; record_every=0 and dt_s=0 raised
        ZeroDivisionError; negative dt_s returned the input unchanged."""
        for kw in ({"method": "rk4"}, {"method": "Forest-Ruth"},
                   {"record_every": 0}, {"dt_s": 0.0}, {"dt_s": -3600.0},
                   {"t_total_s": -3600.0}, {"dt_s": float("nan")}):
            args = {"dt_s": 3600.0, "t_total_s": 7200.0}
            args.update(kw)
            with self.assertRaises(ValueError, msg=str(kw)):
                sg_nbody(self._circular_pair(), **args)


# ══════════════════════════════════════════════════════════════════════════
# sg_odeint
# ══════════════════════════════════════════════════════════════════════════

class TestSGOdeint(unittest.TestCase):

    def test_output_shape(self):
        """sg_odeint should return (n_time, n_state).

        A one-component state is not a [q, p] split, so it needs rk4 (the
        symplectic default rejects it since 2026-10-10)."""
        def f(y, t): return np.array([-y[0]])
        t  = np.linspace(0, 1, 50)
        y0 = np.array([1.0])
        result = sg_odeint(f, y0, t, method="rk4")
        self.assertEqual(result.shape, (50, 1))

    # -- 2026-10-10 audit regressions (fix queue C1); closed-form truth ----

    @staticmethod
    def _sho_yt(y, t):
        return np.array([y[1], -y[0]])

    def test_step_independent_of_output_spacing(self):
        """Outputs every 2 s over 100 s must not mean a 2 s step.

        Before the fix the output spacing was the step: FR4 reached 2.6e43."""
        t = np.arange(0.0, 100.0 + 1e-9, 2.0)
        y = sg_odeint(self._sho_yt, np.array([1.0, 0.0]), t, dt=0.01)
        self.assertLess(float(np.max(np.abs(y[:, 0] - np.cos(t)))), 1e-6)
        y_default = sg_odeint(self._sho_yt, np.array([1.0, 0.0]), t)
        self.assertLess(float(np.max(np.abs(y_default[:, 0] - np.cos(t)))), 1e-3)

    def test_dense_grid_default_step(self):
        """A dense output grid without dt must not be stepped more coarsely
        than the grid (closed form q = cos t).

        Before the repair the default step was always span/1000:
        linspace(0, 300, 30001) gave 0.16 and linspace(0, 100, 10001)
        6.6e-4 (stepping at the grid spacing gives 2.0e-7 and 6.5e-8)."""
        y0 = np.array([1.0, 0.0])
        for t, tol in ((np.linspace(0.0, 300.0, 30001), 1e-5),
                       (np.linspace(0.0, 100.0, 10001), 1e-6)):
            y = sg_odeint(self._sho_yt, y0, t)
            err = float(np.max(np.abs(y[:, 0] - np.cos(t))))
            self.assertLess(err, tol, f"{t.size} points over {t[-1]:g} s")

    def test_non_uniform_grid(self):
        """Row k is y(t[k]) on a non-uniform grid."""
        t = np.array([0.0, 0.3, 0.35, 1.234, 5.0])
        y = sg_odeint(self._sho_yt, np.array([1.0, 0.0]), t, dt=1e-3)
        self.assertEqual(y.shape, (5, 2))
        np.testing.assert_allclose(y[:, 0], np.cos(t), atol=1e-9)

    def test_decreasing_grid(self):
        """Decreasing t integrates backward, as odeint does."""
        t = np.linspace(2.0, 0.0, 11)
        y0 = np.array([math.cos(2.0), -math.sin(2.0)])
        y = sg_odeint(self._sho_yt, y0, t, dt=1e-3)
        np.testing.assert_allclose(y[:, 0], np.cos(t), atol=1e-9)

    def test_argument_order(self):
        """f(time, y) is not swapped; tfirst selects the convention."""
        def f_time_y(time, y):
            return -y

        def f_y_t(y, t):
            return -y

        t = np.linspace(0.0, 2.0, 201)
        for func, kw in ((f_time_y, {}), (f_time_y, {"tfirst": True}),
                         (f_y_t, {}), (f_y_t, {"tfirst": False})):
            y = sg_odeint(func, np.array([1.0]), t, method="rk4", **kw)
            self.assertAlmostEqual(y[-1, 0], math.exp(-2.0), delta=1e-9,
                                   msg=f"{func.__name__} {kw}")

    def test_args_and_list_rhs(self):
        y = sg_odeint(lambda y, t, k: [-k * y[0]], np.array([1.0]),
                      np.linspace(0.0, 1.0, 11), method="rk4", args=(2.0,))
        self.assertAlmostEqual(y[-1, 0], math.exp(-2.0), delta=1e-9)

    def test_exponential_decay(self):
        """dy/dt = -y  =>  y(t) = exp(-t)."""
        def f(y, t): return np.array([-y[0]])
        t   = np.linspace(0, 2, 100)
        y0  = np.array([1.0])
        sol = sg_odeint(f, y0, t, method="rk4")
        y_final   = sol[-1, 0]
        y_expected = math.exp(-2)
        self.assertAlmostEqual(y_final, y_expected, delta=0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
