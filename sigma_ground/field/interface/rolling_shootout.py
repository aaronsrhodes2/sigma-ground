"""Rolling-window n-body prediction shootout.

EXPERIMENTAL DESIGN (Aaron Rhodes, 2026-05)
-------------------------------------------
The classical shootout runs ONE integration from a fixed start (J2025) and
compares to DE440 once per body. This module generalizes it:

  * Pick many WINDOW START times along the DE440 grid (J2000-J2026).
  * For each window start, run EVERY predictor forward to the prediction
    horizon, sampling at intermediate times.
  * Compare every (predictor, body, window, sample_time) cell to DE440.
  * The 4-D error tensor reveals how each predictor's failure mode evolves
    with both prediction-time and window-start time.

The window-to-window SUB-DELTA — how a given (predictor, body) error curve
shifts between adjacent windows — is the experimental novelty. It carries
a fingerprint of which physics is responsible: J2 missing → shift correlates
with inclination; tidal dissipation missing → with eccentricity; GR missing
→ with proximity to massive bodies; SRP missing → with cross-sectional area.

By comparing sub-deltas across predictors with DELIBERATELY DIFFERENT
physics (e.g. over_physics vs over_physics_no_j2), we can attribute residuals
to specific missing physics rather than just measuring their sum.

PREDICTOR ABLATION
------------------
  pure_newton          Newton only (no GR, no SRP, no J2)
  standard             Newton + 1PN GR (no SRP, no J2)
  over_physics_no_j2   Newton + GR + SRP (J2 = 0)        ← ablation
  over_physics         Newton + GR + SRP + J2
  kepler               2-body Keplerian fit per body (in-sample: fitted on
                       all snapshots, including the ones it is scored on)
  kepler_forecast      the same fit on snapshots up to the window start only

The over_physics vs over_physics_no_j2 difference isolates J2's contribution.
The standard vs pure_newton difference isolates 1PN GR's contribution.

USAGE
-----
    from sigma_ground.field.interface.rolling_shootout import (
        run_rolling_shootout, PREDICTORS,
    )

    results = run_rolling_shootout(
        window_start_keys=["j2005", "j2006", ..., "j2020"],  # 16 windows
        prediction_horizon_yr=5.0,
        sample_interval_yr=1.0,
        predictors=PREDICTORS,                                # all five
    )

    # results: list[WindowResult], one per (predictor, window, sample)

Output is also saved as JSON (output_path; by default the untracked
local-cache/rolling_shootout/rolling_shootout_results.json) so the analysis
pass doesn't need to re-run the integrations. The committed
fixtures/rolling_shootout_results.json is written only when passed
explicitly as output_path (or --output).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from sigma_ground.field.constants import G as _G, L_SUN_W as _L_SUN_W
from sigma_ground.field.interface.nbody import (
    CelestialBody, NBodySystem, PhysicsToggles,
)
from sigma_ground.field.interface.orbital import fit_orbit, _ecliptic_to_equatorial


def _pole_unit_from_radec(ra_deg: float, dec_deg: float) -> np.ndarray:
    """Convert IAU 2015 pole RA/Dec (degrees) to an ICRS unit vector.

    Standard astrometric convention:
        x = cos(Dec) * cos(RA)
        y = cos(Dec) * sin(RA)
        z = sin(Dec)
    """
    ra  = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    return np.array([
        math.cos(dec) * math.cos(ra),
        math.cos(dec) * math.sin(ra),
        math.sin(dec),
    ])


AU_KM   = 1.495978707e8
KM_TO_M = 1000.0
DAY_S   = 86400.0
YR_DAYS = 365.25

_FIXTURES = Path(__file__).parent / "fixtures"
_DE440_FILE = _FIXTURES / "de440_state_vectors.json"
# The committed (git-tracked) results fixture: read by rolling_analysis,
# written only when a caller passes it explicitly as output_path.
_RESULTS_FILE = _FIXTURES / "rolling_shootout_results.json"
# Default output of run_rolling_shootout: untracked (local-cache/ is ignored).
_DEFAULT_RESULTS_OUTPUT = (Path(__file__).resolve().parents[3] / "local-cache"
                           / "rolling_shootout" / "rolling_shootout_results.json")
_USE_DEFAULT_OUTPUT = object()


# ── IAU rotational poles (WGCCRE) ─────────────────────────────────────────
# Zonal harmonics (J2/J3/J4) act about each body's own spin axis, so every
# body that carries one needs its pole: ICRS +z is off by 25.5 deg for
# Jupiter, 37 deg for Mars, 28.6 deg for Mercury and 6.5 deg for Saturn.
# Pole right ascension / declination in the ICRF, as published by the IAU
# Working Group on Cartographic Coordinates and Rotational Elements:
#   Archinal et al. 2018, Celest. Mech. Dyn. Astr. 130:22 (the 2015 report)
#   for the Sun, planets, Pluto and satellites; Earth and Moon from
#   Archinal et al. 2011, Celest. Mech. Dyn. Astr. 109:101 (the 2009 report),
#   which the 2015 report keeps. Every number below was checked against
#   NAIF's pck00011.tpc, which encodes the 2015 report.
# Each entry: (ra0, ra_rate, dec0, dec_rate, ra_terms, dec_terms) with
#   ra  = ra0  + ra_rate  T + sum(a sin(A0 + A1 T))   over ra_terms  (a, A0, A1)
#   dec = dec0 + dec_rate T + sum(a cos(A0 + A1 T))   over dec_terms (a, A0, A1)
# T = Julian centuries (TDB) from J2000.0; angles in degrees, rates in
# degrees per century (the Moon's E1..E13 in the PCK's per-century form of
# the report's per-day expressions).
_IAU_POLE_MODELS: dict[str, tuple] = {
    "Sun":     (286.13, 0.0, 63.87, 0.0, (), ()),
    "Mercury": (281.0103, -0.0328, 61.4155, -0.0049, (), ()),
    "Venus":   (272.76, 0.0, 67.16, 0.0, (), ()),
    "Earth":   (0.00, -0.641, 90.00, -0.557, (), ()),
    "Moon":    (269.9949, 0.0031, 66.5392, 0.0130,
                ((-3.8787, 125.045,  -1935.5364525),    # E1
                 (-0.1204, 250.089,  -3871.0729050),    # E2
                 ( 0.0700, 260.008, 475263.3328725),    # E3
                 (-0.0172, 176.625, 487269.6299850),    # E4
                 ( 0.0072, 311.589, 964468.4993100),    # E6
                 (-0.0052,  15.134,  -5806.6093575),    # E10
                 ( 0.0043,  25.053, 473327.7964200)),   # E13
                (( 1.5419, 125.045,  -1935.5364525),    # E1
                 ( 0.0239, 250.089,  -3871.0729050),    # E2
                 (-0.0278, 260.008, 475263.3328725),    # E3
                 ( 0.0068, 176.625, 487269.6299850),    # E4
                 (-0.0029, 311.589, 964468.4993100),    # E6
                 ( 0.0009, 134.963, 477198.8693250),    # E7
                 ( 0.0008,  15.134,  -5806.6093575),    # E10
                 (-0.0009,  25.053, 473327.7964200))),  # E13
    # Mars: the Kuchynka et al. 2014 model adopted by the 2015 report. The
    # slow 0.5042615 deg/cy terms carry 0.42 deg (RA) and 1.59 deg (Dec).
    "Mars":    (317.269202, -0.10927547, 54.432516, -0.05827105,
                ((0.000068, 198.991226, 19139.4819985),
                 (0.000238, 226.292679, 38280.8511281),
                 (0.000052, 249.663391, 57420.7251593),
                 (0.000009, 266.183510, 76560.6367950),
                 (0.419057,  79.398797,     0.5042615)),
                ((0.000051, 122.433576, 19139.9407476),
                 (0.000141,  43.058401, 38280.8753272),
                 (0.000031,  57.663379, 57420.7517205),
                 (0.000005,  79.476401, 76560.6495004),
                 (1.591274, 166.325722,     0.5042615))),
    # Jupiter (angles Ja..Je) and the Galilean satellites (angles J3..J8).
    "Jupiter": (268.056595, -0.006499, 64.495303, 0.002413,
                ((0.000117,  99.360714, 4850.4046),
                 (0.000938, 175.895369, 1191.9605),
                 (0.001432, 300.323162,  262.5475),
                 (0.000030, 114.012305, 6070.2476),
                 (0.002150,  49.511251,   64.3000)),
                (( 0.000050,  99.360714, 4850.4046),
                 ( 0.000404, 175.895369, 1191.9605),
                 ( 0.000617, 300.323162,  262.5475),
                 (-0.000013, 114.012305, 6070.2476),
                 ( 0.000926,  49.511251,   64.3000))),
    "Io":       (268.05, -0.009, 64.50, 0.003,
                 ((0.094, 283.90, 4850.7), (0.024, 355.80, 1191.3)),
                 ((0.040, 283.90, 4850.7), (0.011, 355.80, 1191.3))),
    "Europa":   (268.08, -0.009, 64.51, 0.003,
                 ((1.086, 355.80, 1191.3), (0.060, 119.90, 262.1),
                  (0.015, 229.80, 64.3), (0.009, 352.25, 2382.6)),
                 ((0.468, 355.80, 1191.3), (0.026, 119.90, 262.1),
                  (0.007, 229.80, 64.3), (0.002, 352.25, 2382.6))),
    "Ganymede": (268.20, -0.009, 64.57, 0.003,
                 ((-0.037, 355.80, 1191.3), (0.431, 119.90, 262.1),
                  (0.091, 229.80, 64.3)),
                 ((-0.016, 355.80, 1191.3), (0.186, 119.90, 262.1),
                  (0.039, 229.80, 64.3))),
    "Callisto": (268.72, -0.009, 64.83, 0.003,
                 ((-0.068, 119.90, 262.1), (0.590, 229.80, 64.3),
                  (0.010, 113.35, 6070.0)),
                 ((-0.029, 119.90, 262.1), (0.254, 229.80, 64.3),
                  (-0.004, 113.35, 6070.0))),
    "Saturn":    (40.589, -0.036, 83.537, -0.004, (), ()),
    "Enceladus": (40.66, -0.036, 83.52, -0.004, (), ()),
    "Titan":     (39.4827, 0.0, 83.4279, 0.0, (), ()),
    "Uranus":    (257.311, 0.0, -15.175, 0.0, (), ()),
    # Neptune: N = 357.85 + 52.316 T.
    "Neptune":   (299.36, 0.0, 43.46, 0.0,
                  ((0.70, 357.85, 52.316),),
                  ((-0.51, 357.85, 52.316),)),
    "Pluto":     (132.993, 0.0, -6.163, 0.0, (), ()),
}

_JD_J2000 = 2451545.0


def _iau_pole_radec(name: str, jd_tdb: float = _JD_J2000) -> tuple[float, float]:
    """IAU WGCCRE pole (RA, Dec), degrees, of `name` at TDB Julian date jd_tdb.

    Raises KeyError for a body with no model in _IAU_POLE_MODELS.
    """
    ra0, ra1, dec0, dec1, ra_terms, dec_terms = _IAU_POLE_MODELS[name]
    T = (jd_tdb - _JD_J2000) / 36525.0
    ra = ra0 + ra1 * T
    dec = dec0 + dec1 * T
    for a, A0, A1 in ra_terms:
        ra += a * math.sin(math.radians(A0 + A1 * T))
    for a, A0, A1 in dec_terms:
        dec += a * math.cos(math.radians(A0 + A1 * T))
    return ra, dec


def _iau_pole_unit(name: str, jd_tdb: float = _JD_J2000) -> np.ndarray:
    """ICRF unit vector along `name`'s IAU north pole at TDB Julian date jd_tdb."""
    return _pole_unit_from_radec(*_iau_pole_radec(name, jd_tdb))


# ── Canonical body parameters (radius, albedo, J2, pole) ───────────────────
# Sources:
#   radius_km, albedo: IAU Working Group 2015 / NASA Planetary Fact Sheets
#   J2: NASA Planetary Fact Sheets, Williams 2024 (planets); Pavlis et al. 2008 (Earth);
#       0.0 for bodies with no published value (small moons, TNOs)
#   pole_axis_unit: the IAU WGCCRE pole at J2000.0 (_iau_pole_unit) for every
#       body with a published model. _build_bodies_at_snapshot re-evaluates it
#       at the snapshot epoch, so each window starts with the pole of its own
#       date, then holds it fixed over the window. The largest drifts ignored
#       that way are Earth's precession (0.85 deg/cy) and the Moon's 18.6-yr
#       wobble (about 1.5 deg), whose J2 reaches other bodies at ~1e-13 m/s^2.
#       Bodies with no model and no zonal coefficients keep the +z default,
#       which no force layer reads.
@dataclass(frozen=True)
class BodyParams:
    """Per-body physical parameters (canonical IAU/NASA values).

    For each oblate body we now also carry j3, j4, love_number_k2, and
    pole_axis_unit so the higher-order zonal-harmonic toggles (j3_zonal,
    j4_zonal) and the tidal_force toggle have data to operate on, and so
    J2/J3/J4 are evaluated in the body's true rotational frame rather than
    blindly along ICRS +z.

    love_number_k2: dimensionless tidal Love number (gates tidal_force).
                    Earth 0.30, Moon 0.024, Mars 0.169, Jupiter 0.535,
                    Saturn 0.39, Io 0.7 (tidally hot), Europa 0.26,
                    Ganymede 0.13. Sun ~0.02 (small). Defaults to 0 for
                    moons/dwarfs without published values -- the
                    tidal_force toggle then has no effect for those.
    """
    radius_km:      float
    albedo:         float
    j2:             float
    j3:             float                       = 0.0
    j4:             float                       = 0.0
    love_number_k2: float                       = 0.0
    pole_axis_unit: np.ndarray | None           = None  # None → ICRS +z default


# IAU WGCCRE poles at J2000.0 (see _IAU_POLE_MODELS above); the pole at the
# window's own epoch replaces these in _build_bodies_at_snapshot.
_SUN_POLE     = _iau_pole_unit("Sun")       # RA 286.13,  Dec 63.87 (constant)
_URANUS_POLE  = _iau_pole_unit("Uranus")    # RA 257.311, Dec -15.175 (constant)
_PLUTO_POLE   = _iau_pole_unit("Pluto")     # RA 132.993, Dec -6.163 (constant)
# Neptune: RA 299.36 + 0.70 sin N, Dec 43.46 - 0.51 cos N, which at J2000 is
# RA 299.33, Dec 42.95: 47.05 deg from ICRS +z, not "nearly +z".
_NEPTUNE_POLE = _iau_pole_unit("Neptune")
# Triton carries no zonal coefficients here (J2 = J3 = J4 = 0), so no force
# layer reads its pole; -Neptune's pole is a placeholder, not Triton's IAU pole.
_TRITON_POLE = -_NEPTUNE_POLE


_BODY_PARAMS: dict[str, BodyParams] = {
    # body         R(km)     albedo    J2          J3         J4        k2     pole
    # Sun J2 = 2.1106e-7 is the DE440 fitted value (Park et al. 2021,
    # "The JPL Planetary and Lunar Ephemerides DE440 and DE441", AJ 161:105,
    # Table 3). Earlier table value was 2.0e-7 placeholder.
    "Sun":      BodyParams(695700.0,  0.000,  2.1106e-7, 0.0,       -9.0e-7,
                            love_number_k2=0.02,
                            pole_axis_unit=_SUN_POLE),
    "Mercury":  BodyParams(  2440.0,  0.088,  5.03e-5,   0.0,       -1.0e-5,
                            love_number_k2=0.45,
                            pole_axis_unit=_iau_pole_unit("Mercury")),
    "Venus":    BodyParams(  6052.0,  0.689,  4.46e-6,   0.0,        0.0,
                            love_number_k2=0.295,
                            pole_axis_unit=_iau_pole_unit("Venus")),
    "Earth":    BodyParams(  6371.0,  0.367,  1.0826e-3, -2.5e-6,   -1.6e-6,
                            love_number_k2=0.299,
                            pole_axis_unit=_iau_pole_unit("Earth")),
    "Moon":     BodyParams(  1737.4,  0.120,  2.034e-4,   8.5e-6,   -1.2e-5,
                            love_number_k2=0.024,
                            pole_axis_unit=_iau_pole_unit("Moon")),
    "Mars":     BodyParams(  3389.5,  0.170,  1.9606e-3,  3.5e-5,   -1.4e-5,
                            love_number_k2=0.169,
                            pole_axis_unit=_iau_pole_unit("Mars")),
    "Phobos":   BodyParams(    11.2,  0.071,  0.0),
    "Deimos":   BodyParams(     6.1,  0.068,  0.0),
    "Jupiter":  BodyParams( 71492.0,  0.520,  1.4736e-2,  0.0,      -5.87e-4,
                            love_number_k2=0.535,
                            pole_axis_unit=_iau_pole_unit("Jupiter")),
    "Io":       BodyParams(  1821.6,  0.630,  1.846e-3,   0.0,        0.0,
                            love_number_k2=0.7,
                            pole_axis_unit=_iau_pole_unit("Io")),
    "Europa":   BodyParams(  1560.8,  0.670,  4.355e-4,   0.0,        0.0,
                            love_number_k2=0.26,
                            pole_axis_unit=_iau_pole_unit("Europa")),
    "Ganymede": BodyParams(  2634.1,  0.430,  1.276e-4,   0.0,        0.0,
                            love_number_k2=0.13,
                            pole_axis_unit=_iau_pole_unit("Ganymede")),
    "Callisto": BodyParams(  2410.3,  0.170,  3.5e-5,     0.0,        0.0,
                            love_number_k2=0.05,
                            pole_axis_unit=_iau_pole_unit("Callisto")),
    "Saturn":   BodyParams( 60268.0,  0.470,  1.6298e-2,  0.0,      -9.15e-4,
                            love_number_k2=0.39,
                            pole_axis_unit=_iau_pole_unit("Saturn")),
    # Saturn moons -- added 2026-05-15 to provide the perturbations missing
    # from earlier 26-body fixture. Mimas--Tethys 3:2 resonance and
    # Enceladus--Dione 2:1 resonance now both computable; expected to drop
    # Enceladus prediction error by ~2 orders of magnitude.
    # J2 values are unmeasured / sub-precision for these small icy moons --
    # set to 0 so the j2_zonal toggle skips them.
    "Mimas":    BodyParams(   198.2,  0.962,  0.0,        0.0,        0.0),
    "Enceladus":BodyParams(   252.1,  0.990,  5.4e-3,     0.0,        0.0,
                            love_number_k2=0.01,
                            pole_axis_unit=_iau_pole_unit("Enceladus")),
    "Tethys":   BodyParams(   531.0,  0.800,  0.0,        0.0,        0.0),
    "Dione":    BodyParams(   561.4,  0.998,  0.0,        0.0,        0.0),
    "Rhea":     BodyParams(   763.8,  0.949,  0.0,        0.0,        0.0),
    "Titan":    BodyParams(  2574.7,  0.220,  3.318e-5,   0.0,        0.0,
                            love_number_k2=0.6,
                            pole_axis_unit=_iau_pole_unit("Titan")),
    "Uranus":   BodyParams( 25559.0,  0.510,  3.5107e-3,  0.0,      -3.4e-5,
                            love_number_k2=0.36,
                            pole_axis_unit=_URANUS_POLE),
    "Miranda":  BodyParams(   235.8,  0.320,  0.0),
    "Ariel":    BodyParams(   578.9,  0.530,  0.0),
    "Umbriel":  BodyParams(   584.7,  0.260,  0.0),
    "Titania":  BodyParams(   788.9,  0.350,  0.0),
    "Oberon":   BodyParams(   761.4,  0.310,  0.0),
    "Neptune":  BodyParams( 24764.0,  0.410,  3.539e-3,   0.0,      -3.5e-5,
                            love_number_k2=0.4,
                            pole_axis_unit=_NEPTUNE_POLE),
    "Triton":   BodyParams(  1353.4,  0.760,  0.0,        0.0,        0.0,
                            pole_axis_unit=_TRITON_POLE),
    "Pluto":    BodyParams(  1188.3,  0.520,  0.0,        0.0,        0.0,
                            pole_axis_unit=_PLUTO_POLE),
    "Charon":   BodyParams(   606.0,  0.350,  0.0),
}


# ── Predictor definitions ─────────────────────────────────────────────────

@dataclass(frozen=True)
class Predictor:
    """A predictor specification.

    The `toggles` field carries the full physics configuration (which
    force layers are enabled). All new ablation variants are produced by
    `dataclasses.replace(some_toggles, flag=value)` — no scattered bool
    fields drifting out of sync.

    `use_kepler` is a special path (no n-body integration; per-body
    Keplerian fits). When True, `toggles` is ignored. `kepler_forecast`
    then picks the fit data: False fits every orbit on all fixture
    snapshots (in-sample, including the epochs it is scored on); True fits
    only on snapshots at or before the window start (an honest forecast).

    `fast_body_names` + `fast_substep_factor` enable per-body dt via
    NBodySystem.forest_ruth_step_hierarchical: listed bodies integrate at
    dt_days/fast_substep_factor while others stay at dt_days. Default
    behaviour (fast_body_names=None) uses uniform dt_days for all bodies.
    """
    name:                str
    toggles:             PhysicsToggles
    use_kepler:          bool   = False
    integrator:          str    = "fr4"  # "fr4" or "verlet"
    dt_days:             float  = 1.0
    description:         str    = ""
    fast_body_names:     tuple[str, ...] | None = None
    fast_substep_factor: int    = 1
    kepler_forecast:     bool   = False


# Common toggle bundles for the existing predictor lineup.
_NO_PHYSICS         = PhysicsToggles()  # all False -> pure Newton baseline
_GR_ONLY            = PhysicsToggles(gr_1pn=True)
_GR_SRP             = PhysicsToggles(gr_1pn=True, srp=True)
_GR_SRP_J2          = PhysicsToggles(gr_1pn=True, srp=True, j2_zonal=True)

# JPL DE440 canonical force model (Park et al. 2021, AJ 161:105):
#   - Newton + full N-body 1PN EIH (eih_cross, replaces single-body gr_1pn)
#   - Sun J2 + Earth J2 + Moon J2 + planetary J2 (j2_zonal)
#   - J3 zonal harmonics where measured (j3_zonal)
#   - J4 zonal (Saturn, Jupiter, Earth -- j4_zonal)
#   - Solar radiation pressure on small bodies (srp)
#   - Mutual tidal force via Love numbers (tidal_force)
#
# Not yet in our model: lunar full 6DOF orientation, asteroid masses.
#
# J4 history: j4_zonal was temporarily disabled 2026-05-15 because it
# appeared to REGRESS Enceladus by +1.25% / +20% at 3y / 5y horizons.
# Root cause was missing Dione (Enceladus's 2:1 resonance partner) in
# the DE440 fixture -- J4 was being layered on a baseline that was
# already wrong from the missing perturber. After the 2026-05-15
# fixture extension added Mimas/Tethys/Dione/Rhea (see
# misc/saturn_enceladus_j4_verdict_2026-05-15.md), j4_zonal was re-tested:
#   Mimas:     -49.2%   (large improvement -- J4 actually helps)
#   Enceladus:  +0.76%  (down from +1.25%, now in dt-noise floor)
#   Tethys/Dione/Rhea: <0.5% (noise)
#   Titan:     -0.27%   (slight improvement)
# Re-enabling j4_zonal as canonical. The Enceladus residual at 3e-3 AU is
# now dt=0.1d integrator noise (1.37-day orbit gets only ~14 steps/orbit),
# not a J4 formula issue.
_JPL_DE440          = PhysicsToggles(
    eih_cross=True,
    srp=True,
    j2_zonal=True,
    j3_zonal=True,
    j4_zonal=True,
    tidal_force=True,
)


# Measured hygiene (window j2015 -> j2016 against the DE440 fixture, after the
# pole fix; scratch evidence of fix track F, 2026-10-10). The four dt = 1 d
# predictors below do not resolve the inner moons: Phobos, Deimos, Io,
# Miranda, Mimas and Tethys end the year unbound (with GR + SRP + J2 also
# Enceladus and Ariel), and Mercury is off by 31,000-34,000 km, Earth by
# ~900 km, Jupiter by 10,000-14,000 km (recoil from the ejected moons), so
# their differences measure the integrator, not GR, SRP or J2 (the 1PN effect
# on Mercury is ~230 km in that year). gr_1pn (standard, over_physics*) is a
# rest-frame approximation: for the moons it misses 19-97 % of the EIH
# differential acceleration (1-yr position: Moon 3.4 km of a 6.0 km 1PN
# effect, Phobos 20 of 38 km). Which predictors, dt and GR model are canonical
# for the recalculation is the Captain's decision; these entries are unchanged.
PREDICTORS: list[Predictor] = [
    Predictor("pure_newton",        _NO_PHYSICS,
              description="Newton only (no GR, no SRP, no J2) -- baseline; "
                          "dt=1d ejects the inner moons"),
    Predictor("standard",           _GR_ONLY,
              description="Newton + 1PN GR (gr_1pn, no SRP, no J2); "
                          "dt=1d ejects the inner moons"),
    Predictor("over_physics_no_j2", _GR_SRP,
              description="Newton + GR + SRP (J2 ablated) -- diagnostic; "
                          "dt=1d ejects the inner moons"),
    Predictor("over_physics",       _GR_SRP_J2,
              description="Newton + GR + SRP + J2 at dt=1d (ejects the inner "
                          "moons; not the best predictor)"),
    Predictor("over_physics_finedt", _GR_SRP_J2,
              dt_days=0.1,
              description="Over_physics with dt=0.1d (10x finer; targets fast-moon row)"),
    Predictor("jpl_de440",          _JPL_DE440,
              dt_days=0.1,
              description="JPL DE440 canonical: EIH N-body 1PN + J2/J3/J4 zonals + tides + SRP"),
    Predictor("jpl_de440_finer",    _JPL_DE440,
              dt_days=0.02,
              description="JPL DE440 stack with dt=0.02d (5x finer) -- targets fast-moon "
                          "integrator-noise floor (Enceladus, Io, Europa at 1-2d period)"),
    # RESPA per-body dt: macro dt=0.1d for slow forces (Sun on moons, J2/J3/J4,
    # EIH cross-terms, etc.), inner dt=0.01d for the parent-pair Newton force
    # (Saturn-Mimas, Mars-Phobos, etc.). Symplectic via Strang splitting; it
    # relies on the parent's pull dominating the Sun's, which holds for the
    # giant planets' moons (Saturn pulls Mimas 10000x harder than the Sun)
    # but NOT for the Moon: the Sun pulls the Moon 2.2x harder than Earth
    # does, the regime respa_step's docstring says fails. Measured over
    # j2015 -> j2016: Moon 5,635 km (jpl_de440: 10 km), and the Moon's error
    # leaks into Earth (74 km against 1.3). Taking the Moon out of the fast
    # group does not help (6,835 km), and taking the planets' Sun pairs out
    # wrecks the planets (Mercury 21,000 km), so the cure is in respa_step,
    # not in these parent links. The giant planets' moons do gain over
    # jpl_de440 at 0.1 d (Europa 61,300 vs 407,400 km) but stay far behind
    # uniform FR4 at 0.02 d (1,600 km).
    # Parent-attractor links are read from the DE440 fixture's
    # gravitational_anchor field. See misc/respa_attempt_2026-05-16.md.
    Predictor("jpl_de440_respa",    _JPL_DE440,
              dt_days=0.1,
              integrator="respa",
              fast_substep_factor=10,
              description="JPL DE440 with RESPA symplectic multi-timestep: "
                          "slow forces @ 0.1d, parent-pair Newton @ 0.01d"),
    # NOTE (2026-05-16): jpl_de440_hier removed from canonical PREDICTORS
    # because the underlying forest_ruth_step_hierarchical method has a
    # known correctness bug in the slow/fast operator split (the slow-body
    # advancement under the frozen-fast-body assumption produces a wrong
    # slow trajectory, which then mis-leads the fast-body substeps).
    # Validation tests in test_nbody.py demonstrate this:
    #   - 2-body Earth-Moon hierarchical(1d/0.1d) is 8.5x WORSE than uniform 1d
    #   - 3-body Sun-Earth-Moon similar
    # See misc/hierarchical_dt_known_bug_2026-05-16.md.
    # The methods _selective_fr_step and forest_ruth_step_hierarchical
    # remain in nbody.py for future repair; the right fix is symplectic
    # multi-timestep (Tuckerman 1992 RESPA, split by force type) rather
    # than the body-split operator scheme.
    # Two Kepler baselines (the canonical one is the Captain's call):
    # "kepler" keeps its historical meaning, an IN-SAMPLE fit on all 27
    # snapshots J2000-J2026, including the epochs it is scored on;
    # "kepler_forecast" fits only snapshots at or before the window start,
    # the honest forecast (NaN for windows with fewer than 3 such snapshots).
    # In both, a moon rides on its parent's Kepler prediction at the sample
    # epoch, not on the parent frozen at the window start.
    Predictor("kepler",             _NO_PHYSICS, use_kepler=True,
              description="2-body Keplerian fit per body, in-sample "
                          "(fitted on all snapshots incl. the scored ones)"),
    Predictor("kepler_forecast",    _NO_PHYSICS, use_kepler=True,
              kepler_forecast=True,
              description="2-body Keplerian fit per body on snapshots up to "
                          "the window start only (honest forecast)"),
]


# ── Initial state extraction ──────────────────────────────────────────────

def _load_de440() -> dict:
    return json.loads(_DE440_FILE.read_text())


def _build_bodies_at_snapshot(
    snap_key: str,
    de440:    dict,
    predictor: Predictor,
) -> tuple[list[CelestialBody], list[str], int, list[int | None]]:
    """Build CelestialBody list from a DE440 annual snapshot.

    Per-body parameters (j2, j3, j4, area_m2) are ALWAYS set from _BODY_PARAMS
    regardless of the predictor's toggles; pole_axis_unit is the IAU pole at
    the snapshot epoch (_iau_pole_unit) for bodies with a published model, else
    the _BODY_PARAMS value. Predictor-level
    gating now happens at the force-computation layer inside NBodySystem
    (via PhysicsToggles), not by zeroing per-body data here.

    Returns (bodies, names, sun_idx, parent_attractor_indices).
    parent_attractor_indices[i] = index of body i's `gravitational_anchor`
    in the same list, or None for Sun / unanchored bodies. Used by
    NBodySystem.respa_step to identify parent-pair fast forces.
    """
    snap = de440["snapshots"][snap_key]
    snap_jd = snap["epoch"]["jd_tdb"]
    bodies: list[CelestialBody] = []
    names: list[str] = []
    anchor_names: list[str | None] = []  # parallel list of anchor body names
    sun_idx = -1

    for entry in snap["bodies"]:
        name = entry["name"]
        sv   = entry["state_vector"]
        gm   = entry["gm_km3_s2"] * 1e9            # -> m³/s²
        mass = gm / _G
        pos  = np.array([sv["x_km"],  sv["y_km"],  sv["z_km"]])  * KM_TO_M
        vel  = np.array([sv["vx_km_s"],sv["vy_km_s"],sv["vz_km_s"]]) * KM_TO_M

        params = _BODY_PARAMS.get(name, BodyParams(1.0, 0.0, 0.0))
        r_m = params.radius_km * KM_TO_M

        # SRP-relevant cross-section (None for Sun, since Sun is the emitter).
        # Always populated; toggle.srp + solar_luminosity_W > 0 decides
        # whether the SRP block runs.
        area_m2 = math.pi * r_m * r_m if (name != "Sun") else 0.0

        # Zonal harmonics act about the body's spin axis at this epoch.
        pole = (_iau_pole_unit(name, snap_jd) if name in _IAU_POLE_MODELS
                else params.pole_axis_unit)

        body = CelestialBody(
            mass_kg=mass,
            position_m=pos,
            velocity_m_s=vel,
            radius_m=r_m,
            love_number_k2=params.love_number_k2,  # was 0.0 -- bug fix for tidal_force
            area_m2=area_m2,
            reflectivity=params.albedo,
            j2=params.j2,
            j3=params.j3,
            j4=params.j4,
            pole_axis_unit=pole,
        )
        bodies.append(body)
        names.append(name)
        anchor_names.append(entry.get("gravitational_anchor"))
        if name == "Sun":
            sun_idx = len(bodies) - 1

    if sun_idx < 0:
        raise ValueError(f"snapshot {snap_key} has no Sun")

    # Resolve anchor names to indices. Bodies whose anchor isn't present in
    # this snapshot (or is the body itself) get None.
    name_to_idx = {n: i for i, n in enumerate(names)}
    parent_attractor_indices: list[int | None] = []
    for i, a_name in enumerate(anchor_names):
        if a_name is None or a_name == names[i]:
            parent_attractor_indices.append(None)
        else:
            parent_attractor_indices.append(name_to_idx.get(a_name))

    return bodies, names, sun_idx, parent_attractor_indices


def _de440_heliocentric_km(de440: dict, snap_key: str, name: str) -> np.ndarray | None:
    snap = de440["snapshots"].get(snap_key)
    if snap is None:
        return None
    by_name = {b["name"]: b for b in snap["bodies"]}
    if name not in by_name or "Sun" not in by_name:
        return None
    def _sv(b):
        sv = b["state_vector"]
        return np.array([sv["x_km"], sv["y_km"], sv["z_km"]])
    return _sv(by_name[name]) - _sv(by_name["Sun"])


# ── N-body predictor integration ─────────────────────────────────────────

def _integrate_nbody(
    predictor:        Predictor,
    bodies:           list[CelestialBody],
    body_names:       list[str],
    sun_idx:          int,
    start_jd:         float,
    sample_jds:       list[float],
    parent_attractor_indices: list[int | None] | None = None,
) -> dict[str, np.ndarray]:
    """Forward-integrate from start_jd, sample heliocentric km at sample_jds.

    sample_jds[0] must equal start_jd (the trivial sample = initial state).

    parent_attractor_indices: optional per-body parent attractor indices,
    used by integrator='respa' to identify the fast Newton pair for each
    body. Read from DE440 fixture's gravitational_anchor field upstream.

    Returns
    -------
    dict mapping body_name → array of shape (len(sample_jds), 3),
    each row is heliocentric position in km at the corresponding sample JD.
    """
    if sample_jds[0] != start_jd:
        raise ValueError("sample_jds[0] must equal start_jd")

    dt_s = predictor.dt_days * DAY_S
    # Toggles dictate which force layers run; solar_luminosity_W is a
    # parameter (need a value for L_sun), gated by toggles.srp.
    system = NBodySystem(
        bodies,
        toggles=predictor.toggles,
        solar_luminosity_W=_L_SUN_W if predictor.toggles.srp else 0.0,
        parent_attractor_indices=parent_attractor_indices,
    )

    # Resolve fast-body indices (for hierarchical substepping). Bodies named
    # in predictor.fast_body_names that aren't present in this fixture snapshot
    # are silently skipped -- caller can specify a generous list without
    # tripping the integrator on snapshots where some bodies are absent.
    fast_indices: list[int] = []
    if predictor.fast_body_names and predictor.fast_substep_factor > 1:
        name_to_idx = {n: i for i, n in enumerate(body_names)}
        for fname in predictor.fast_body_names:
            if fname in name_to_idx:
                fast_indices.append(name_to_idx[fname])

    if predictor.integrator == "respa":
        sub = max(1, predictor.fast_substep_factor)
        step_fn: Callable[[float], None] = (
            lambda dt: system.respa_step(dt, sub)
        )
    elif predictor.integrator == "fr4":
        if fast_indices:
            sub = predictor.fast_substep_factor
            step_fn = (
                lambda dt: system.forest_ruth_step_hierarchical(
                    dt, fast_indices, sub
                )
            )
        else:
            step_fn = lambda dt: system.forest_ruth_step(dt)
    else:
        step_fn = lambda dt: system.step(dt)

    # Sample storage: name → list of 3-vectors in km, heliocentric
    samples: dict[str, list[np.ndarray]] = {n: [] for n in body_names}

    def _snapshot() -> None:
        sun_pos = system.bodies[sun_idx].position_m
        for i, n in enumerate(body_names):
            hel_km = (system.bodies[i].position_m - sun_pos) / KM_TO_M
            samples[n].append(hel_km.copy())

    # t = sample_jds[0]: initial state
    _snapshot()

    current_jd = start_jd
    for target_jd in sample_jds[1:]:
        dt_total_days = target_jd - current_jd
        n_steps = max(1, int(round(dt_total_days / predictor.dt_days)))
        # use exact step size that lands on target_jd
        exact_dt_s = (dt_total_days * DAY_S) / n_steps
        for _ in range(n_steps):
            step_fn(exact_dt_s)
        current_jd = target_jd
        _snapshot()

    return {n: np.array(samples[n]) for n in body_names}


# ── Kepler predictor (special path) ──────────────────────────────────────

def _kepler_predictions(
    de440:           dict,
    anchor_snap_key: str,
    body_names:      list[str],
    sample_jds:      list[float],
    forecast:        bool = False,
) -> dict[str, np.ndarray]:
    """Per-body Keplerian predictions, heliocentric km at the sample JDs.

    Each body is its fitted two-body orbit about its anchor (orbital.
    ANCHOR_MAP) added to the anchor's own Kepler prediction AT THE SAME
    SAMPLE EPOCH, recursively down to the Sun, which is the heliocentric
    origin. A moon therefore rides on its parent's predicted motion rather
    than on the parent frozen at the window start, and no DE440 state at a
    sample epoch enters any prediction.

    forecast=False: orbits are fitted on every fixture snapshot (J2000-J2026),
    including the epochs they are scored on -- an in-sample fit.
    forecast=True: orbits are fitted only on snapshots at or before the window
    start, an honest forecast. fit_orbit needs three snapshots, so windows
    starting before J2002 get NaN.

    Returns heliocentric km arrays of shape (len(sample_jds), 3) per body:
    zeros for the Sun, NaN for a body (or any ancestor) without a fit.
    """
    if forecast:
        start_jd = de440["snapshots"][anchor_snap_key]["epoch"]["jd_tdb"]
        fixture = {"snapshots": {k: v for k, v in de440["snapshots"].items()
                                 if v["epoch"]["jd_tdb"] <= start_jd}}
    else:
        fixture = de440

    n = len(sample_jds)
    helio: dict[str, np.ndarray] = {"Sun": np.zeros((n, 3))}

    def _helio(name: str, chain: tuple[str, ...] = ()) -> np.ndarray:
        if name in helio:
            return helio[name]
        orbit = fit_orbit(name, fixture) if name not in chain else None
        if orbit is None:
            out = np.full((n, 3), np.nan)
        else:
            parent = _helio(orbit.anchor, chain + (name,))
            rel = np.array([_ecliptic_to_equatorial(*orbit.predict_relative_ecliptic(jd))
                            for jd in sample_jds])
            out = parent + rel
        helio[name] = out
        return out

    return {name: _helio(name).copy() for name in body_names}


# ── Result dataclass ─────────────────────────────────────────────────────

@dataclass
class WindowSample:
    """One (predictor, body, window, sample-time) cell of the error tensor."""
    predictor:       str
    body:            str
    window_start_jd: float
    sample_jd:       float
    predicted_km:    list[float]      # heliocentric (x, y, z) in km
    truth_km:        list[float] | None
    error_au:        float | None     # |predicted - truth| in AU


def _samples_to_dict(s: WindowSample) -> dict:
    return {
        "predictor":       s.predictor,
        "body":            s.body,
        "window_start_jd": s.window_start_jd,
        "sample_jd":       s.sample_jd,
        "predicted_km":    s.predicted_km,
        "truth_km":        s.truth_km,
        "error_au":        s.error_au,
    }


# ── Main entry point ─────────────────────────────────────────────────────

def run_rolling_shootout(
    window_start_keys:      list[str],
    prediction_horizon_yr:  float,
    sample_interval_yr:     float = 1.0,
    predictors:             list[Predictor] = PREDICTORS,
    de440:                  dict | None = None,
    verbose:                bool = True,
    output_path:            "Path | str | None" = _USE_DEFAULT_OUTPUT,
) -> list[WindowSample]:
    """Run the rolling-window shootout.

    Parameters
    ----------
    window_start_keys     : DE440 snapshot keys to use as window starts
                           (e.g. ["j2005", "j2006", ..., "j2020"])
    prediction_horizon_yr : how far into the future each window predicts
    sample_interval_yr    : sampling cadence within each window. Samples sit
                            at start + k * sample_interval_yr (k = 1, 2, ...
                            up to the horizon); one within half a day of a
                            DE440 snapshot is placed exactly on it and scored.
                            With the annual fixture, 1.0 scores every sample;
                            sub-annual samples between snapshots carry the
                            prediction with truth_km = error_au = None.
    predictors            : list of Predictor specs (default: PREDICTORS)
    output_path           : where to write the results JSON. Default: the
                            untracked _DEFAULT_RESULTS_OUTPUT (local-cache/);
                            None writes nothing. The committed
                            fixtures/rolling_shootout_results.json is written
                            only when passed here explicitly.

    Returns
    -------
    list of WindowSample, one per (predictor, body, window, sample_time).
    """
    if de440 is None:
        de440 = _load_de440()
    if not sample_interval_yr > 0.0:
        raise ValueError(f"sample_interval_yr must be > 0, got {sample_interval_yr}")
    if output_path is _USE_DEFAULT_OUTPUT:
        output_path = _DEFAULT_RESULTS_OUTPUT

    snap_keys_all = sorted(de440["snapshots"].keys())
    # JD lookup helper
    def _jd_of(snap_key: str) -> float:
        return de440["snapshots"][snap_key]["epoch"]["jd_tdb"]

    out: list[WindowSample] = []
    n_windows = len(window_start_keys)

    t0_global = time.time()

    for w_idx, w_key in enumerate(window_start_keys):
        start_jd = _jd_of(w_key)
        # Samples every sample_interval_yr from the window start. A sample
        # within half a day of a DE440 snapshot is moved exactly onto it, so
        # prediction and truth refer to the SAME instant (the annual
        # snapshots are exactly 365.25 d apart, so an annual cadence lands
        # on them exactly).
        n_samples = int(math.floor(prediction_horizon_yr / sample_interval_yr + 1e-9))
        sample_jds: list[float] = [start_jd]
        sample_snap_keys: list[str | None] = [w_key]
        for k in range(1, n_samples + 1):
            jd = start_jd + k * sample_interval_yr * YR_DAYS
            key = next((sk for sk in snap_keys_all
                        if abs(_jd_of(sk) - jd) <= 0.5), None)
            sample_jds.append(_jd_of(key) if key is not None else jd)
            sample_snap_keys.append(key)

        if verbose:
            print(f"\n  Window {w_idx+1}/{n_windows}: start={w_key} (JD={start_jd:.1f})")
            print(f"    horizon={prediction_horizon_yr}y, "
                  f"samples={len(sample_jds)}, "
                  f"truth coverage={sum(1 for k in sample_snap_keys if k)}/"
                  f"{len(sample_snap_keys)}")

        # Build initial state ONCE per window — same for all predictors.
        # But CelestialBody param values differ by predictor (j2_enabled, srp).
        for predictor in predictors:
            t0 = time.time()

            if predictor.use_kepler:
                # In-sample fit on all snapshots, or (kepler_forecast) only
                # on snapshots up to the window start.
                body_names = [b["name"] for b in de440["snapshots"][w_key]["bodies"]]
                preds = _kepler_predictions(de440, w_key, body_names, sample_jds,
                                            forecast=predictor.kepler_forecast)
            else:
                bodies, body_names, sun_idx, parent_idx = _build_bodies_at_snapshot(
                    w_key, de440, predictor)
                preds = _integrate_nbody(
                    predictor, bodies, body_names, sun_idx, start_jd, sample_jds,
                    parent_attractor_indices=parent_idx)

            elapsed = time.time() - t0
            if verbose:
                print(f"    {predictor.name:22s}  done in {elapsed:5.1f}s")

            # Build WindowSamples
            for body_name, pred_km_arr in preds.items():
                if body_name == "Sun":
                    continue   # Sun is the origin in heliocentric frame
                for s_idx, sjd in enumerate(sample_jds):
                    pred = pred_km_arr[s_idx]
                    truth = None
                    err_au = None
                    snap_key = sample_snap_keys[s_idx]
                    if snap_key is not None:
                        truth_vec = _de440_heliocentric_km(de440, snap_key, body_name)
                        if truth_vec is not None:
                            truth = truth_vec.tolist()
                            if not np.any(np.isnan(pred)):
                                err_au = float(np.linalg.norm(pred - truth_vec)) / AU_KM
                    out.append(WindowSample(
                        predictor=predictor.name,
                        body=body_name,
                        window_start_jd=start_jd,
                        sample_jd=sjd,
                        predicted_km=[float(x) for x in pred] if not np.any(np.isnan(pred))
                                     else [None, None, None],
                        truth_km=truth,
                        error_au=err_au,
                    ))

    if output_path is None:
        if verbose:
            print(f"\n  Total elapsed: {time.time() - t0_global:.1f}s "
                  f"({len(out)} samples, not written)")
        return out

    payload = {
        "metadata": {
            "n_windows":             n_windows,
            "window_start_keys":     window_start_keys,
            "prediction_horizon_yr": prediction_horizon_yr,
            "sample_interval_yr":    sample_interval_yr,
            "predictors":            [p.name for p in predictors],
            "elapsed_s":             round(time.time() - t0_global, 1),
        },
        "samples": [_samples_to_dict(s) for s in out],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")))

    if verbose:
        print(f"\n  Total elapsed: {time.time() - t0_global:.1f}s")
        print(f"  {len(out)} samples written -> {output_path}")

    return out


# ── Convenience CLI ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rolling-window n-body prediction shootout"
    )
    parser.add_argument("--pass", dest="which_pass", choices=["short", "long", "both"],
                        default="short",
                        help="window strategy: short (5y horizon, 1y slide), "
                             "long (20y horizon, 5y slide), or both")
    parser.add_argument("--predictors", nargs="+", default=None,
                        help="predictor names to run (default: all)")
    parser.add_argument("--output", type=Path, default=_DEFAULT_RESULTS_OUTPUT,
                        help="results JSON (default: the untracked "
                             f"{_DEFAULT_RESULTS_OUTPUT}); with --pass both the "
                             "long pass goes to the same name with _long added")
    args = parser.parse_args()

    selected = PREDICTORS
    if args.predictors:
        selected = [p for p in PREDICTORS if p.name in args.predictors]

    all_results: list[WindowSample] = []

    if args.which_pass in ("short", "both"):
        # 16 windows: J2005..J2020, 5y horizon
        keys = [f"j{y}" for y in range(2005, 2021)]
        results = run_rolling_shootout(
            window_start_keys=keys,
            prediction_horizon_yr=5.0,
            sample_interval_yr=1.0,
            predictors=selected,
            output_path=args.output,
        )
        all_results.extend(results)

    if args.which_pass in ("long", "both"):
        # Long: starts every 5y from J2000 to J2005, 20y horizon
        # (last possible start is J2006 for truth coverage to J2026)
        keys = [f"j{y}" for y in range(2000, 2007)]
        long_out = (args.output.with_name(args.output.stem + "_long" + args.output.suffix)
                    if args.which_pass == "both" else args.output)
        results = run_rolling_shootout(
            window_start_keys=keys,
            prediction_horizon_yr=20.0,
            sample_interval_yr=1.0,
            predictors=selected,
            output_path=long_out,
        )
        all_results.extend(results)
