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
  kepler               2-body Keplerian fit per body

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

Output is also saved to fixtures/rolling_shootout_results.json so the
analysis pass doesn't need to re-run the integrations.
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
from sigma_ground.field.interface.orbital import fit_orbit, predict_ssb_position


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
_RESULTS_FILE = _FIXTURES / "rolling_shootout_results.json"


# ── Canonical body parameters (radius, albedo, J2, pole) ───────────────────
# Sources:
#   radius_km, albedo: IAU Working Group 2015 / NASA Planetary Fact Sheets
#   J2: NASA Planetary Fact Sheets, Williams 2024 (planets); Pavlis et al. 2008 (Earth);
#       0.0 for bodies with no published value (small moons, TNOs)
#   pole_axis_unit: simplified — z-axis for all bodies. Solar system planets
#       have varying obliquities (Uranus ~97°!), but for this first-pass
#       experiment we use the ICRF frame's z-axis. A future refinement is to
#       set each body's true pole from IAU 2015 rotation models.
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


# IAU 2015 pole RA/Dec (J2000.0) for high-obliquity bodies.
# Most planets have poles within ~30° of ICRS +z so the default works fine;
# these need explicit vectors for J2/J3/J4 to be evaluated correctly.
_SUN_POLE    = _pole_unit_from_radec(286.13,   63.87)  # IAU 2015 solar pole
_URANUS_POLE = _pole_unit_from_radec(257.311, -15.175)
_PLUTO_POLE  = _pole_unit_from_radec(132.993,  -6.163)
# Neptune's pole RA/Dec (43.46°, 89.45° -- nearly +z but slightly tilted).
_NEPTUNE_POLE = _pole_unit_from_radec(299.36,  43.46)
# Triton orbits Neptune retrograde at ~157° to Neptune's equator. Triton's
# own rotational pole is roughly opposite Neptune's orbital normal; we use
# -Neptune's pole as a first approximation since Triton's own J2 = 0.
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
                            love_number_k2=0.45),
    "Venus":    BodyParams(  6052.0,  0.689,  4.46e-6,   0.0,        0.0,
                            love_number_k2=0.295),
    "Earth":    BodyParams(  6371.0,  0.367,  1.0826e-3, -2.5e-6,   -1.6e-6,
                            love_number_k2=0.299),
    "Moon":     BodyParams(  1737.4,  0.120,  2.034e-4,   8.5e-6,   -1.2e-5,
                            love_number_k2=0.024),
    "Mars":     BodyParams(  3389.5,  0.170,  1.9606e-3,  3.5e-5,   -1.4e-5,
                            love_number_k2=0.169),
    "Phobos":   BodyParams(    11.2,  0.071,  0.0),
    "Deimos":   BodyParams(     6.1,  0.068,  0.0),
    "Jupiter":  BodyParams( 71492.0,  0.520,  1.4736e-2,  0.0,      -5.87e-4,
                            love_number_k2=0.535),
    "Io":       BodyParams(  1821.6,  0.630,  1.846e-3,   0.0,        0.0,
                            love_number_k2=0.7),
    "Europa":   BodyParams(  1560.8,  0.670,  4.355e-4,   0.0,        0.0,
                            love_number_k2=0.26),
    "Ganymede": BodyParams(  2634.1,  0.430,  1.276e-4,   0.0,        0.0,
                            love_number_k2=0.13),
    "Callisto": BodyParams(  2410.3,  0.170,  3.5e-5,     0.0,        0.0,
                            love_number_k2=0.05),
    "Saturn":   BodyParams( 60268.0,  0.470,  1.6298e-2,  0.0,      -9.15e-4,
                            love_number_k2=0.39),
    # Saturn moons -- added 2026-05-15 to provide the perturbations missing
    # from earlier 26-body fixture. Mimas--Tethys 3:2 resonance and
    # Enceladus--Dione 2:1 resonance now both computable; expected to drop
    # Enceladus prediction error by ~2 orders of magnitude.
    # J2 values are unmeasured / sub-precision for these small icy moons --
    # set to 0 so the j2_zonal toggle skips them.
    "Mimas":    BodyParams(   198.2,  0.962,  0.0,        0.0,        0.0),
    "Enceladus":BodyParams(   252.1,  0.990,  5.4e-3,     0.0,        0.0,
                            love_number_k2=0.01),
    "Tethys":   BodyParams(   531.0,  0.800,  0.0,        0.0,        0.0),
    "Dione":    BodyParams(   561.4,  0.998,  0.0,        0.0,        0.0),
    "Rhea":     BodyParams(   763.8,  0.949,  0.0,        0.0,        0.0),
    "Titan":    BodyParams(  2574.7,  0.220,  3.318e-5,   0.0,        0.0,
                            love_number_k2=0.6),
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
    Keplerian fits). When True, `toggles` is ignored.

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


PREDICTORS: list[Predictor] = [
    Predictor("pure_newton",        _NO_PHYSICS,
              description="Newton only (no GR, no SRP, no J2) -- baseline"),
    Predictor("standard",           _GR_ONLY,
              description="Newton + 1PN GR (no SRP, no J2)"),
    Predictor("over_physics_no_j2", _GR_SRP,
              description="Newton + GR + SRP (J2 ablated) -- diagnostic"),
    Predictor("over_physics",       _GR_SRP_J2,
              description="Newton + GR + SRP + J2 (current best)"),
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
    Predictor("kepler",             _NO_PHYSICS, use_kepler=True,
              description="2-body Keplerian fit per body"),
]


# ── Initial state extraction ──────────────────────────────────────────────

def _load_de440() -> dict:
    return json.loads(_DE440_FILE.read_text())


def _build_bodies_at_snapshot(
    snap_key: str,
    de440:    dict,
    predictor: Predictor,
) -> tuple[list[CelestialBody], list[str], int]:
    """Build CelestialBody list from a DE440 annual snapshot.

    Per-body parameters (j2, j3, j4, area_m2, pole_axis_unit) are ALWAYS set
    from _BODY_PARAMS regardless of the predictor's toggles. Predictor-level
    gating now happens at the force-computation layer inside NBodySystem
    (via PhysicsToggles), not by zeroing per-body data here.

    This is the fix for the j2-zero bug: previously, `j2_enabled=False`
    zeroed `body.j2`, which lost the true data value and forced bodies to be
    re-instantiated for any ablation. Now the body always carries its real
    parameter values; the toggle decides whether the force is computed.
    """
    snap = de440["snapshots"][snap_key]
    bodies: list[CelestialBody] = []
    names: list[str] = []
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
            pole_axis_unit=params.pole_axis_unit,
        )
        bodies.append(body)
        names.append(name)
        if name == "Sun":
            sun_idx = len(bodies) - 1

    if sun_idx < 0:
        raise ValueError(f"snapshot {snap_key} has no Sun")
    return bodies, names, sun_idx


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
) -> dict[str, np.ndarray]:
    """Forward-integrate from start_jd, sample heliocentric km at sample_jds.

    sample_jds[0] must equal start_jd (the trivial sample = initial state).

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

    if predictor.integrator == "fr4":
        if fast_indices:
            sub = predictor.fast_substep_factor
            step_fn: Callable[[float], None] = (
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
) -> dict[str, np.ndarray]:
    """Per-body Keplerian fits from the anchor snapshot, evaluated at sample JDs.

    Returns heliocentric km arrays of shape (len(sample_jds), 3) per body.
    For bodies where fit_orbit returns None (Sun, missing data), returns
    an array of zeros — same shape, sentinel value.
    """
    anchor_snap = de440["snapshots"][anchor_snap_key]
    results: dict[str, np.ndarray] = {}

    for name in body_names:
        if name == "Sun":
            results[name] = np.zeros((len(sample_jds), 3))
            continue
        orbit = fit_orbit(name, de440)
        if orbit is None:
            results[name] = np.full((len(sample_jds), 3), np.nan)
            continue
        positions = []
        for jd in sample_jds:
            ssb_km = predict_ssb_position(orbit, anchor_snap, jd)
            if ssb_km is None:
                positions.append(np.array([np.nan, np.nan, np.nan]))
                continue
            sun_ssb = _de440_heliocentric_km(de440, anchor_snap_key, name)  # not what we want
            # We need the Sun's SSB position at jd. Use anchor snap Sun position
            # (DE440 snapshots store SSB-equatorial). Kepler predict gives SSB,
            # so heliocentric = SSB - Sun_SSB.
            sun_sv = [b for b in anchor_snap["bodies"] if b["name"] == "Sun"][0]["state_vector"]
            sun_pos = np.array([sun_sv["x_km"], sun_sv["y_km"], sun_sv["z_km"]])
            positions.append(np.array(ssb_km) - sun_pos)
        results[name] = np.array(positions)
    return results


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
) -> list[WindowSample]:
    """Run the rolling-window shootout.

    Parameters
    ----------
    window_start_keys     : DE440 snapshot keys to use as window starts
                           (e.g. ["j2005", "j2006", ..., "j2020"])
    prediction_horizon_yr : how far into the future each window predicts
    sample_interval_yr    : sampling cadence within each window (1.0 = annual,
                            which lets us compare directly to DE440 snapshots)
    predictors            : list of Predictor specs (default: all 5)

    Returns
    -------
    list of WindowSample, one per (predictor, body, window, sample_time).
    Also writes the same data to fixtures/rolling_shootout_results.json.
    """
    if de440 is None:
        de440 = _load_de440()

    snap_keys_all = sorted(de440["snapshots"].keys())
    # JD lookup helper
    def _jd_of(snap_key: str) -> float:
        return de440["snapshots"][snap_key]["epoch"]["jd_tdb"]

    out: list[WindowSample] = []
    n_windows = len(window_start_keys)

    t0_global = time.time()

    for w_idx, w_key in enumerate(window_start_keys):
        start_jd = _jd_of(w_key)
        # Sample JDs are the actual DE440 snapshot JDs that fall within
        # the prediction window. This ensures predictions are compared to
        # truth at the SAME instant — no time-alignment error.
        # (Using start_jd + round(k*365.25) instead gives Earth a spurious
        # ~0.005 AU error from snapshot epochs not being exactly 365.25d apart.)
        end_jd = start_jd + prediction_horizon_yr * YR_DAYS
        sample_jds: list[float] = [start_jd]
        sample_snap_keys: list[str | None] = [w_key]
        for snap_key in snap_keys_all:
            snap_jd = _jd_of(snap_key)
            if snap_jd <= start_jd + 0.5:
                continue  # at or before window start
            if snap_jd > end_jd + 0.5:
                break     # past prediction horizon
            sample_jds.append(snap_jd)
            sample_snap_keys.append(snap_key)

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
                # Anchor on the window start; fit orbit globally from de440 data
                body_names = [b["name"] for b in de440["snapshots"][w_key]["bodies"]]
                preds = _kepler_predictions(de440, w_key, body_names, sample_jds)
            else:
                bodies, body_names, sun_idx = _build_bodies_at_snapshot(
                    w_key, de440, predictor)
                preds = _integrate_nbody(
                    predictor, bodies, body_names, sun_idx, start_jd, sample_jds)

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

    # Save to fixture file
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
    _RESULTS_FILE.write_text(json.dumps(payload, separators=(",", ":")))

    if verbose:
        print(f"\n  Total elapsed: {time.time() - t0_global:.1f}s")
        print(f"  {len(out)} samples written -> {_RESULTS_FILE.name}")

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
        )
        all_results.extend(results)

    if args.which_pass in ("long", "both"):
        # Long: starts every 5y from J2000 to J2005, 20y horizon
        # (last possible start is J2006 for truth coverage to J2026)
        keys = [f"j{y}" for y in range(2000, 2007)]
        results = run_rolling_shootout(
            window_start_keys=keys,
            prediction_horizon_yr=20.0,
            sample_interval_yr=1.0,
            predictors=selected,
        )
        all_results.extend(results)
