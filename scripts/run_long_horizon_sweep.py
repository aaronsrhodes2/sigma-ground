"""Long-horizon prediction sweep with monthly cadence sampling.

The CNEOS Sentry / ESA NEOCC standard for planetary-defense impact monitoring
is **100 years**. This script implements a single rolling-window pass at that
horizon, sampled at monthly intervals (1200 samples total per integration).

The DE440 snapshot fixture only covers J2000–J2026 (annual). For truth data
beyond J2026 -- AND at monthly cadence rather than annual -- we tap the
jplephem DE440 kernel directly (D:/datasets/jpl/de440s.bsp, the location
adapters._jplephem_bridge._load_kernel searches; coverage 1849-12 → 2150-01).

Truth coverage caveat: de440s.bsp carries only the system barycentres
(NAIF 1-9), the Sun (10), and the Mercury, Venus, Earth and Moon centres
(199, 299, 399, 301). Mars's moons, Jupiter's Galileans, Saturn's / Uranus's /
Neptune's moons, and Pluto's moons are NOT in this kernel. We still INTEGRATE
them (they perturb Jupiter, Saturn, etc.) but drop them from truth comparison.

Like is compared with like. For Mercury, Venus, Earth and the Moon the truth
is the body centre. For Mars through Pluto the kernel only knows the SYSTEM
BARYCENTRE, so the prediction compared with it is the mass-weighted
barycentre of the planet and every integrated body anchored to it (the
fixture's gravitational_anchor). Comparing the kernel barycentre with the
integrated planet centre, as this script used to, put a floor of 174 km
(Jupiter), 276 km (Saturn), 527 km (Uranus), 74 km (Neptune) and 2131 km
(Pluto) under every error at t = 0 (window j2015). With barycentres the t = 0
floor (truth_floor_km) is 0.0 km for Jupiter, 0.05 km for Neptune and 0.1 km
for Pluto. Two floors remain, both in the fixture rather than the method:
Saturn 11.7 km (the fixture has no Iapetus, Hyperion or Phoebe), and Uranus
350-1160 km, growing from J2000 to J2026, because the fixture's Uranus-system
states disagree with the de440s Uranus barycentre by that much (the moons
can only move the barycentre ~35 km).

Bodies with truth comparison enabled:
  Mercury, Venus, Earth, Moon, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto

Integration is rolling_shootout._integrate_nbody itself, so every predictor
runs exactly as it does in the shoot-out: Forest-Ruth for 'fr4', RESPA with
the fixture's parent-attractor indices for 'respa', velocity Verlet for
'verlet'.

Compute estimate:
  100y × 365.25d/y / 0.1d_per_step = 365,250 force evals per predictor.
  At ~30 bodies, ~5e-4 s per step → ~180 seconds per predictor per window.
  Per-window cost dominated by integration; truth-comparison overhead is
  negligible (1200 fast kernel queries).

Usage:
    python scripts/run_long_horizon_sweep.py --predictors over_physics_finedt
    python scripts/run_long_horizon_sweep.py --predictors over_physics_finedt finedt_all
    python scripts/run_long_horizon_sweep.py --window j2025 --horizon-yr 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sigma_ground.field.interface.nbody import CelestialBody, PhysicsToggles
from sigma_ground.field.interface.rolling_shootout import (
    _build_bodies_at_snapshot, _integrate_nbody, _load_de440, AU_KM, YR_DAYS,
    Predictor, PREDICTORS,
)
from sigma_ground.field.interface.adapters._jplephem_bridge import (
    _NAIF_CODES, _load_kernel,
)


# -- Truth-comparison-eligible bodies (in de440s.bsp coverage) -----------
_TRUTH_BODIES = [
    "Mercury", "Venus", "Earth", "Moon", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto",
]
# Bodies whose de440s truth is the system barycentre (NAIF 4-9); the
# prediction for them is the barycentre of the planet and its moons.
_BARYCENTRE_TRUTH = {"Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"}


def _query_jplephem_ssb_km(kernel, body_name: str, jd: float) -> np.ndarray | None:
    """SSB-equatorial position (km) at the given JD via jplephem, or None.

    Note: DE440 SPK doesn't have direct SSB→Earth (or SSB→Moon) segments;
    instead they chain through the Earth-Moon barycenter (target 3):
      Earth SSB  = kernel[0,3] + kernel[3,399]
      Moon  SSB  = kernel[0,3] + kernel[3,301]
    We do the chaining here for Earth/Moon; other bodies route directly.
    """
    code = _NAIF_CODES.get(body_name)
    if code is None:
        return None
    try:
        if body_name in ("Earth", "Moon"):
            emb_ssb, _    = kernel[0, 3].compute_and_differentiate(jd)
            body_emb, _   = kernel[3, code].compute_and_differentiate(jd)
            return np.array(emb_ssb) + np.array(body_emb)
        pos, _ = kernel[0, code].compute_and_differentiate(jd)
        return np.array(pos)
    except Exception:
        return None


def _heliocentric_truth_km(kernel, body_name: str, jd: float,
                            sun_pos_km: np.ndarray | None = None) -> np.ndarray | None:
    """Heliocentric (Sun-relative) truth position at JD via jplephem.

    For the bodies in _BARYCENTRE_TRUTH this is the system barycentre.
    """
    if sun_pos_km is None:
        sun_pos_km = _query_jplephem_ssb_km(kernel, "Sun", jd)
        if sun_pos_km is None:
            return None
    body_pos = _query_jplephem_ssb_km(kernel, body_name, jd)
    if body_pos is None:
        return None
    return body_pos - sun_pos_km


def _system_members(de440: dict, window_start_key: str,
                    body_names: list[str]) -> dict[str, list[int]]:
    """Index lists of the bodies that make up each truth point.

    A barycentre-truth planet takes itself plus every body whose fixture
    gravitational_anchor is that planet; every other truth body is itself.
    """
    anchors = {b["name"]: b.get("gravitational_anchor")
               for b in de440["snapshots"][window_start_key]["bodies"]}
    members: dict[str, list[int]] = {}
    for body in _TRUTH_BODIES:
        if body not in body_names:
            continue
        idx = [body_names.index(body)]
        if body in _BARYCENTRE_TRUTH:
            idx += [i for i, n in enumerate(body_names) if anchors.get(n) == body]
        members[body] = idx
    return members


def _model_points_km(preds: dict[str, np.ndarray], bodies: list[CelestialBody],
                     body_names: list[str],
                     members: dict[str, list[int]]) -> dict[str, np.ndarray]:
    """Heliocentric predictions of each truth point (centre or barycentre)."""
    out: dict[str, np.ndarray] = {}
    for body, idx in members.items():
        if len(idx) == 1:
            out[body] = preds[body]
            continue
        m = np.array([bodies[i].mass_kg for i in idx])
        stack = np.stack([preds[body_names[i]] for i in idx])   # (k, n_samples, 3)
        out[body] = np.tensordot(m / m.sum(), stack, axes=1)
    return out


def truth_floor_km(window_start_key: str, de440: dict | None = None,
                   kernel=None) -> dict[str, float]:
    """|truth - prediction point| at t = 0, built from the fixture states.

    This is the error every predictor carries before it takes a step: the
    gap between the de440s point and the same point assembled from the
    DE440 fixture (whose moons list is incomplete, e.g. no Iapetus).
    """
    de440 = de440 or _load_de440()
    kernel = kernel or _load_kernel("de440s")
    pred0 = Predictor("t0", PhysicsToggles())
    bodies, names, sun_idx, _ = _build_bodies_at_snapshot(window_start_key, de440, pred0)
    sun = bodies[sun_idx].position_m
    preds = {n: ((b.position_m - sun) / 1000.0)[None, :] for n, b in zip(names, bodies)}
    points = _model_points_km(preds, bodies, names,
                              _system_members(de440, window_start_key, names))
    jd = de440["snapshots"][window_start_key]["epoch"]["jd_tdb"]
    out = {}
    for body, p in points.items():
        t = _heliocentric_truth_km(kernel, body, jd)
        out[body] = float(np.linalg.norm(p[0] - t)) if t is not None else float("nan")
    return out


def _build_sample_jds(start_jd: float, horizon_yr: float,
                      month_step: int = 1) -> list[float]:
    """Sample JDs every month_step months from start_jd, always ending at the
    horizon (start_jd + round(12 * horizon_yr) months).

    month_step=1 → every month; month_step=12 → annual; etc.
    """
    month_days = YR_DAYS / 12.0   # 30.4375
    n_months   = int(round(horizon_yr * 12))
    months = list(range(0, n_months + 1, month_step))
    if months[-1] != n_months:
        months.append(n_months)
    return [start_jd + k * month_days for k in months]


def _integrate_with_samples(
    predictor:  Predictor,
    bodies:     list[CelestialBody],
    body_names: list[str],
    sun_idx:    int,
    sample_jds: list[float],
    parent_attractor_indices: list[int | None] | None = None,
) -> dict[str, np.ndarray]:
    """Forward-integrate with an arbitrary sample schedule.

    Delegates to rolling_shootout._integrate_nbody, so the integrator
    dispatch (fr4 / respa with parent indices / verlet) is the shoot-out's.
    """
    if len(sample_jds) < 1:
        raise ValueError("need at least one sample JD")
    return _integrate_nbody(
        predictor, bodies, body_names, sun_idx, sample_jds[0], sample_jds,
        parent_attractor_indices=parent_attractor_indices,
    )


def run_long_horizon_sweep(
    window_start_key:  str,
    horizon_yr:        float,
    month_step:        int,
    predictors:        list[Predictor],
    output_path:       Path,
    verbose:           bool = True,
) -> None:
    """Run the long-horizon sweep and save results.

    Output JSON shape:
      {
        "metadata": {window_start_key, horizon_yr, month_step, predictors, truth_points, ...},
        "samples": [{predictor, body, month, sample_jd, error_au, predicted_helio_km, truth_helio_km}, ...]
      }
    For the bodies in _BARYCENTRE_TRUTH both predicted_helio_km and
    truth_helio_km are the system barycentre (metadata.truth_points).
    """
    de440  = _load_de440()
    kernel = _load_kernel("de440s")
    start_jd = de440["snapshots"][window_start_key]["epoch"]["jd_tdb"]
    month_days = YR_DAYS / 12.0

    sample_jds = _build_sample_jds(start_jd, horizon_yr, month_step)
    if verbose:
        print(f"  window:       {window_start_key} (JD={start_jd:.1f})")
        print(f"  horizon:      {horizon_yr}y")
        print(f"  month_step:   {month_step}")
        print(f"  samples:      {len(sample_jds)} (every {month_step} month(s), plus the horizon end)")
        print(f"  truth bodies: {', '.join(_TRUTH_BODIES)}")
        print(f"  predictors:   {[p.name for p in predictors]}")

    out_samples = []
    t0_global = time.time()

    # Pre-compute truth state for each sample JD (cached) -- avoids redundant
    # jplephem queries when we have multiple predictors.
    if verbose:
        print(f"\n  Pre-computing truth at {len(sample_jds)} sample JDs...")
    truth_helio_km_cache: dict[tuple[float, str], np.ndarray | None] = {}
    t0 = time.time()
    for sjd in sample_jds:
        sun_ssb = _query_jplephem_ssb_km(kernel, "Sun", sjd)
        for body in _TRUTH_BODIES:
            truth_helio_km_cache[(sjd, body)] = _heliocentric_truth_km(
                kernel, body, sjd, sun_pos_km=sun_ssb,
            )
    if verbose:
        print(f"    done in {time.time() - t0:.1f}s")

    for pred_idx, predictor in enumerate(predictors):
        if verbose:
            print(f"\n  Predictor {pred_idx+1}/{len(predictors)}: {predictor.name}")
        t0 = time.time()

        if predictor.use_kepler:
            raise ValueError(f"{predictor.name}: the Kepler predictor is not an "
                             f"integration; the long-horizon sweep runs n-body predictors only")

        # Build bodies from the window-start snapshot
        bodies, body_names, sun_idx, parent_idx = _build_bodies_at_snapshot(
            window_start_key, de440, predictor,
        )

        # Run integration with monthly samples
        preds = _integrate_with_samples(
            predictor, bodies, body_names, sun_idx, sample_jds,
            parent_attractor_indices=parent_idx,
        )
        points = _model_points_km(
            preds, bodies, body_names,
            _system_members(de440, window_start_key, body_names))
        if verbose:
            print(f"    integration: {time.time() - t0:.1f}s")

        # Compare to truth at each (sample, body)
        for body_name in _TRUTH_BODIES:
            if body_name not in points:
                continue
            pred_arr = points[body_name]
            for s_idx, sjd in enumerate(sample_jds):
                truth_km = truth_helio_km_cache.get((sjd, body_name))
                pred_km  = pred_arr[s_idx]
                err_au   = None
                if truth_km is not None:
                    err_au = float(np.linalg.norm(pred_km - truth_km)) / AU_KM
                out_samples.append({
                    "predictor": predictor.name,
                    "body":      body_name,
                    "month":     int(round((sjd - start_jd) / month_days)),
                    "sample_jd": float(sjd),
                    "predicted_helio_km": [float(x) for x in pred_km],
                    "truth_helio_km":     ([float(x) for x in truth_km]
                                            if truth_km is not None else None),
                    "error_au":  err_au,
                })

    payload = {
        "metadata": {
            "window_start_key":   window_start_key,
            "horizon_yr":         horizon_yr,
            "month_step":         month_step,
            "n_samples":          len(sample_jds),
            "truth_bodies":       _TRUTH_BODIES,
            "truth_points":       {b: ("system barycentre" if b in _BARYCENTRE_TRUTH
                                       else "body centre") for b in _TRUTH_BODIES},
            "truth_kernel":       "de440s",
            "predictors":         [p.name for p in predictors],
            "integrators":        {p.name: p.integrator for p in predictors},
            "elapsed_s":          round(time.time() - t0_global, 1),
        },
        "samples": out_samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")))
    if verbose:
        print(f"\n  Total: {time.time() - t0_global:.1f}s")
        print(f"  Written {len(out_samples):,} samples -> {output_path}")


# -- Default predictors for the sweep (built on over_physics_finedt) -----

_BASE_TOGGLES = PhysicsToggles(gr_1pn=True, srp=True, j2_zonal=True)

DEFAULT_LONG_HORIZON_PREDICTORS: list[Predictor] = [
    Predictor(
        name="finedt",
        toggles=_BASE_TOGGLES,
        integrator="fr4", dt_days=0.1,
        description="Newton + 1PN GR + SRP + J2 (baseline at dt=0.1d)",
    ),
]

# Default output (untracked); the committed fixture is long_horizon_smoke.json,
# regenerated only by passing it explicitly as --output.
_DEFAULT_OUTPUT = (Path(__file__).parent.parent / "sigma_ground" / "field"
                   / "interface" / "fixtures" / "long_horizon_sweep.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default="j2025",
                        help="DE440 snapshot key for window start (default: j2025)")
    parser.add_argument("--horizon-yr", type=float, default=100.0,
                        help="prediction horizon in years (default: 100)")
    parser.add_argument("--month-step", type=int, default=1,
                        help="sample every N months (default: 1)")
    parser.add_argument("--predictors", nargs="+", default=None,
                        help="predictor names from rolling_shootout.PREDICTORS, "
                             "or 'finedt' for the default")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT,
                        help=f"output JSON path (default: {_DEFAULT_OUTPUT})")
    args = parser.parse_args()

    if args.predictors:
        all_preds = {p.name: p for p in PREDICTORS}
        all_preds.update({p.name: p for p in DEFAULT_LONG_HORIZON_PREDICTORS})
        selected = [all_preds[n] for n in args.predictors if n in all_preds]
        missing  = [n for n in args.predictors if n not in all_preds]
        if missing:
            raise SystemExit(f"unknown predictors: {missing}; "
                              f"available: {list(all_preds.keys())}")
    else:
        selected = DEFAULT_LONG_HORIZON_PREDICTORS

    run_long_horizon_sweep(
        window_start_key=args.window,
        horizon_yr=args.horizon_yr,
        month_step=args.month_step,
        predictors=selected,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
