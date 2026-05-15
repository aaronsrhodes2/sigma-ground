"""Long-horizon prediction sweep with monthly cadence sampling.

The CNEOS Sentry / ESA NEOCC standard for planetary-defense impact monitoring
is **100 years**. This script implements a single rolling-window pass at that
horizon, sampled at monthly intervals (1200 samples total per integration).

The DE440 snapshot fixture only covers J2000–J2026 (annual). For truth data
beyond J2026 -- AND at monthly cadence rather than annual -- we tap the
jplephem DE440 kernel directly (cached at ~/.materia/ephemeris/de440s.bsp,
coverage 1849-12 → 2150-01, 300 years).

Truth coverage caveat: de440s.bsp carries only the 11 major bodies + Moon
(NAIF codes 1-10, 199, 299, 301, 399). Mars's moons, Jupiter's Galileans,
Saturn's / Uranus's / Neptune's moons, and Pluto's moons are NOT in this
kernel. We still INTEGRATE them (they perturb Jupiter, Saturn, etc.) but
we drop them from truth comparison since we have no reference.

Bodies with truth comparison enabled:
  Mercury, Venus, Earth, Moon, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto

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
import math
import time
from pathlib import Path

import numpy as np

from sigma_ground.field.constants import G as _G, L_SUN_W as _L_SUN_W
from sigma_ground.field.interface.nbody import (
    CelestialBody, NBodySystem, PhysicsToggles,
)
from sigma_ground.field.interface.rolling_shootout import (
    _build_bodies_at_snapshot, _load_de440, AU_KM, KM_TO_M, DAY_S, YR_DAYS,
    Predictor, PREDICTORS,
)
from sigma_ground.field.interface.adapters._jplephem_bridge import (
    _NAIF_CODES, _load_kernel,
)


# -- Truth-comparison-eligible bodies (in de440s.bsp coverage) -----------
# Pluto here is via NAIF code 9 (Pluto barycenter); good enough at the
# 100y horizon for impact-monitoring precision (~1000 km accuracy).
_TRUTH_BODIES = [
    "Mercury", "Venus", "Earth", "Moon", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto",
]


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
    """Heliocentric (Sun-relative) truth position at JD via jplephem."""
    if sun_pos_km is None:
        sun_pos_km = _query_jplephem_ssb_km(kernel, "Sun", jd)
        if sun_pos_km is None:
            return None
    body_pos = _query_jplephem_ssb_km(kernel, body_name, jd)
    if body_pos is None:
        return None
    return body_pos - sun_pos_km


def _build_sample_jds(start_jd: float, horizon_yr: float,
                      month_step: int = 1) -> list[float]:
    """Generate monthly sample JDs from start_jd to start_jd + horizon_yr.

    month_step=1 → every month; month_step=12 → annual; etc.
    Returns [start_jd, start_jd + month, start_jd + 2*month, ...].
    """
    month_days = YR_DAYS / 12.0   # 30.4375
    n_months   = int(round(horizon_yr * 12)) + 1
    return [start_jd + k * month_days for k in range(0, n_months, month_step)]


def _integrate_with_samples(
    predictor:  Predictor,
    bodies:     list[CelestialBody],
    body_names: list[str],
    sun_idx:    int,
    sample_jds: list[float],
) -> dict[str, np.ndarray]:
    """Forward-integrate with arbitrary sample schedule (not snapshot-aligned).

    Same integration logic as rolling_shootout._integrate_nbody but accepts
    monthly cadence sample times instead of relying on annual-snapshot JDs.
    """
    if len(sample_jds) < 1:
        raise ValueError("need at least one sample JD")
    start_jd = sample_jds[0]
    dt_s = predictor.dt_days * DAY_S

    system = NBodySystem(
        bodies,
        toggles=predictor.toggles,
        solar_luminosity_W=_L_SUN_W if predictor.toggles.srp else 0.0,
    )
    step_fn = (system.forest_ruth_step if predictor.integrator == "fr4"
                else system.step)

    samples: dict[str, list[np.ndarray]] = {n: [] for n in body_names}

    def _snapshot() -> None:
        sun_pos = system.bodies[sun_idx].position_m
        for i, n in enumerate(body_names):
            hel_km = (system.bodies[i].position_m - sun_pos) / KM_TO_M
            samples[n].append(hel_km.copy())

    _snapshot()

    current_jd = start_jd
    for target_jd in sample_jds[1:]:
        dt_total_days = target_jd - current_jd
        n_steps = max(1, int(round(dt_total_days / predictor.dt_days)))
        exact_dt_s = (dt_total_days * DAY_S) / n_steps
        for _ in range(n_steps):
            step_fn(exact_dt_s)
        current_jd = target_jd
        _snapshot()

    return {n: np.array(samples[n]) for n in body_names}


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
        "metadata": {window_start_key, horizon_yr, month_step, predictors, ...},
        "samples": [{predictor, body, month, sample_jd, error_au, predicted_helio_km, truth_helio_km}, ...]
      }
    """
    de440  = _load_de440()
    kernel = _load_kernel("de440s")
    start_jd = de440["snapshots"][window_start_key]["epoch"]["jd_tdb"]

    sample_jds = _build_sample_jds(start_jd, horizon_yr, month_step)
    if verbose:
        print(f"  window:       {window_start_key} (JD={start_jd:.1f})")
        print(f"  horizon:      {horizon_yr}y")
        print(f"  month_step:   {month_step}")
        print(f"  samples:      {len(sample_jds)} (every {month_step} month(s))")
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

        # Build bodies from the window-start snapshot
        bodies, body_names, sun_idx = _build_bodies_at_snapshot(
            window_start_key, de440, predictor,
        )

        # Run integration with monthly samples
        preds = _integrate_with_samples(
            predictor, bodies, body_names, sun_idx, sample_jds,
        )
        if verbose:
            print(f"    integration: {time.time() - t0:.1f}s")

        # Compare to truth at each (sample, body)
        for body_name in _TRUTH_BODIES:
            if body_name not in preds:
                continue
            pred_arr = preds[body_name]
            for s_idx, sjd in enumerate(sample_jds):
                truth_km = truth_helio_km_cache.get((sjd, body_name))
                pred_km  = pred_arr[s_idx]
                err_au   = None
                if truth_km is not None:
                    err_au = float(np.linalg.norm(pred_km - truth_km)) / AU_KM
                out_samples.append({
                    "predictor": predictor.name,
                    "body":      body_name,
                    "month":     s_idx * month_step,
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
            "predictors":         [p.name for p in predictors],
            "elapsed_s":          round(time.time() - t0_global, 1),
        },
        "samples": out_samples,
    }
    output_path.write_text(json.dumps(payload, separators=(",", ":")))
    if verbose:
        print(f"\n  Total: {time.time() - t0_global:.1f}s")
        print(f"  Written {len(out_samples):,} samples -> {output_path.name}")


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
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent.parent
                                / "sigma_ground" / "field" / "interface" / "fixtures"
                                / "long_horizon_sweep.json",
                        help="output JSON path")
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
