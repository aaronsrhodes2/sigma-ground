"""How wrong is a close-in moon's MEAN MOTION? The yearly fixture samples cannot show it directly (the phase wraps), but the wrapped phase error phi_k = angle(model, truth) at k = 1..5 yr is
phi_k = k * delta (mod 2 pi) when the mean motion is off by a constant, so one delta fits all five and gives the relative mean-motion error  dn/n = delta / (2 pi * orbits per year).

Runs the small system of scripts/slow_moon_diagnosis.py (Sun, planet, moon [, neighbours]) at steps_per_orbit steps per orbit, then prints phi_k, the radius ratio |model|/|truth| and the out-of-plane part.

usage: python scripts/phase_slope.py --moon Proteus --planet Neptune [--extra Triton] --window j2010 [--years 5] [--steps-per-orbit 300] [--zonals] [--fixture PATH] [--out FILE.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slow_moon_diagnosis as sd  # noqa: E402
from system_local_moons import periods_days  # noqa: E402

KEYS = ("x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s")


def wrap(x):
    return (np.asarray(x) + math.pi) % (2 * math.pi) - math.pi


def fit_delta(phi: np.ndarray) -> tuple[float, float]:
    """The per-year phase slope delta in (-pi, pi] that best explains phi_k = k delta (mod 2 pi), k = 1..len; and the rms residual (rad)."""
    ks = np.arange(1, len(phi) + 1)
    grid = np.linspace(-math.pi, math.pi, 200001)
    cost = (wrap(phi[None, :] - ks[None, :] * grid[:, None]) ** 2).sum(axis=1)
    i = int(np.argmin(cost))
    return float(grid[i]), float(math.sqrt(cost[i] / len(phi)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--moon", required=True)
    ap.add_argument("--planet", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--window", default="j2010")
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--steps-per-orbit", type=float, default=300.0)
    ap.add_argument("--variant", default="base", help="a variant of scripts/slow_moon_diagnosis.py (no_zonals, no_tide, no_srp, no_gr, newtonian, ...)")
    ap.add_argument("--origin-shift", type=float, default=0.0, help="test: shift the moon's initial state by s * (planet centre to the barycentre of its listed members), s = this value")
    ap.add_argument("--zonals", action="store_true")
    ap.add_argument("--fixture", type=Path, default=sd.FIXTURE)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    fix = json.loads(a.fixture.read_text())
    snap0 = fix["snapshots"][a.window]
    period = periods_days(snap0, a.planet, [a.moon])[a.moon]
    dt = period / a.steps_per_orbit
    dx = dv = None
    if a.origin_shift:
        by = {b["name"]: b for b in snap0["bodies"]}
        mem = [a.planet] + list(a.extra)
        gm = np.array([by[m]["gm_km3_s2"] for m in mem])
        st = np.array([[by[m]["state_vector"][c] for c in KEYS] for m in mem])
        w = (gm[:, None] * (st - st[0])).sum(0) / gm.sum()                      # barycentre minus planet centre, km and km/s
        dx, dv = a.origin_shift * w[:3] * 1e3, a.origin_shift * w[3:] * 1e3
        print("origin offset |w| = %.2f km, %.3f m/s" % (np.linalg.norm(w[:3]), np.linalg.norm(w[3:]) * 1e3), flush=True)
    res = sd.run(a.moon, a.planet, a.window, a.years, a.variant, fix, tuple(a.extra), vec=True, dt_days=dt, zonals=a.zonals, dx_m=dx, dv_m_s=dv)
    keys = sorted((k for k in fix["snapshots"] if k.startswith("j")), key=lambda k: int(k[1:]))
    k0 = keys.index(a.window)
    by0 = {b["name"]: b["state_vector"] for b in snap0["bodies"]}
    r0 = np.array([by0[a.moon][c] - by0[a.planet][c] for c in KEYS[:3]])
    v0 = np.array([by0[a.moon][c] - by0[a.planet][c] for c in KEYS[3:]])
    nrm = np.cross(r0, v0)
    nrm /= np.linalg.norm(nrm)
    phis, ratios, oop = [], [], []
    for k in range(1, a.years + 1):
        b = {x["name"]: x["state_vector"] for x in fix["snapshots"][keys[k0 + k]]["bodies"]}
        truth = np.array([b[a.moon][c] - b[a.planet][c] for c in KEYS[:3]])
        pred = res[k] + truth
        phis.append(math.atan2(float(np.cross(truth, pred) @ nrm), float(truth @ pred)))
        ratios.append(float(np.linalg.norm(pred) / np.linalg.norm(truth)))
        oop.append(float(pred @ nrm - truth @ nrm))
    delta, rms = fit_delta(np.array(phis))
    orbits_per_yr = 365.25 / period
    out = {"variant": a.variant, "moon": a.moon, "window": a.window, "period_d": period, "dt_d": dt, "zonals": a.zonals, "phi_rad": phis, "radius_ratio": ratios, "out_of_plane_km": oop,
           "delta_rad_per_yr": delta, "fit_rms_rad": rms, "dn_over_n": delta / (2 * math.pi * orbits_per_yr)}
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
