"""Build the dense forecast table that Mentat's body_position serves for future dates: our own N-body integration of the Sun, Mercury, Venus, Earth, Moon, Mars and the giants and Pluto (as system barycentres)
from DE440's state at the start epoch, with the layers of the `restated_zon` predictor (asteroids, ring, Kuiper-belt objects, the Moon's C22, measured Jupiter and Saturn zonals), stored every step so a query is a cubic
Hermite interpolation of positions and velocities.

This is a PREDICTION made on the date it is run, from JPL's own start state: it is not a measurement and not a claim to beat JPL. The file is sealed: its sha256, the creation time, the git commit, the predictor and the start
epoch go into fixtures/forecast_manifest.json (small, kept in git), and the table itself goes to the data volume (D:/datasets/forecast/, mirrored to skippy-core by SkippyTel/tools/sync_datasets.sh). A new run writes a NEW dated file;
nothing is overwritten. The part of the table that has already happened is checked against DE440 and the error is recorded in the manifest.

usage: python scripts/make_forecast_table.py [--start-jd 2461041.5] [--years 30] [--dt-days 0.025] [--every 10] [--predictor restated_zon] [--out-dir D:/datasets/forecast]

The step matters: 0.25 d is far too coarse for Mercury and the Moon (a first build at 0.25 d was 102 km and 259 km from DE440 after 0.8 yr and was rejected). The step is 0.025 d, the shoot-out's, and the table stores every --every steps
(0.25 d at the defaults) for Hermite interpolation. A build is refused if its step is above 0.05 d, and a table whose elapsed part is more than 2 km from DE440 is not sealed.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
MANIFEST = _ROOT / "sigma_ground/field/interface/fixtures/forecast_manifest.json"
J2026_JD = 2461041.5                                             # 2026-01-01 00:00 TDB, the last epoch of the DE440 fixture


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


MAX_STEP_DAYS = 0.05                  # the convergence study (MOON_STEP_CONVERGENCE.md): the Moon is converged at 0.025 d; 0.25 d is wrong by hundreds of km
MAX_CHECK_KM = 2.0                    # a table whose elapsed part is further than this from DE440 (any body) is a failed build: the shoot-out's +1 yr medians are 0.005 to 0.1 km


def refusal(dt_days: float, check_km: dict) -> str | None:
    """Why this build must not be sealed, or None."""
    if dt_days > MAX_STEP_DAYS:
        return f"step {dt_days} d is above {MAX_STEP_DAYS} d: too coarse for Mercury and the Moon"
    bad = {n: e for n, e in check_km.items() if e > MAX_CHECK_KM}
    if bad:
        return "the elapsed part of the table is further than %.0f km from DE440 for %s" % (MAX_CHECK_KM, ", ".join(f"{n} ({e:.1f} km)" for n, e in bad.items()))
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-jd", type=float, default=J2026_JD)
    ap.add_argument("--years", type=float, default=30.0)
    ap.add_argument("--dt-days", type=float, default=0.025)
    ap.add_argument("--every", type=int, default=10, help="store every N integration steps")
    ap.add_argument("--force", action="store_true", help="seal a table although the step guard or the check against DE440 says no (never for a served table)")
    ap.add_argument("--predictor", default="restated_zon")
    ap.add_argument("--fixture-key", default="j2026", help="snapshot the masses and body parameters are read from (not part of the initial state)")
    ap.add_argument("--out-dir", type=Path, default=Path("D:/datasets/forecast"))
    a = ap.parse_args()
    from jplephem.spk import SPK
    from sigma_ground.field import datasets
    from sigma_ground.field.interface import reduced_model as rm
    kfile = datasets.de440_kernel()
    if kfile is None:
        raise SystemExit("the DE440s kernel is not installed here (SIGMA_DATASETS / D:/datasets/jpl/de440s.bsp)")
    kernel = SPK.open(str(kfile))
    jd1 = a.start_jd + a.years * 365.25
    if jd1 > kernel[0, 3].end_jd:
        raise SystemExit(f"the kernel ends at JD {kernel[0, 3].end_jd}; asked for {jd1}")
    if a.dt_days > MAX_STEP_DAYS and not a.force:
        raise SystemExit(refusal(a.dt_days, {}))
    fingerprint = rm.model_fingerprint(a.predictor)                      # of the physics as it is when the build STARTS
    t0 = time.time()
    system, order = rm.build_system(kernel, a.start_jd, a.predictor, dt_days=a.dt_days, horizon_days=jd1 - a.start_jd + 10.0, fixture_key=a.fixture_key)
    print(f"built {len(order)} bodies with layers of {a.predictor} in {time.time() - t0:.0f} s; integrating {a.years} yr at {a.dt_days} d", flush=True)
    d = rm.propagate_dense(system, a.start_jd, jd1, a.dt_days, every=a.every)
    print(f"integrated {len(d['jd'])} samples in {time.time() - t0:.0f} s", flush=True)
    # check the part that has already happened against DE440 (heliocentric position, km)
    now_jd = datetime.datetime.now(datetime.timezone.utc).timestamp() / 86400.0 + 2440587.5
    ok = d["jd"] <= min(now_jd, jd1)
    check = {}
    if ok.sum() > 2:
        idx = np.where(ok)[0][-1]
        sun_i = order.index("Sun")
        for i, n in enumerate(order):
            if n == "Sun":
                continue
            truth = rm.kernel_state(kernel, n, d["jd"][idx])[0] - rm.kernel_state(kernel, "Sun", d["jd"][idx])[0]
            check[n] = float(np.linalg.norm((d["pos_km"][idx, i] - d["pos_km"][idx, sun_i]) - truth))
        check_jd = float(d["jd"][idx])
    else:
        check_jd = None
    why = refusal(a.dt_days, check)
    if why and not a.force:
        raise SystemExit("NOT SEALED: " + why)
    stamp = datetime.datetime.now(datetime.timezone.utc)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    name = f"forecast_{a.predictor}_{stamp:%Y-%m-%d}_{a.years:g}yr.npz"
    path = a.out_dir / name
    np.savez_compressed(path, jd=d["jd"], pos_km=d["pos_km"], vel_km_s=d["vel_km_s"], names=np.array(order))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    entry = {"file": name, "sha256": sha, "bytes": path.stat().st_size, "created_utc": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"), "git_commit": _git_commit(), "predictor": a.predictor,
             "start_jd_tdb": a.start_jd, "end_jd_tdb": float(d["jd"][-1]), "dt_days": a.dt_days, "stored_every_steps": a.every, "bodies": order, "frame": "ICRF, solar-system barycentre, km and km/s",
             "start_state": "JPL DE440s kernel state at the start epoch (a propagation from JPL's state, not an independent measurement)",
             "reference_points": "Sun, Mercury, Venus, Earth, Moon: body centres; Mars, Jupiter, Saturn, Uranus, Neptune, Pluto: planet-system barycentres",
             "checked_against_de440_at_jd": check_jd, "heliocentric_error_km_at_check": check, "model_fingerprint": fingerprint, "wall_seconds": time.time() - t0}
    doc = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"tables": []}
    doc["tables"].append(entry)
    MANIFEST.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"wrote {path} ({entry['bytes'] / 1e6:.1f} MB, sha256 {sha[:16]}...)")
    if check:
        print("check against DE440 at JD", check_jd, {k: round(v, 3) for k, v in check.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
