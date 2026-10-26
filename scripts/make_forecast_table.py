"""Build the dense forecast table that Mentat's body_position serves for future dates: our own N-body integration of the Sun, Mercury, Venus, Earth, Moon, Mars and the giants and Pluto (as system barycentres)
from DE440's state at the start epoch, with the layers of the `restated_zon` predictor (asteroids, ring, Kuiper-belt objects, the Moon's C22, measured Jupiter and Saturn zonals), stored every step so a query is a cubic
Hermite interpolation of positions and velocities.

This is a PREDICTION made on the date it is run, from JPL's own start state: it is not a measurement and not a claim to beat JPL. The file is sealed: its sha256, the creation time, the git commit, the predictor and the start
epoch go into fixtures/forecast_manifest.json (small, kept in git), and the table itself goes to the data volume (D:/datasets/forecast/, mirrored to skippy-core by SkippyTel/tools/sync_datasets.sh). A new run writes a NEW dated file;
nothing is overwritten. The part of the table that has already happened is checked against DE440 and the error is recorded in the manifest.

usage: python scripts/make_forecast_table.py [--start-jd 2461041.5] [--years 30] [--dt-days 0.25] [--predictor restated_zon] [--out-dir D:/datasets/forecast]
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-jd", type=float, default=J2026_JD)
    ap.add_argument("--years", type=float, default=30.0)
    ap.add_argument("--dt-days", type=float, default=0.25)
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
    t0 = time.time()
    system, order = rm.build_system(kernel, a.start_jd, a.predictor, dt_days=a.dt_days, horizon_days=jd1 - a.start_jd + 10.0, fixture_key=a.fixture_key)
    print(f"built {len(order)} bodies with layers of {a.predictor} in {time.time() - t0:.0f} s; integrating {a.years} yr at {a.dt_days} d", flush=True)
    d = rm.propagate_dense(system, a.start_jd, jd1, a.dt_days, every=1)
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
    stamp = datetime.datetime.now(datetime.timezone.utc)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    name = f"forecast_{a.predictor}_{stamp:%Y-%m-%d}_{a.years:g}yr.npz"
    path = a.out_dir / name
    np.savez_compressed(path, jd=d["jd"], pos_km=d["pos_km"], vel_km_s=d["vel_km_s"], names=np.array(order))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    entry = {"file": name, "sha256": sha, "bytes": path.stat().st_size, "created_utc": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"), "git_commit": _git_commit(), "predictor": a.predictor,
             "start_jd_tdb": a.start_jd, "end_jd_tdb": float(d["jd"][-1]), "dt_days": a.dt_days, "bodies": order, "frame": "ICRF, solar-system barycentre, km and km/s",
             "start_state": "JPL DE440s kernel state at the start epoch (a propagation from JPL's state, not an independent measurement)",
             "reference_points": "Sun, Mercury, Venus, Earth, Moon: body centres; Mars, Jupiter, Saturn, Uranus, Neptune, Pluto: planet-system barycentres",
             "checked_against_de440_at_jd": check_jd, "heliocentric_error_km_at_check": check, "wall_seconds": time.time() - t0}
    doc = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"tables": []}
    doc["tables"].append(entry)
    MANIFEST.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"wrote {path} ({entry['bytes'] / 1e6:.1f} MB, sha256 {sha[:16]}...)")
    if check:
        print("check against DE440 at JD", check_jd, {k: round(v, 3) for k, v in check.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
