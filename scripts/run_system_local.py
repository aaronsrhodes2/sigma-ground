"""Driver for scripts/system_local_moons.py: one subprocess per (system, window), retried, a log with ' done ' lines and a 'finished:' marker for jobwatch.

usage: python scripts/run_system_local.py --outdir D:/datasets/sigma_runs/sysloc --systems Jupiter Saturn Uranus Neptune Pluto --windows j2010 j2015 --workers 6 [--steps-per-orbit 300] [--zonals] [--reduced] [--with-planets] [--fixture PATH] [--tag NAME]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--windows", nargs="+", default=["j2010", "j2015"])
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--steps-per-orbit", type=float, default=300.0)
    ap.add_argument("--zonals", action="store_true")
    ap.add_argument("--reduced", action="store_true")
    ap.add_argument("--with-planets", action="store_true")
    ap.add_argument("--fixture", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=2)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    log = a.outdir / "sysloc.log"
    t0 = time.time()
    jobs = [(s, w) for s in a.systems for w in a.windows]
    log.write_text(f"{len(jobs)} jobs, {a.workers} workers\n")

    def one(job):
        s, w = job
        out = a.outdir / f"{s}_{w}_spo{int(a.steps_per_orbit)}.json"
        cmd = [sys.executable, "-X", "utf8", str(HERE / "system_local_moons.py"), "--system", s, "--windows", w, "--years", str(a.years),
               "--steps-per-orbit", str(a.steps_per_orbit), "--out", str(out)] + (["--zonals"] if a.zonals else []) + (["--reduced"] if a.reduced else [])               + (["--with-planets"] if a.with_planets else []) + (["--fixture", str(a.fixture)] if a.fixture else [])
        for k in range(a.retries + 1):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and out.exists():
                with log.open("a") as f:
                    f.write(f"[{time.time() - t0:7.0f}s] done {s}/{w}\n")
                return True
            with log.open("a") as f:
                f.write(f"[{time.time() - t0:7.0f}s] FAILED {s}/{w} (attempt {k + 1}), exit {r.returncode}, retrying: {r.stderr.strip()[-300:]}\n")
        with log.open("a") as f:
            f.write(f"GAVE UP {s}/{w}\n")
        return False

    with ThreadPoolExecutor(a.workers) as ex:
        ok = list(ex.map(one, jobs))
    with log.open("a") as f:
        f.write(f"finished: {sum(ok)} ok, {len(ok) - sum(ok)} failed, {time.time() - t0:.0f}s wall\n")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
