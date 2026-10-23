"""Driver for scripts/slow_moon_diagnosis.py: one subprocess per (window, variant), retried, a log with ' done ' lines and a 'finished:' marker for jobwatch.

usage: python scripts/run_slow_moon_diag.py --moon Nereid --planet Neptune --extra Triton --outdir D:/datasets/sigma_runs/diag_nereid --workers 2 [--variants ...] [--windows ...]
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
    ap.add_argument("--moon", required=True)
    ap.add_argument("--planet", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--windows", nargs="+", default=["j2010", "j2015"])
    ap.add_argument("--variants", nargs="+", default=["base", "no_zonals", "no_tide", "no_srp", "no_triton", "coarse", "fine"])
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--retries", type=int, default=2)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    log = a.outdir / "diag.log"
    t0 = time.time()
    jobs = [(w, v) for w in a.windows for v in a.variants]
    log.write_text(f"{len(jobs)} jobs, {a.workers} workers\n")

    def one(job):
        w, v = job
        out = a.outdir / f"{a.moon}_{w}_{v}.json"
        cmd = [sys.executable, "-X", "utf8", str(HERE / "slow_moon_diagnosis.py"), "--moon", a.moon, "--planet", a.planet, "--windows", w, "--variants", v, "--years", str(a.years), "--out", str(out)]
        if a.extra:
            cmd += ["--extra", *a.extra]
        for k in range(a.retries + 1):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and out.exists():
                with log.open("a") as f:
                    f.write(f"[{time.time() - t0:7.0f}s] done {w}/{v}: {r.stdout.strip().splitlines()[-1]}\n")
                return True
            with log.open("a") as f:
                f.write(f"[{time.time() - t0:7.0f}s] FAILED {w}/{v} (attempt {k + 1}), exit {r.returncode}, retrying: {r.stderr.strip()[-300:]}\n")
        with log.open("a") as f:
            f.write(f"GAVE UP {w}/{v}\n")
        return False

    with ThreadPoolExecutor(a.workers) as ex:
        ok = list(ex.map(one, jobs))
    with log.open("a") as f:
        f.write(f"finished: {sum(ok)} ok, {len(ok) - sum(ok)} failed, {time.time() - t0:.0f}s wall\n")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
