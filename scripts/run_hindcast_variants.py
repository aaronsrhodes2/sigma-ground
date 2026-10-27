"""Run scripts/make_prediction_error_table.py for one predictor, one process per start year, with retries and logged exit codes (the long-job rule: a driver that retries and logs, with jobwatch beside it).

Each start writes <outdir>/<predictor>_<year>.json (the same table format as fixtures/prediction_error_table.json, one start) and <outdir>/hind_<predictor>_<year>.log. The driver log <outdir>/driver.log gets one line
per attempt and a last line `finished: N ok, M failed`; jobwatch's success regex is that line.

usage: python scripts/run_hindcast_variants.py --predictor restated_hdr --starts 2000 2005 2010 2015 --outdir D:/datasets/sigma_runs/hdr_hindcast [--max-lead 20] [--tries 3] [--dt-days 0.025]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictor", required=True)
    ap.add_argument("--starts", nargs="+", type=int, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--max-lead", type=int, default=20)
    ap.add_argument("--dt-days", type=float, default=0.025)
    ap.add_argument("--tries", type=int, default=3)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    dlog = a.outdir / "driver.log"

    def log(msg: str) -> None:
        with dlog.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

    def cmd(y: int) -> list[str]:
        return [sys.executable, "-X", "utf8", str(_ROOT / "scripts/make_prediction_error_table.py"), "--starts", str(y), "--max-lead", str(a.max_lead),
                "--predictor", a.predictor, "--dt-days", str(a.dt_days), "--out", str(a.outdir / f"{a.predictor}_{y}.json")]

    running: dict[int, subprocess.Popen] = {}
    tries = {y: 0 for y in a.starts}
    done: dict[int, int] = {}
    log(f"driver start: predictor {a.predictor}, starts {a.starts}, max-lead {a.max_lead}, dt {a.dt_days}")
    while len(done) < len(a.starts):
        for y in a.starts:
            if y in done or y in running:
                continue
            if tries[y] >= a.tries:
                done[y] = -1
                log(f"GAVE UP start {y} after {a.tries} tries")
                continue
            tries[y] += 1
            fh = (a.outdir / f"hind_{a.predictor}_{y}.log").open("a", encoding="utf-8")
            running[y] = subprocess.Popen(cmd(y), stdout=fh, stderr=subprocess.STDOUT, cwd=str(_ROOT))
            log(f"start {y} try {tries[y]} pid {running[y].pid}")
        time.sleep(15)
        for y, p in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            del running[y]
            log(f"start {y} try {tries[y]} exit {rc}")
            if rc == 0:
                done[y] = 0
            elif tries[y] < a.tries:
                log(f"FAILED start {y}, retrying")
    ok = sum(1 for v in done.values() if v == 0)
    log(f"finished: {ok} ok, {len(done) - ok} failed")
    return 0 if ok == len(a.starts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
