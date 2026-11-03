"""Driver for the daily MPC live scoreboard (misc/CANDIDATE_LEDGER.md C27): runs scripts/mpc_live_score.py with retries, then scripts/mpc_live_report.py, and logs every exit code.

The scorer resumes on its own (a day's MPC files and Horizons responses are kept, the per-body watermark advances only after a row is written), so a retry only redoes what failed.
Writes D:/datasets/mpc_live/<date>/driver.log, and D:/datasets/mpc_live/<date>/DONE on a clean finish; the last line printed is 'finished: ...' (the watchdog's success marker)
or 'GAVE UP ...'. Scheduled by the Windows task 'sigma-ground mpc live daily' through scripts/run_mpc_live.bat; attach scripts/jobwatch.py when running it from a session.

usage: python scripts/mpc_live_driver.py [--date YYYY-MM-DD] [--bodies 433 1] [--attempts 3]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date as _date, datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
ROOT = Path("D:/datasets/mpc_live")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=_date.today().isoformat())
    ap.add_argument("--bodies", nargs="+", type=int, default=None)
    ap.add_argument("--attempts", type=int, default=3)
    a = ap.parse_args()
    day = ROOT / a.date
    day.mkdir(parents=True, exist_ok=True)
    logf = open(day / "driver.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    def run(name, cmd, attempts):
        for k in range(1, attempts + 1):
            t0 = time.time()
            log(f"{name} attempt {k}/{attempts}: {' '.join(cmd)}")
            with open(day / f"{name}.out", "a", encoding="utf-8") as out:
                r = subprocess.run(cmd, cwd=_ROOT, stdout=out, stderr=subprocess.STDOUT)
            log(f"{name} attempt {k}/{attempts}: exit code {r.returncode} after {time.time() - t0:.0f} s")
            if r.returncode == 0:
                return 0
            time.sleep(min(600, 60 * k))
        return r.returncode

    py = sys.executable
    score_cmd = [py, "-X", "utf8", str(_ROOT / "scripts" / "mpc_live_score.py"), "--date", a.date] + (["--bodies"] + [str(b) for b in a.bodies] if a.bodies else [])
    rc = run("score", score_cmd, a.attempts)
    rc2 = run("report", [py, "-X", "utf8", str(_ROOT / "scripts" / "mpc_live_report.py")], 2)
    summary = {"date": a.date, "score_exit": rc, "report_exit": rc2, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if rc == 0 and rc2 == 0:
        (day / "DONE").write_text(json.dumps(summary), encoding="utf-8")
        log(f"finished: score ok, report ok ({a.date})")
        return 0
    log(f"GAVE UP: score exit {rc}, report exit {rc2} ({a.date}); the scorer's own log is {day / 'score.log'}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
