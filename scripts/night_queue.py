"""The night window (the Captain, 2026-11-04): long calculation runs are queued by day and run one at a time between 01:00 and 07:00 local time.

    python scripts/night_queue.py add --name NAME --est-minutes M [--resumable] [--cwd DIR] -- CMD ARGS...   # queue a job (CMD runs from the repo root unless --cwd)
    python scripts/night_queue.py list                                                                      # pending, running, and the last results
    python scripts/night_queue.py run [--start 01:00] [--end 07:00] [--now]                                 # drain the queue inside the window (the Windows task's entry)
    python scripts/night_queue.py drop NAME                                                                  # remove a pending job

Rules. One job at a time (the RAM rule). A job starts only if its estimate fits before the window's end; at the end a running job is stopped with its whole process
tree and marked CUT: a `--resumable` job (one that picks up where it left off, like the MPC scorer with its watermarks) goes back to the front of the queue for the next
night, any other goes to failed/ with its log. Each job's stdout and stderr go to logs/<name>.log; every start, finish, cut and failure is a line in night_queue.jsonl.
`--now` runs the queue immediately regardless of the clock (for a test), still one job at a time and with no cut-off unless --end is given.

Queue directory: D:/datasets/sigma_runs/night_queue/{pending,running,done,failed,cut,logs}. A job file is JSON: name, cmd (list), cwd, est_minutes, resumable, queued_utc.
The Windows task 'sigma-ground night window' (scripts/run_night_queue.bat) starts `run` at 01:00 daily. The 04:30 scoreboard, 05:00 benchmark and 06:30 distill tasks
share the machine with the window's tail; a queued job that must not overlap them should carry an estimate that ends it before 04:30.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, time as _time, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
QUEUE = Path("D:/datasets/sigma_runs/night_queue")
DIRS = {k: QUEUE / k for k in ("pending", "running", "done", "failed", "cut", "logs")}
EVENTS = QUEUE / "night_queue.jsonl"


def _mk():
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def _event(kind: str, job: dict, **extra):
    line = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": kind, "name": job["name"], **extra}
    with EVENTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    print(f"{line['utc']} {kind}: {job['name']}" + (f" {extra}" if extra else ""), flush=True)


def _pending() -> list[Path]:
    return sorted(DIRS["pending"].glob("*.json"))


def add(a) -> int:
    _mk()
    if not a.cmd:
        print("no command given (put it after --)"); return 2
    job = {"name": a.name, "cmd": a.cmd, "cwd": a.cwd or str(_ROOT), "est_minutes": a.est_minutes, "resumable": a.resumable,
           "queued_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cut_count": 0}
    f = DIRS["pending"] / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{a.name}.json"
    f.write_text(json.dumps(job, indent=1), encoding="utf-8")
    print(f"queued {a.name} ({a.est_minutes} min est{', resumable' if a.resumable else ''}) -> {f}")
    return 0


def drop(a) -> int:
    _mk()
    hits = [f for f in _pending() if json.loads(f.read_text(encoding='utf-8'))["name"] == a.name]
    for f in hits:
        f.unlink(); print(f"dropped {f}")
    return 0 if hits else 1


def list_(a) -> int:
    _mk()
    for k in ("pending", "running"):
        for f in sorted(DIRS[k].glob("*.json")):
            j = json.loads(f.read_text(encoding="utf-8"))
            print(f"{k:8s} {j['name']:40s} est {j['est_minutes']:>5} min {'resumable' if j['resumable'] else '':9s} cut {j.get('cut_count', 0)}  {' '.join(j['cmd'])[:80]}")
    if EVENTS.exists():
        tail = EVENTS.read_text(encoding="utf-8").strip().splitlines()[-8:]
        print("recent:"); [print("  " + t) for t in tail]
    return 0


def _kill_tree(p: subprocess.Popen):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
    else:
        p.kill()


def run(a) -> int:
    _mk()
    now = datetime.now()
    end = None if (a.now and not a.end_given) else datetime.combine(now.date(), _time.fromisoformat(a.end))
    if end is not None and end <= now:
        end += timedelta(days=1)
    if not a.now:
        start = datetime.combine(now.date(), _time.fromisoformat(a.start))
        if not (start <= now < end):
            print(f"outside the window {a.start}-{a.end} (now {now:%H:%M}); nothing started"); return 0
    ran = 0
    while True:
        pend = _pending()
        if not pend:
            print("queue empty"); break
        left = None if end is None else (end - datetime.now()).total_seconds() / 60.0
        pick = None
        for f in pend:
            j = json.loads(f.read_text(encoding="utf-8"))
            if left is None or (j["est_minutes"] or 60) <= left:
                pick = (f, j); break
        if pick is None:
            print(f"{left:.0f} min left: no pending job fits; stopping"); break
        f, job = pick
        rf = DIRS["running"] / f.name; f.rename(rf)
        log = DIRS["logs"] / f"{job['name']}.log"
        _event("start", job, log=str(log), minutes_left=None if left is None else round(left))
        t0 = time.time()
        with log.open("a", encoding="utf-8") as lf:
            lf.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} start: {' '.join(job['cmd'])}\n"); lf.flush()
            p = subprocess.Popen(job["cmd"], cwd=job["cwd"], stdout=lf, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUTF8": "1"})
            cut = False
            while p.poll() is None:
                if end is not None and datetime.now() >= end:
                    cut = True; _kill_tree(p); p.wait(timeout=60); break
                time.sleep(20)
        mins = (time.time() - t0) / 60.0
        if cut:
            job["cut_count"] = job.get("cut_count", 0) + 1
            if job["resumable"]:
                back = DIRS["pending"] / ("00000000T000000_" + job["name"] + ".json")           # the front of the queue for the next night
                back.write_text(json.dumps(job, indent=1), encoding="utf-8"); rf.unlink()
                _event("cut_requeued", job, minutes=round(mins, 1))
            else:
                rf.rename(DIRS["cut"] / rf.name); _event("cut", job, minutes=round(mins, 1))
            break                                                                             # the window is over
        dest = DIRS["done" if p.returncode == 0 else "failed"] / rf.name
        job["exit_code"] = p.returncode; job["minutes"] = round(mins, 1); job["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        dest.write_text(json.dumps(job, indent=1), encoding="utf-8"); rf.unlink()
        _event("done" if p.returncode == 0 else "failed", job, exit_code=p.returncode, minutes=round(mins, 1))
        ran += 1
    print(f"night window: {ran} job(s) finished")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("add"); s.add_argument("--name", required=True); s.add_argument("--est-minutes", type=int, required=True); s.add_argument("--resumable", action="store_true")
    s.add_argument("--cwd"); s.add_argument("cmd", nargs=argparse.REMAINDER); s.set_defaults(fn=add)
    s = sub.add_parser("list"); s.set_defaults(fn=list_)
    s = sub.add_parser("drop"); s.add_argument("name"); s.set_defaults(fn=drop)
    s = sub.add_parser("run"); s.add_argument("--start", default="01:00"); s.add_argument("--end", default=None); s.add_argument("--now", action="store_true"); s.set_defaults(fn=run)
    a = ap.parse_args()
    if a.verb == "run":
        a.end_given = a.end is not None; a.end = a.end or "07:00"
    if a.verb == "add" and a.cmd and a.cmd[0] == "--":
        a.cmd = a.cmd[1:]
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
