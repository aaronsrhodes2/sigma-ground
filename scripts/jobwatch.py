"""jobwatch: a watchdog for long jobs. Run it as a *tracked background process*: it exits the moment something goes wrong (or the job finishes), and that exit is
what alerts whoever launched it (the Claude Code harness re-invokes the session when a background command exits).

It exists because expensive scripts have died silently here three times (the 20-yr shoot-out jobs, a pool worker in the fair contest, and a Windows process
kill with no event-log trace). It never touches the job; it only watches and logs.

Each check (every ``--interval`` seconds) it looks at:
  * the job's driver process(es) (by PID, immune to PID reuse): alive or gone;
  * the driver's python children (workers): a driver with zero children for two checks is stuck or wedged;
  * the CPU used by the driver and its workers between checks (busy or idle): it tells a long computation from a hang when the stall limit hits, and a BUSY job gets one grace period;
  * progress signals: a growing line count in a log or results file, a newer mtime in a checkpoint folder, more files matching a glob. If NONE changes for
    ``--stall-minutes`` the job is stalled (or in an infinite loop): ALERT;
  * new lines in the job's logs and stderr files matching an error pattern (new lines only, from the moment the watch starts unless ``--from-start``):
    FATAL patterns (Traceback, GAVE UP, MemoryError, abruptly, Killed) are ALERTs, retry noise (FAILED ... retrying) is a WARN that is logged but does not exit;
  * memory pressure (available RAM) as a WARN, because low memory is the leading suspect for silent worker deaths.
It ends with exit code 0 and a DONE line when the driver is gone AND a success marker is present (a success regex in the logs, or a success file); with exit code 2
and an ALERT line otherwise. Every check is appended to ``<state-dir>/<name>.jsonl`` and the latest state is in ``<name>.status.json`` (read it for a board).

Presets bundle the right signals for our runners:
  --preset shootout --outdir DIR   (scripts/run_shootout_parallel.py: short_run.log/long_run.log ' done ', ckpt folder, 'finished:' marker)
  --preset fair     --outdir DIR   (scripts/fair_jpl_contest.py: results jsonl, contest log ' done ', fair_contest_contest.json success file)

usage (as a tracked background command):
  python scripts/jobwatch.py --name fair --pid 3220 --preset fair --outdir <dir> --log <dir>/contest2.log --interval 120
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

FATAL = r"Traceback|GAVE UP|MemoryError|abruptly|Killed|Fatal|BrokenProcessPool|exit code (?!0\b)\d+ .*(?:attempt 3/3)"
WARN = r"FAILED|Error|error:|WARNING|retry"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def proc_key(pid: int):
    try:
        p = psutil.Process(pid)
        return (pid, p.create_time())
    except psutil.Error:
        return None


def alive(key) -> bool:
    if key is None:
        return False
    try:
        p = psutil.Process(key[0])
        return abs(p.create_time() - key[1]) < 1e-3 and p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def children_python(keys) -> int:
    n = 0
    for k in keys:
        if alive(k):
            try:
                n += sum(1 for c in psutil.Process(k[0]).children(recursive=True) if "python" in c.name().lower())
            except psutil.Error:
                pass
    return n


def cpu_seconds(keys) -> float:
    """CPU seconds (user + system) used so far by the driver and all its descendants: the difference between checks says busy or idle."""
    total = 0.0
    for k in keys:
        if alive(k):
            try:
                p = psutil.Process(k[0])
                for q in [p] + p.children(recursive=True):
                    try:
                        t = q.cpu_times()
                        total += t.user + t.system
                    except psutil.Error:
                        pass
            except psutil.Error:
                pass
    return total


def signal_value(spec: str):
    """A progress signal: 'lines:PATH[::REGEX]' count of matching lines, 'mtime:PATH' newest mtime under a file or folder, 'count:GLOB' number of files."""
    kind, _, rest = spec.partition(":")
    if kind == "lines":
        path, _, rx = rest.partition("::")                       # lines:PATH or lines:PATH::REGEX (paths contain ':' on Windows)
        p = Path(path)
        if not p.exists():
            return None
        try:
            lines = p.read_text(errors="replace").splitlines()
        except OSError:
            return None
        return sum(1 for ln in lines if re.search(rx, ln)) if rx else len(lines)
    if kind == "mtime":
        p = Path(rest)
        if not p.exists():
            return None
        if p.is_file():
            return p.stat().st_mtime
        newest = 0.0
        for root, _, files in os.walk(p):
            for f in files:
                try:
                    newest = max(newest, os.stat(os.path.join(root, f)).st_mtime)
                except OSError:
                    pass
        return newest
    if kind == "count":
        return len(glob.glob(rest))
    raise ValueError(f"unknown progress signal {spec!r}")


class Tail:
    """Reads only the new lines of a file between calls (from the current end unless from_start)."""

    def __init__(self, path: str, from_start: bool):
        self.path = Path(path)
        self.pos = 0 if from_start or not self.path.exists() else self.path.stat().st_size

    def new_lines(self) -> list[str]:
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size < self.pos:
            self.pos = 0
        with open(self.path, "rb") as fh:
            fh.seek(self.pos)
            data = fh.read()
            self.pos = fh.tell()
        return data.decode(errors="replace").splitlines()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--pid", type=int, nargs="+", required=True, help="driver process id(s) to watch")
    ap.add_argument("--progress", action="append", default=[], help="lines:PATH[::REGEX] | mtime:PATH | count:GLOB (repeatable)")
    ap.add_argument("--log", action="append", default=[], help="log file(s) scanned for new error lines")
    ap.add_argument("--err-glob", action="append", default=[], help="stderr file glob(s): any new text is scanned too")
    ap.add_argument("--success-regex", default=r"finished:|\bDONE\b|merged ->", help="a line in a log that marks a clean finish")
    ap.add_argument("--success-file", action="append", default=[], help="a file whose existence marks a clean finish")
    ap.add_argument("--fatal-regex", default=FATAL)
    ap.add_argument("--warn-regex", default=WARN)
    ap.add_argument("--stall-minutes", type=float, default=45.0)
    ap.add_argument("--expected-job-minutes", type=float, default=0.0,
                    help="longest single job in minutes; the stall limit becomes at least 2.5x this (a limit shorter than one job is a false alarm by construction)")
    ap.add_argument("--busy-grace", type=int, default=1,
                    help="when the stall limit hits while the job is still burning CPU, extend it this many times before alerting; an IDLE stalled job alerts at once")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--mem-warn-gb", type=float, default=1.0)
    ap.add_argument("--min-children", type=int, default=None, help="alert if the driver is alive with fewer python children than this for two checks (0 disables). Default: 1 with a preset "
                    "(a driver with workers), 0 without one (a single-process job such as a trickle has no children; the default of 1 raised a false alarm on 2026-10-16)")
    ap.add_argument("--from-start", action="store_true", help="scan logs from their beginning instead of from now")
    ap.add_argument("--exit-on-warn", action="store_true")
    ap.add_argument("--state-dir", type=Path, default=Path("local-cache/jobwatch"))
    ap.add_argument("--preset", choices=["shootout", "fair"], default=None)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--max-hours", type=float, default=0.0, help="stop watching (exit 3) after this long; 0 = no limit")
    a = ap.parse_args(argv)
    if a.min_children is None:
        a.min_children = 1 if a.preset else 0

    if a.preset == "shootout" and a.outdir:
        for f in ("short_run.log", "long_run.log", "driver.out"):          # the launcher's stdout, whatever it was redirected to (driver.out too: a false 'driver gone, no success marker' alert on 2026-10-23)
            if (a.outdir / f).exists():
                a.progress.append(f"lines:{a.outdir / f}:: done ")
                a.log.append(str(a.outdir / f))
        a.progress.append(f"mtime:{a.outdir / 'ckpt'}")
        a.err_glob += [str(a.outdir / "*" / "*.log"), str(a.outdir / "*_run.err")]
    if a.preset == "fair" and a.outdir:
        for tag in ("contest", "validate"):                              # the runner names its outputs by mode (--validate writes ..._validate...)
            a.progress.append(f"lines:{a.outdir / f'fair_contest_{tag}.jsonl'}")
            a.success_file.append(str(a.outdir / f"fair_contest_{tag}.json"))
        a.progress.append(f"mtime:{a.outdir / 'jobs'}")
        a.err_glob.append(str(a.outdir / "jobs" / "*.err"))

    if a.expected_job_minutes:
        a.stall_minutes = max(a.stall_minutes, 2.5 * a.expected_job_minutes)
    keys = [proc_key(p) for p in a.pid]
    if not any(keys):
        print(f"JOBWATCH ALERT {a.name}: none of the PIDs {a.pid} is running at start", flush=True)
        return 2
    a.state_dir.mkdir(parents=True, exist_ok=True)
    events = a.state_dir / f"{a.name}.jsonl"
    status = a.state_dir / f"{a.name}.status.json"
    tails = [Tail(p, a.from_start) for p in a.log]
    err_tails: dict[str, Tail] = {}
    last_vals: dict[str, object] = {}
    last_progress = time.time()
    t_start = time.time()
    low_children = 0
    low_mem = 0
    last_cpu = cpu_seconds(keys)
    graces_used = 0
    ok_re, fatal_re, warn_re = re.compile(a.success_regex), re.compile(a.fatal_regex), re.compile(a.warn_regex)
    success_seen = False

    def log(kind: str, msg: str, **extra) -> None:
        rec = {"t": now(), "kind": kind, "msg": msg, **extra}
        with open(events, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def finish(code: int, line: str) -> int:
        log("END", line, code=code)
        print(line, flush=True)
        return code

    log("START", f"watching {a.pid}", progress=a.progress, logs=a.log, stall_minutes=a.stall_minutes)
    print(f"jobwatch {a.name}: watching PID(s) {a.pid} every {a.interval:g}s; stall limit {a.stall_minutes:g} min; state in {a.state_dir}", flush=True)
    while True:
        time.sleep(a.interval)
        alive_now = [alive(k) for k in keys]
        driver_up = any(alive_now)
        # --- logs and stderr: new lines only
        new_lines: list[tuple[str, str]] = []
        for t in tails:
            new_lines += [(str(t.path), ln) for ln in t.new_lines()]
        for pat in a.err_glob:
            for f in glob.glob(pat):
                if f not in err_tails:
                    err_tails[f] = Tail(f, a.from_start)
                new_lines += [(f, ln) for ln in err_tails[f].new_lines() if ln.strip()]
        for src, ln in new_lines:
            if ok_re.search(ln):
                success_seen = True
            if fatal_re.search(ln):
                return finish(2, f"JOBWATCH ALERT {a.name} {now()}: error in {Path(src).name}: {ln.strip()[:240]}")
            if warn_re.search(ln):
                log("WARN", ln.strip()[:240], src=Path(src).name)
                if a.exit_on_warn:
                    return finish(2, f"JOBWATCH ALERT {a.name} {now()}: warning in {Path(src).name}: {ln.strip()[:240]}")
        # --- progress
        changed = False
        vals = {}
        for spec in a.progress:
            v = signal_value(spec)
            vals[spec] = v
            if last_vals.get(spec, "__unset__") != v:
                changed = True
        if changed:
            last_progress = time.time()
        last_vals = vals
        stalled_min = (time.time() - last_progress) / 60.0
        children = children_python(keys)
        cpu_now = cpu_seconds(keys)
        cpu_delta = cpu_now - last_cpu
        last_cpu = cpu_now
        busy = cpu_delta > 0.5 * a.interval                              # at least half a core's worth of work since the last check
        mem_gb = psutil.virtual_memory().available / 1e9
        st = {"name": a.name, "time": now(), "driver_alive": driver_up, "python_children": children, "available_ram_gb": round(mem_gb, 2),
              "minutes_since_progress": round(stalled_min, 1), "progress": {k: (v if not isinstance(v, float) else round(v, 1)) for k, v in vals.items()},
              "elapsed_h": round((time.time() - t_start) / 3600, 2), "cpu_seconds_last_interval": round(cpu_delta, 1), "cpu_busy": busy}
        status.write_text(json.dumps(st, indent=1))
        log("CHECK", "", **{k: st[k] for k in ("driver_alive", "python_children", "available_ram_gb", "minutes_since_progress", "cpu_seconds_last_interval")})
        # --- verdicts
        for f in a.success_file:
            if Path(f).exists():
                success_seen = True
        if not driver_up:
            if success_seen:
                return finish(0, f"JOBWATCH DONE {a.name} {now()}: driver exited and a success marker was seen")
            return finish(2, f"JOBWATCH ALERT {a.name} {now()}: the driver process is gone and no success marker was seen (killed or crashed). "
                             f"Last progress {stalled_min:.0f} min ago. See {events}")
        if a.min_children and children < a.min_children:
            low_children += 1
            if low_children >= 2 and not success_seen:
                return finish(2, f"JOBWATCH ALERT {a.name} {now()}: driver alive but only {children} python children (expected >= {a.min_children}) for two checks")
        else:
            low_children = 0
        if stalled_min >= a.stall_minutes * (1 + graces_used):
            if busy and graces_used < a.busy_grace:
                graces_used += 1
                log("WARN", f"no progress for {stalled_min:.0f} min but the job is busy ({cpu_delta:.0f} CPU-s in the last check): limit extended (grace {graces_used}/{a.busy_grace}). "
                            "If one job really takes longer than this, raise --expected-job-minutes")
            else:
                state = f"BUSY ({cpu_delta:.0f} CPU-s in the last check: looping, or one job is longer than the limit)" if busy else f"IDLE ({cpu_delta:.0f} CPU-s in the last check: hung)"
                return finish(2, f"JOBWATCH ALERT {a.name} {now()}: no progress signal changed for {stalled_min:.0f} min (limit {a.stall_minutes:g}); workers are {state}")
        if mem_gb < a.mem_warn_gb:
            low_mem += 1
            if low_mem in (1, 5):
                log("WARN", f"available RAM {mem_gb:.2f} GB below {a.mem_warn_gb:g} GB")
        else:
            low_mem = 0
        if a.max_hours and (time.time() - t_start) / 3600 > a.max_hours:
            return finish(3, f"JOBWATCH TIMEOUT {a.name} {now()}: watched for {a.max_hours:g} h and the job is still running (not an error)")


if __name__ == "__main__":
    sys.exit(main())
