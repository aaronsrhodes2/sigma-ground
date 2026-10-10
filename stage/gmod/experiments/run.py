"""Run one engine experiment end to end, one step at a time:  python stage/gmod/experiments/run.py S01 [--shots]

build the spec (Mentat predictions, cached) -> arm the in-game trigger -> start Garry's Mod -> wait for the result
file -> close it -> grade it independently.

Where it runs:
  * the DEDICATED SERVER (default when installed): no window, no sound, no mouse, nothing to pop up. Same Havok.
  * the game itself with --visible / --shots / --client: for watching, and for screenshots (a server has no camera).
One instance of the chosen kind only: refuses if one is already running.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import build  # noqa: E402
import grade  # noqa: E402

DATA = grade.DEFAULT_DATA
DS_DIR = Path(os.environ.get("GMOD_DS_DIR", "D:/Aaron/servers/gmod-ds"))
DS_DATA = DS_DIR / "garrysmod" / "data" / "mentat"


def _bash() -> str:
    """Git Bash. A bare `bash` on a Windows PATH is often the WSL launcher, which cannot run these scripts."""
    import shutil
    for cand in (os.environ.get("GMOD_BASH"), "C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe"):
        if cand and Path(cand).exists():
            return cand
    return shutil.which("bash") or "bash"


def _ctl_cmd(script: str) -> list:
    return [_bash(), str(REPO / "stage" / "gmod" / script).replace("\\", "/")]


CTL = _ctl_cmd("gmod_ctl.sh")
DS_CTL = _ctl_cmd("gmod_ds_ctl.sh")


class _Out:
    def __init__(self, rc, text):
        self.returncode, self.stdout, self.stderr = rc, text, ""


def ctl(*a, server=False):
    """Run gmod_ctl.sh (or gmod_ds_ctl.sh). Output goes to a FILE, not a pipe: the game it starts inherits the process's handles on
    Windows, and a pipe would then stay open (and subprocess.run would block) until the game exits."""
    import tempfile
    with tempfile.TemporaryFile("w+", encoding="utf-8") as f:
        rc = subprocess.run((DS_CTL if server else CTL) + list(a), stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
        f.seek(0)
        return _Out(rc, f.read())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("--shots", action="store_true", help="save screenshots (setup, release, mid, end); needs the game, not the server")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--timeout", type=int, default=150)
    ap.add_argument("--keep-open", action="store_true", help="leave the game or server running afterwards")
    ap.add_argument("--client", action="store_true", help="use the game instead of the dedicated server")
    ap.add_argument("--visible", action="store_true", help="use the game with its window shown (default game mode is minimized, muted, no mouse capture)")
    ap.add_argument("--map", default="gm_flatgrass",
                    help="gm_flatgrass (an empty plain: no walls or ledges to disturb a fall) or gm_construct")
    args = ap.parse_args(argv)
    sid = args.id.upper()
    run_id = args.run_id or time.strftime("%H%M%S")
    client = args.client or args.visible or args.shots or not (DS_DIR / "srcds.exe").exists()
    server = not client
    data = DATA if client else DS_DATA
    kind = "game" if client else "dedicated server"

    if ctl("status", server=server).returncode == 0:
        print(f"The {kind} is already running (one instance only). Stop it first.")
        return 2
    if server:
        out = ctl("setup", server=True)
        if out.returncode != 0:
            print(out.stdout.strip())
            return 2
    build.write(sid)
    data.mkdir(parents=True, exist_ok=True)
    for f in data.glob(f"experiment_{sid}.json"):
        f.unlink()
    for f in data.glob(f"shot_exp_{sid}_*.jpg"):
        f.unlink()
    trigger = {"id": sid, "run_id": run_id, "shots": bool(args.shots), "quit": server}
    (data / "experiment.txt").write_text(json.dumps(trigger), encoding="utf-8")
    if server:
        out = ctl("start", args.map, server=True)
    else:
        out = ctl("start", args.map, *([] if (args.visible or args.shots) else ["--quiet"]))
    print(out.stdout.strip() or out.stderr.strip())
    if out.returncode != 0:
        return 2
    result = data / f"experiment_{sid}.json"
    t0 = time.time()
    while not result.exists() and time.time() - t0 < args.timeout:
        time.sleep(2)
    got = result.exists()
    print(f"result {'written' if got else 'NOT written'} after {time.time() - t0:.0f} s on the {kind}")
    time.sleep(1.5)                                    # let the file finish flushing
    if not args.keep_open:
        if server:                                     # it quits by itself after writing; give it a moment, then make sure
            for _ in range(10):
                if ctl("status", server=True).returncode != 0:
                    break
                time.sleep(1)
        print(ctl("stop", server=server).stdout.strip())
    if not got:
        return 1
    return grade.main([sid, "--run-id", run_id, "--data", str(data)])


if __name__ == "__main__":
    sys.exit(main())
