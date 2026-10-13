"""Live check of every file the addon persists, in the REAL engine (dedicated server: no window, no mouse, no focus).

    python stage/gmod/holodeck/persist_probe.py [--sid 76561197912345678] [--out result.json] [--wait-minutes 30]

Runs tools/persist_probe.lua twice with a full server STOP and START in between (the restart is the point): phase "write" stores a record for a made-up
SteamID64 of real magnitude and for a bot's, through the real wakenames / faces / director prefs / journal code; phase "read" reads them back after the
restart and feeds the old keyed file formats to the loaders. Waits (up to --wait-minutes) while another hl2 / gmod / srcds process runs, never starts a
second instance, and leaves nothing running. The server's own persistence files are saved first and put back at the end.
Exit code 0 only when every check is true.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments"))
import run as exp_run  # noqa: E402  (DS_DATA)
import run_tool  # noqa: E402

GAME_PROCS = ("hl2.exe", "gmod.exe", "srcds.exe", "srcds_win64.exe")
FILES = ["wake_words.json", "wake_words.bak.txt", "faces.json", "faces.bak.txt", "skippy_prefs.json", "skippy_prefs.bak.txt"]


def running() -> list[str]:
    import psutil
    return [p.info["name"] for p in psutil.process_iter(["name"]) if (p.info["name"] or "").lower() in GAME_PROCS]


def wait_for_quiet(minutes: float) -> None:
    t0 = time.time()
    while running():
        if time.time() - t0 > minutes * 60:
            raise RuntimeError(f"another game process is still running after {minutes} minutes: {running()}")
        time.sleep(10)


def snapshot(data: Path, sids: list[str]) -> dict[Path, bytes | None]:
    snap = {data / n: ((data / n).read_bytes() if (data / n).exists() else None) for n in FILES}
    for s in sids:
        for suffix in (".txt", ".1.txt"):
            q = data / "journal" / f"{s}{suffix}"
            snap[q] = q.read_bytes() if q.exists() else None
    return snap


def restore(snap: dict[Path, bytes | None]) -> None:
    for p, b in snap.items():
        if b is None:
            p.unlink(missing_ok=True)
        else:
            p.write_bytes(b)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", default="76561197912345678", help="a made-up 17 digit id of real SteamID64 magnitude (NOT a real person's)")
    ap.add_argument("--out")
    ap.add_argument("--wait-minutes", type=float, default=30)
    a = ap.parse_args(argv)
    wait_for_quiet(a.wait_minutes)
    data = exp_run.DS_DATA
    data.mkdir(parents=True, exist_ok=True)
    snap = snapshot(data, [a.sid])                      # (the bot's id is not known yet: its journal file is removed below)
    result: dict = {}
    bot_sid = None
    try:
        w = run_tool.run_tool("persist_probe", {"phase": "write", "sid": a.sid}, timeout=240)
        bot_sid = w.get("bot_sid")
        r = run_tool.run_tool("persist_probe", {"phase": "read", "sid": a.sid, "bot_sid": bot_sid}, timeout=240)     # a fresh server: the restart
        result = {"write": w, "read": r}
    finally:
        restore(snap)
        for s in [a.sid, bot_sid]:
            if s:
                for suffix in (".txt", ".1.txt"):
                    q = data / "journal" / f"{s}{suffix}"
                    if q not in snap or snap[q] is None:        # (the probe wipes and rewrites these ids itself: a bot's id is never a person's journal)
                        q.unlink(missing_ok=True)
        shutil.rmtree(data / "persist_probe", ignore_errors=True)
        left = running()
        if left:
            print("WARNING: still running:", left)
    checks = {f"{ph}.{k}": v for ph in ("write", "read") for k, v in result.get(ph, {}).get("checks", {}).items()}
    bad = {k: v for k, v in checks.items() if not v["ok"]}
    text = json.dumps(result, indent=1)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(f"{len(checks) - len(bad)}/{len(checks)} checks true; engine extension rules: {json.dumps(result.get('write', {}).get('ext'))}")
    print("number keys (default / ignoreConversions):", result.get("write", {}).get("keys_default"), result.get("write", {}).get("keys_ignore_conversions"))
    for k, v in bad.items():
        print("FAILED", k, v.get("evidence"))
    return 0 if checks and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
