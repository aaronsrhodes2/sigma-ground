"""Skippy's solid turns, in the real game: a scripted conversation with two bots as players, through the real chat hook and the voice path's entry.
Hands-off (a DEDICATED server: no window, no mouse, no focus taken), one game process, nothing left running.

    python stage/gmod/holodeck/turns_live.py [--out FILE] [--wait-minutes 30] [--no-tiebreak]

Waits (up to --wait-minutes) until no hl2 / gmod / srcds process is running, starts a LOCAL bridge on loopback with its OWN private state in a temporary folder
(so Aaron's real counters, history and cache are untouched) and the REAL turns engine: the frontier model when a key is reachable (environment, else the single
variable of the vault file; the key is never printed), else the local Ollama model. For the scene "invalid_model_output" the driver makes the frontier model misbehave
(a junk reply to every attempt) so the retry and the fallback line are exercised. Runs tools/turns_live.lua, prints the decisions log and what he said, and restores
the dedicated server's data files it touched (the captains' file, the journal). See stage/gmod/holodeck/README.md "Solid turns".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments"))
import run as exp_run  # noqa: E402
import run_tool  # noqa: E402


def game_processes() -> list[str]:
    out = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout.lower()
    return sorted({n for n in ("hl2.exe", "gmod.exe", "srcds.exe", "srcds_win64.exe") if n in out})


def wait_for_calm(minutes: float) -> bool:
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        if not game_processes():
            return True
        print("another game process is running (%s); waiting..." % ", ".join(game_processes()), flush=True)
        time.sleep(30)
    return not game_processes()


def sha(p: Path) -> str | None:
    return hashlib.sha1(p.read_bytes()).hexdigest() if p.exists() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "data" / "turns_live_result.json"))
    ap.add_argument("--wait-minutes", type=float, default=30)
    ap.add_argument("--no-tiebreak", action="store_true")
    ap.add_argument("--no-captain-file", action="store_true")
    ap.add_argument("--timeout", type=int, default=480)
    a = ap.parse_args(argv)
    if not wait_for_calm(a.wait_minutes):
        print("a game process is still running after %.0f minutes; not starting" % a.wait_minutes)
        return 3
    os.environ.setdefault("GMOD_DS_MAXPLAYERS", "4")
    state = Path(tempfile.mkdtemp(prefix="skippy_turns_live_"))
    os.environ["SKIPPY_QUIP_DIR"] = str(state)                       # BEFORE the quip and turns modules are first used
    os.environ["SKIPPY_QUIP_PROVIDER"] = "template"                  # the sign-off line is a bank line: no model call for it
    import skippy_bridge  # noqa: E402
    import skippy_quips  # noqa: E402
    import skippy_turns  # noqa: E402
    skippy_bridge.TOKEN = secrets.token_hex(8)
    skippy_quips.set_engine(None)
    skippy_turns.set_engine(None)
    skippy_quips.ensure_gate(skippy_bridge)
    skippy_bridge.GATE.per_minute["quip"], skippy_bridge.GATE.per_caller["quip"] = 60, 60
    skippy_bridge.GATE.rate["quip"] = skippy_bridge.Window(60, 60, skippy_bridge.GATE.clock)

    # fault injection, for ONE scene only: the frontier model answers junk to every attempt for a sentence that says INVALIDOUTPUT
    real_frontier = skippy_turns.Engine._frontier
    injected = {"n": 0}

    def frontier(self, system, user, timeout, model, max_tokens, tiebreak=False, temperature=None):
        if "INVALIDOUTPUT" in user:
            injected["n"] += 1
            self.ledger.record(50, 10, tiebreak)
            return '{"say": "**Bold** claim, see https://example.com", "cmd": "rm -rf"}'
        return real_frontier(self, system, user, timeout, model, max_tokens, tiebreak, temperature)

    skippy_turns.Engine._frontier = frontier
    eng = skippy_turns.get_engine(skippy_bridge)                      # builds the engine (and logs "key present: yes/no (source)": never the value)
    status = eng.status()
    srv = skippy_bridge.serve("127.0.0.1", 8777)
    data = exp_run.DS_DATA
    captain = data / "captain_sids.txt"
    captain_before = captain.read_bytes() if captain.exists() else None
    captain_sha = sha(captain)
    jdir = data / "journal"
    jbak = state / "journal_backup"
    if jdir.exists():
        shutil.copytree(jdir, jbak)
    try:
        res = run_tool.run_tool("turns_live", {"bridge": "http://127.0.0.1:8777", "token": skippy_bridge.TOKEN, "tiebreak": not a.no_tiebreak,
                                               "captain_file": not a.no_captain_file}, client=False, timeout=a.timeout)
    finally:
        srv.shutdown()
        srv.server_close()
        skippy_turns.Engine._frontier = real_frontier
        if captain_before is None:
            captain.unlink(missing_ok=True)
        else:
            captain.write_bytes(captain_before)
        shutil.rmtree(jdir, ignore_errors=True)
        if jbak.exists():
            shutil.copytree(jbak, jdir)
    restored = sha(captain) == captain_sha
    leftover = game_processes()
    view = eng.ledger.view(skippy_turns.frontier_cap_default())
    res["bridge"] = {"key_present": status["key_present"], "key_source": status["key_source"], "frontier_model": status["frontier_model"], "local_model": status["local_model"],
                     "injected_junk_attempts": injected["n"],
                     "ledger": {k: view[k] for k in ("calls", "tiebreaks", "in_tokens", "out_tokens", "errors", "local", "cache_hits", "fallbacks", "paused")}}
    res["captain_file_restored"] = restored
    res["game_processes_left"] = leftover
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print_report(res)
    shutil.rmtree(state, ignore_errors=True)
    return 0 if restored and not leftover else 1


def print_report(res: dict) -> None:
    b = res["bridge"]
    print(f"\nkey present: {'yes' if b['key_present'] else 'no'} ({b['key_source']}); model route in the game: {res.get('model_route')}; roast levels {res.get('roast')}")
    print(f"ledger: {b['ledger']}  (injected junk attempts: {b['injected_junk_attempts']})")
    print(f"{'scene':26s} {'by':6s} {'how':8s} {'built':5s} decision(s)  ->  what he said")
    for s in res["scenes"]:
        dec = s["decisions"][-1] if s["decisions"] else {}
        d = f"{dec.get('addressed', '-')}/{dec.get('final', '-')}" + (f" ({dec.get('score')})" if dec else "")
        said = " | ".join(s["skippy_said"])[:150]
        lt = s.get("last_turn") or {}
        src = f" [{lt.get('source')}, tries {lt.get('tries')}; {'; '.join(lt.get('why') or [])[:90]}]" if s["skippy_said"] and lt else ""
        print(f"{s['scene']:26s} {s['by']:6s} {s['how']:8s} {s['built']:5d} {d:22s} {s['text'][:34]!r} -> {said!r}{src}")
    print(f"counts {res.get('counts')}; avatar active after sign-off: {res.get('avatar_active_after')}; captain file restored: {res['captain_file_restored']}; game processes left: {res['game_processes_left']}")


if __name__ == "__main__":
    raise SystemExit(main())
