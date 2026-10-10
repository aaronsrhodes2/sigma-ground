"""Cached access to Mentat for the engine experiments (PLATINUM section 21: physics is computed, never recalled).

Every number an experiment predicts comes from a Mentat tool through here. Answers are cached on disk
(`mentat_cache.json`) together with the tool's own `source` and `provenance_tag`, so grading and the test
suite run offline, deterministically, and a prediction is always traceable to the Mentat call that made it.

    from mentat_io import call
    r = call("free_fall_time", {"height_m": 2.0})     # {"value":..., "units":..., "source":..., "tag":...}

Modes: a cache hit is returned as is. On a miss it asks the live container (docker exec skippytel-mentat); if that
is unavailable it raises. MENTAT_OFFLINE=1 forbids live calls (what the tests set); MENTAT_REFRESH=1 re-asks everything.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "mentat_cache.json"
sys.path.insert(0, str(HERE.parents[2] / "tools"))        # tools/mentat_client.py

_mem: dict | None = None


def _load() -> dict:
    global _mem
    if _mem is None:
        _mem = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    return _mem


def _key(tool: str, args: dict) -> str:
    return tool + " " + json.dumps(args, sort_keys=True, separators=(",", ":"))


def call(tool: str, args: dict | None = None) -> dict:
    """Return {"value","units","source","tag","formula","inputs"} for one Mentat tool call."""
    args = args or {}
    cache, key = _load(), _key(tool, args)
    if key in cache and not os.environ.get("MENTAT_REFRESH"):
        return cache[key]
    if os.environ.get("MENTAT_OFFLINE"):
        raise RuntimeError(f"offline and not cached: {key}  (run stage/gmod/experiments/build.py with Docker up to record it)")
    from mentat_client import mentat_raw                      # imported late: needs Docker only on a live call
    reply = mentat_raw(tool, args)
    if not reply.get("ok"):
        raise RuntimeError(f"mentat {tool} {args}: {reply.get('error')}")
    res = reply["result"]
    entry = {"value": res.get("value"), "units": res.get("units"), "source": res.get("source"),
             "tag": res.get("provenance_tag"), "formula": res.get("formula"), "notes": res.get("notes"),
             "inputs": res.get("inputs")}
    for extra in ("procedure", "steps", "result"):            # procedure tools keep their cascade here
        if extra in res:
            entry[extra] = res[extra]
    cache[key] = entry
    _save(cache)
    return entry


def _save(cache: dict) -> None:
    """Write the cache atomically (temp file, then replace), retrying: on Windows a plain write now and then fails with
    'Invalid argument' while another process has the file open."""
    text = json.dumps(cache, indent=1, sort_keys=True, ensure_ascii=False)
    tmp = CACHE.with_name(CACHE.name + ".tmp")
    for attempt in range(8):
        try:
            tmp.write_text(text, encoding="utf-8", newline="\n")
            os.replace(tmp, CACHE)
            return
        except OSError:
            if attempt == 7:
                raise
            time.sleep(0.25)


def provenance(tool: str, args: dict, entry: dict) -> dict:
    """The record carried in a spec so every prediction names the Mentat call behind it."""
    return {"tool": tool, "args": args, "source": entry.get("source"), "tag": entry.get("tag")}
