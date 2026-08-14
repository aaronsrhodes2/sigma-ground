"""run_apt_probe.py -- APT.0 ANGLE GO/NO-GO runner (R4.7 Phase 2 probe spike).

Starts serve.py, launches ONE dedicated _CDPBrowser, navigates to
apt_probe.html, polls misc/BROWSER_REPORT.json for slug "apt_probe",
prints the verdict, tears down unconditionally (finally block: cdp.close()
kills the Chrome process and reaps its throwaway profile dir; proc.terminate()
kills serve.py). Direct sibling of measure_compile_time.py's _run_probe,
trimmed to a single shot -- no sweep/variant machinery needed for a one-off
spike.

Run:
    python tools/run_apt_probe.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

import render_regression as rr                                # noqa: E402

_BROWSER_REPORT = _ROOT / "misc" / "BROWSER_REPORT.json"
SLUG = "apt_probe"


def _read_report() -> dict:
    try:
        d = json.loads(_BROWSER_REPORT.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def main():
    before = _read_report().get(SLUG)
    port = rr._free_port()
    proc = rr._start_server(port)
    cdp = rr._CDPBrowser()
    try:
        cdp.navigate(f"http://127.0.0.1:{port}/apt_probe.html")
        t0, result = time.time(), None
        while time.time() - t0 < 60.0:
            cur = _read_report().get(SLUG)
            if cur is not None and cur != before:
                result = cur
                break
            time.sleep(0.3)
    finally:
        cdp.close()
        proc.terminate()

    if result is None:
        print("NO-GO (infra): apt_probe.html never posted to /probe within 60s")
        sys.exit(2)

    print(f"renderer: {result.get('renderer', '?')}")
    for key, p in (result.get("probes") or {}).items():
        print(f"  {key}: {'PASS' if p.get('pass') else 'FAIL'}  {json.dumps(p)[:300]}")
    if result.get("console_errors"):
        print("console errors:")
        for e in result["console_errors"]:
            print(f"  {e}")
    verdict = "GO" if result.get("ok") else "NO-GO"
    print(f"\nVERDICT: {verdict}")
    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
