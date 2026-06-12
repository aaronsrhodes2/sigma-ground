"""Browser-verify the gauntlet — every artifact must DRAW, not just parse.

Starts the viewer server, opens each gauntlet scene with ``?probe=1`` (the
viewer's probe mode POSTs its in-page SDF self-check + console errors to
``/probe`` after the first drawn frame), waits for all reports, and prints
the verdict table. misc/BROWSER_REPORT.json holds the raw reports.

    python tools/browser_gauntlet.py            # all gauntlet slugs
    python tools/browser_gauntlet.py g_a g_b    # specific slugs
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time
import webbrowser

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DATA = _ROOT / "sigma_ground" / "radiance" / "web" / "data"
_REPORT = _ROOT / "misc" / "BROWSER_REPORT.json"
_URL = "http://127.0.0.1:8765"


def _gauntlet_slugs() -> list:
    try:
        idx = json.load(open(_DATA / "scenes.json", encoding="utf-8"))
    except Exception:
        return []
    return [e["slug"] for e in idx if e.get("group") == "gauntlet"]


def main(argv: list) -> None:
    slugs = argv or _gauntlet_slugs()
    if not slugs:
        raise SystemExit("no gauntlet slugs found in scenes.json")
    _REPORT.write_text("{}", encoding="utf-8")          # fresh run

    server = subprocess.Popen(
        [sys.executable, "-m", "sigma_ground.radiance.web.serve"],
        cwd=str(_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(2.0)
        done: set = set()
        for slug in slugs:
            webbrowser.open(f"{_URL}/?scene={slug}&probe=1")
            # wait for THIS slug's report before opening the next (one tab's
            # worth of GPU at a time; 20 s timeout per scene)
            t0 = time.time()
            while time.time() - t0 < 20.0:
                try:
                    rep = json.load(open(_REPORT, encoding="utf-8"))
                except Exception:
                    rep = {}
                if slug in rep:
                    done.add(slug)
                    break
                time.sleep(0.5)
            print(f"  {'+' if slug in done else '!'} {slug}"
                  + ("" if slug in done else "  (no report in 20 s)"))

        rep = json.load(open(_REPORT, encoding="utf-8"))
        ok = 0
        print(f"\n{'slug':24s} {'drawn':6s} {'selfcheck':10s} errors")
        for slug in slugs:
            r = rep.get(slug)
            if not r:
                print(f"{slug:24s} {'-':6s} {'-':10s} NO REPORT")
                continue
            sc = r.get("selfcheck_pass")
            sc_s = "pass" if sc else ("n/a" if sc is None else "FAIL")
            errs = "; ".join(r.get("console_errors") or [])[:60]
            good = r.get("shader_ok") and sc is not False and not errs
            ok += bool(good)
            print(f"{slug:24s} {'yes' if r.get('shader_ok') else 'NO':6s} "
                  f"{sc_s:10s} {errs}")
        print(f"\n{ok}/{len(slugs)} scenes browser-verified")
        if ok < len(slugs):
            raise SystemExit(1)
    finally:
        server.terminate()


if __name__ == "__main__":
    main(sys.argv[1:])
