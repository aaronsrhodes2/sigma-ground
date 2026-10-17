"""Send an approved group of JPL Horizons requests: one at a time, 0.5 s apart, one file per request in a NEW directory, never overwriting a good file.

The group definitions are exactly the ones listed in misc/NEXT_SHOOTOUT_FETCH_REQUESTS.md and misc/COMET_FETCH_REQUESTS.md (the Captain approved them there). The first request of a group is sent
alone and checked (HTTP 200, an ephemeris block, the expected body name); the rest are sent only if it passes, and every response is checked the same way. A response that fails a check is logged
and kept under ``rejected/`` (never used, never guessed at).

usage: python scripts/fetch_horizons_batch.py {asteroids,moons,comet73p} [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
FIX = _ROOT / "sigma_ground/field/interface/fixtures"
BASE = "https://ssd.jpl.nasa.gov/api/horizons.api"

MOONS = [(505, "Amalthea"), (514, "Thebe"), (506, "Himalia"), (610, "Janus"), (611, "Epimetheus"), (616, "Prometheus"), (617, "Pandora"), (615, "Atlas"), (618, "Pan"), (612, "Helene"),
         (715, "Puck"), (712, "Portia"), (711, "Juliet"), (714, "Belinda"), (808, "Proteus"), (802, "Nereid"), (807, "Larissa"), (806, "Galatea"), (805, "Despina"),
         (902, "Nix"), (903, "Hydra"), (904, "Kerberos"), (905, "Styx")]


def tlist(keys: list[str]) -> str:
    fx = json.loads((FIX / "de440_state_vectors_v2physics.json").read_text())["snapshots"]
    return " ".join(f"{fx[k]['epoch']['jd_tdb']:.2f}" for k in keys)


def url(command: str, times: str) -> str:
    q = {"format": "json", "COMMAND": f"'{command}'", "OBJ_DATA": "'YES'", "MAKE_EPHEM": "'YES'", "EPHEM_TYPE": "'VECTORS'", "CENTER": "'500@0'", "TLIST": f"'{times}'", "TLIST_TYPE": "'JD'",
         "TIME_TYPE": "'TDB'", "VEC_TABLE": "'2'", "REF_PLANE": "'FRAME'", "REF_SYSTEM": "'ICRF'", "OUT_UNITS": "'KM-D'", "CSV_FORMAT": "'YES'"}
    return BASE + "?" + urllib.parse.urlencode(q, quote_via=urllib.parse.quote)


def group(name: str):
    """(directory, [(filename, url, expected-name regex)])"""
    all27 = sorted((k for k in json.loads((FIX / "de440_state_vectors_v2physics.json").read_text())["snapshots"] if k.startswith("j")), key=lambda k: int(k[1:]))
    if name == "asteroids":
        cat = json.loads((FIX / "de440_perturbers_v2physics.json").read_text())["asteroids"]
        top = sorted(cat, key=lambda a: -a["gm_km3_s2"])[:30]
        t = tlist(all27)
        return Path("D:/datasets/jpl/horizons_asteroids"), [(f"horizons_ast_{a['number']}.json", url(f"{a['number']};", t), rf"^\s*{a['number']}\b") for a in top]
    if name == "moons":
        t = tlist(all27)
        return Path("D:/datasets/jpl/horizons_moons"), [(f"horizons_moon_{i}_{n}.json", url(str(i), t), rf"^{re.escape(n)}\b") for i, n in MOONS]
    if name == "comet73p":
        out = []
        for w in ("j2005", "j2010", "j2015"):
            y = int(w[1:])
            out.append((f"horizons_73P_{w}.json", url("90000740", tlist([f"j{y + k}" for k in range(6)])), r"73P"))
        return Path("D:/datasets/jpl/horizons_comets"), out
    raise SystemExit(f"unknown group {name}")


def check(path: Path, expect: str) -> str | None:
    try:
        res = json.loads(path.read_text(encoding="utf-8"))["result"]
    except Exception as e:
        return f"unreadable: {e}"
    if "$$SOE" not in res or "$$EOE" not in res:
        return "no ephemeris block: " + res[:120].replace("\n", " ")
    m = re.search(r"Target body name:\s*(.+?)\s*\{", res)
    if m is None or not re.search(expect, m.group(1), re.I):
        return f"unexpected body: {m.group(1) if m else 'no name'} (wanted /{expect}/)"
    rows = res.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines()
    return None if len(rows) >= 6 else f"only {len(rows)} ephemeris rows"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("group", choices=["asteroids", "moons", "comet73p"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    d, reqs = group(a.group)
    d.mkdir(parents=True, exist_ok=True)
    (d / "rejected").mkdir(exist_ok=True)
    print(f"{a.group}: {len(reqs)} requests -> {d}", flush=True)
    if a.dry_run:
        print(reqs[0][1])
        return 0
    log, first = [], True
    for fname, u, expect in reqs:
        f = d / fname
        if f.exists() and check(f, expect) is None:
            log.append((fname, "kept (already good)"))
            continue
        tmp = d / (fname + ".part")
        r = subprocess.run(["curl", "-sS", "--max-time", "120", "-o", str(tmp), "-w", "%{http_code}", u], capture_output=True, text=True)
        err = None if r.stdout.strip() == "200" else f"HTTP {r.stdout.strip()} {r.stderr.strip()[:80]}"
        if err is None:
            err = check(tmp, expect)
        if err is None:
            if f.exists():                                           # replacing a response that failed its check: keep the old one for the record
                f.replace(d / "rejected" / (fname + ".old"))
            tmp.replace(f)
            log.append((fname, f"ok {f.stat().st_size} bytes"))
        else:
            if tmp.exists():
                tmp.replace(d / "rejected" / fname)
            log.append((fname, "REJECTED: " + err))
        print(*log[-1], flush=True)
        if first:
            first = False
            if log[-1][1].startswith("REJECTED"):
                print("the first request failed its check: stopping before sending the rest", flush=True)
                break
        time.sleep(0.5)
    (d / f"fetch_log_{a.group}.json").write_text(json.dumps(log, indent=1))
    bad = [x for x in log if x[1].startswith("REJECTED")]
    print(f"done: {len(log) - len(bad)} good, {len(bad)} rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
