"""The comet candidate ladder against JPL's own solutions (Horizons states stored under D:/datasets/jpl/horizons_comets/).

For every stored (comet, window): start from JPL's barycentric state at the window start, propagate with our model and compare with JPL's states at +1..+5 yr (the same solution).
Rungs: R0 Newton (Sun + nine planet systems from DE440s), R1 + the Sun's 1PN term, R2 + the Marsden non-gravitational push with the solution's A1, A2, A3 (DT ignored),
R3 + the solution's DT (the Marsden time delay), R3n = R3 without the Sun's 1PN, R4 = R3 + the pull of DE440's 343 asteroids. Each rung is a falsifiable addition: it stays only if it lowers the error on comets it was not chosen on.

usage: python scripts/comet_ladder.py [--dir D] [--out FILE.json] [--rungs R0 R1 R2 R3]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
FIXTURE = _ROOT / "sigma_ground/field/interface/fixtures/de440_state_vectors_v2physics.json"
KERNEL = Path("D:/datasets/jpl/de440s.bsp")


def gm_planets(snapshot: str = "j2010"):
    bodies = json.loads(FIXTURE.read_text())["snapshots"][snapshot]["bodies"]
    gm = {x["name"]: x["gm_km3_s2"] for x in bodies}
    anc = {x["name"]: x.get("gravitational_anchor") for x in bodies}
    sysgm = lambda n: gm[n] + sum(g for m, g in gm.items() if anc.get(m) == n)
    return [gm["Sun"], sysgm("Mercury"), sysgm("Venus"), gm["Earth"] + gm["Moon"], sysgm("Mars"), sysgm("Jupiter"), sysgm("Saturn"), sysgm("Uranus"), sysgm("Neptune"), sysgm("Pluto")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("D:/datasets/jpl/horizons_comets"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rungs", nargs="+", default=["R0", "R1", "R2", "R3"])
    a = ap.parse_args()
    from jplephem.spk import SPK
    from sigma_ground.field.interface import comet_orbits as co
    from sigma_ground.field.interface.comet_truth import parse_horizons_comet
    kernel = SPK.open(str(KERNEL))
    gmp = gm_planets()
    results = []
    files = sorted(glob.glob(str(a.dir / "horizons_*P_j*.json")), key=lambda p: (int(re.search(r"_(\d+)P_", p).group(1)), p))
    for f in files:
        des, win = re.search(r"horizons_(\d+P)_(j\d{4})", f).groups()
        try:
            t = parse_horizons_comet(f)
        except ValueError as e:
            print(f"skip {des} {win}: {e}")
            continue
        jd, S = t["jd"], t["state_km_kmday"]
        sun0 = np.asarray(kernel[0, 10].compute(jd[0]), dtype=float)
        rec = {"comet": des, "window": win, "solution": t["solution_source"], "arc": t["obs_arc"], "A": t["A"], "DT": t["DT"], "has_model": t["has_nongrav_model"],
               "r0_au": float(np.linalg.norm(S[0, :3] - sun0) / co.AU_KM), "err_km": {}}
        for rung in a.rungs:
            kw = dict(gr=rung not in ("R0", "R3n"))                                      # R3n = R3 without the Sun's 1PN term (does relativity matter?)
            if rung in ("R2", "R3", "R3n", "R4") and t["has_nongrav_model"]:
                kw.update(nongrav=t["A"], law=t["law"])
                if rung in ("R3", "R3n", "R4"):
                    kw.update(dt_days=t["DT"])
            if rung == "R4":                                                              # R3 + the pull of DE440's 343 asteroids, as in the planets' restatement
                from sigma_ground.field.interface import perturber_paths as pp
                from sigma_ground.field.interface.asteroid_perturbers import Perturber
                span = float(jd[-1] - jd[0]) + 800.0
                cat = [x for x in json.loads((_ROOT / "sigma_ground/field/interface/fixtures/de440_perturbers_v2physics.json").read_text())["asteroids"] if x["elements"]]
                pert = [Perturber(str(x["number"]), x["gm_km3_s2"], 0.0, *[x["elements"][k_] for k_ in ("epoch_jd_tdb", "a_au", "e", "i_deg", "om_deg", "w_deg", "ma_deg")]) for x in cat]
                nodes = pp.asteroid_window_paths(pert, kernel, np.array(gmp), jd[0] - 400.0, span, 1.0)
                kw.update(sources=pp.to_prescribed_sources(nodes, np.array([x["gm_km3_s2"] for x in cat]), 1.0), sources_epoch_jd=jd[0] - 400.0)
            t0 = time.time()
            try:
                out = co.propagate(S[0], jd[0], list(jd[1:]), kernel, gmp, **kw)
                rec["err_km"][rung] = [float(np.linalg.norm(out[j][:3] - S[i + 1, :3])) for i, j in enumerate(jd[1:])]
            except Exception as e:                                                            # a failed integration is data, not a reason to stop the ladder
                rec["err_km"][rung] = None
                rec.setdefault("failed", {})[rung] = f"{type(e).__name__}: {e}"[:120]
        rec["min_helio_au_at_epochs"] = float(min(np.linalg.norm(S[i, :3] - np.asarray(kernel[0, 10].compute(jd[i]), dtype=float)) for i in range(len(jd))) / co.AU_KM)
        results.append(rec)
        e = rec["err_km"]
        print(f"{des:5s} {win}  r0 {rec['r0_au']:6.2f} au  " + "  ".join(f"{k}: " + ("fail" if v is None else f"{v[-1]:10.1f}") for k, v in e.items()) + "   (+5 yr, km)", flush=True)
    if a.out:
        a.out.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
