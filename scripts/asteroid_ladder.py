"""Score the 30 heaviest DE440 asteroids as bodies: our propagation against JPL's own solution (Horizons states stored in fixtures/horizons_asteroids/).

For each (asteroid, window): start from JPL's barycentric state at the window start, propagate under the Sun and the nine planet systems (DE440s kernel) and compare with JPL's states at +1..+5 yr (the same solution;
the 27 yearly fixture epochs). Rungs: R0 Newton, R1 + the Sun's 1PN term, R4 + the pull of DE440's other asteroids as prescribed sources (the target's own mass is excluded: it cannot pull on itself).
Truth is JPL's solution as propagated by Horizons (DE441 planets, 16 massive asteroids): this measures agreement with JPL's dynamics for these bodies, not their real positions. The asteroids have no
non-gravitational section in their solutions, so none is modelled.

usage: python scripts/asteroid_ladder.py [--windows j2005 j2010 j2015] [--out FILE.json] [--rungs R0 R1 R4] [--only 1 4 2]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
FIX = _ROOT / "sigma_ground/field/interface/fixtures"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=FIX / "horizons_asteroids")
    ap.add_argument("--windows", nargs="+", default=["j2005", "j2010", "j2015"])
    ap.add_argument("--rungs", nargs="+", default=["R0", "R1", "R4"])
    ap.add_argument("--only", nargs="*", type=int, default=None, help="asteroid numbers to score (default: all 30 files)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    from jplephem.spk import SPK
    from comet_ladder import gm_planets
    from sigma_ground.field.interface import comet_orbits as co
    from sigma_ground.field.interface import perturber_paths as pp
    from sigma_ground.field.interface.asteroid_perturbers import Perturber
    from sigma_ground.field.interface.comet_truth import parse_horizons_comet
    from sigma_ground.field.interface.prescribed_sources import PrescribedSources
    kernel = SPK.open("D:/datasets/jpl/de440s.bsp")
    gmp = gm_planets()
    fixture = json.loads((FIX / "de440_state_vectors_v2physics.json").read_text())["snapshots"]
    cat = [x for x in json.loads((FIX / "de440_perturbers_v2physics.json").read_text())["asteroids"] if x["elements"]]
    numbers = [x["number"] for x in cat]
    pert = [Perturber(str(x["number"]), x["gm_km3_s2"], 0.0, *[x["elements"][k] for k in ("epoch_jd_tdb", "a_au", "e", "i_deg", "om_deg", "w_deg", "ma_deg")]) for x in cat]
    gm_all = np.array([x["gm_km3_s2"] for x in cat])
    truth = {}
    for f in sorted(glob.glob(str(a.dir / "horizons_ast_*.json"))):
        n = int(re.search(r"horizons_ast_(\d+)\.json", f).group(1))
        if a.only is None or n in a.only:
            truth[n] = parse_horizons_comet(f)
    rows = []
    t_all = time.time()
    for win in a.windows:
        jd0 = fixture[win]["epoch"]["jd_tdb"]
        table = None
        if "R4" in a.rungs:                                                   # the perturber paths once per window; masked per target
            nodes = pp.asteroid_window_paths(pert, kernel, np.array(gmp), jd0 - 5.0, 5 * 365.25 + 40.0, 1.0)
            table = pp.to_prescribed_sources(nodes, gm_all, 1.0)
        for n, t in sorted(truth.items()):
            i = int(np.argmin(np.abs(t["jd"] - jd0)))
            if abs(t["jd"][i] - jd0) > 1e-3 or i + 5 >= len(t["jd"]):
                continue
            targets = [float(j) for j in t["jd"][i + 1:i + 6]]
            rec = {"asteroid": n, "name": t["target"], "window": win, "gm_km3_s2": float(gm_all[numbers.index(n)]) if n in numbers else None, "err_km": {}}
            for rung in a.rungs:
                kw = dict(gr=rung != "R0")
                if rung == "R4":
                    gm_masked = table.gm.copy()
                    if n in numbers:
                        gm_masked[numbers.index(n)] = 0.0
                    kw.update(sources=PrescribedSources(gm_masked, table.t0, table.h, table.pos, table.vel), sources_epoch_jd=jd0 - 5.0)
                try:
                    out = co.propagate(t["state_km_kmday"][i], float(t["jd"][i]), targets, kernel, gmp, **kw)
                    rec["err_km"][rung] = [float(np.linalg.norm(out[j][:3] - t["state_km_kmday"][i + 1 + k][:3])) for k, j in enumerate(targets)]
                except Exception as e:                                         # a failed integration is data
                    rec["err_km"][rung] = None
                    rec.setdefault("failed", {})[rung] = f"{type(e).__name__}: {e}"[:100]
            rows.append(rec)
            e = rec["err_km"]
            print(f"{n:4d} {t['target'][:22]:22s} {win}  " + "  ".join(f"{k}: " + ("fail" if v is None else f"{v[-1]:10.2f}") for k, v in e.items()) + f"   (+5 yr, km)  [{time.time() - t_all:.0f}s]", flush=True)
    if a.out:
        a.out.write_text(json.dumps(rows, indent=1))
    ok = [r for r in rows if all(v is not None for v in r["err_km"].values())]
    print(f"\n{len(rows)} runs ({len(ok)} complete). Median km at +1 / +3 / +5 yr:")
    for rung in a.rungs:
        v = [r["err_km"][rung] for r in ok]
        if v:
            print(f"  {rung}: " + " / ".join(f"{statistics.median(x[i] for x in v):10.2f}" for i in (0, 2, 4)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
