"""U13 v2 resolution diagnostics (post hoc; predictions for the new runs are in u13_prereg_v2.json).

S1  operator identity check: imposed homologous field v = -k r on the seed (N = 64 fine h-scan, N = 512 h-scan,
    both readings).  [BY CONSTRUCTION: the engine's mu-form Pi gives ratio = (alpha k h / 2 c_s) f_eta in the
    continuum; see u13_derive_v2.py]
S2  the real collapse (dt = t_ff/600, alpha = 1, h = 1.5 spacing) at N = 64, 128, 256, 512 out to 1 t_ff,
    samples at steps 50, 300, 600 with both h-scan readings, the kinematic coefficient C_kin and flow structure.
Usage:  python -X utf8 u13_supp_v2.py small   -> N = 64, 128, 256 and S1   -> u13_supp_small_v2.json
        python -X utf8 u13_supp_v2.py big     -> N = 512 S2                 -> u13_supp_big_v2.json
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from u13_common_v2 import (U, HSET, HFINE, READ_S, READ_H, jdump, scene_N, hscan, both_readings,  # noqa: E402
                           slope_of, row_at, radial_structure, state_arrays, continuum_step, Vec3)

STEPS = (50, 300, 600)


def s1(n, mults, k=1.0e-5):
    sc, d = scene_N(n)
    ps = sc.continuum_parcels()
    m, x, v = state_arrays(ps)
    xc = (m[:, None] * x).sum(0) / m.sum()
    for p in ps:
        p.velocity = Vec3(-k * (p.position.x - xc[0]), -k * (p.position.y - xc[1]), -k * (p.position.z - xc[2]))
    rows = hscan(ps, d, mults)
    return dict(N=n, k_imposed_s=k, rows=rows, slopes=both_readings(rows),
                slope_all=slope_of(rows, mults),
                ratio_over_alpha_k_h_over_2cs={str(r["h_over_spacing"]): r["ratio"] / (k * r["h_m"] / (2 * U.CS))
                                               for r in rows},
                C_implied={str(r["h_over_spacing"]): r["ratio"] / (k * r["h_m"] / (2 * U.CS)) / 6.0 for r in rows})


def s2(n):
    sc, d = scene_N(n)
    dt = U.T_FF / 600.0
    out, t0 = [], time.time()
    for st in range(max(STEPS) + 1):
        if st in STEPS:
            ps = sc.continuum_parcels()
            rows = hscan(ps, d, HSET)
            r15 = row_at(rows, 1.5)
            out.append(dict(step=st, t_over_tff=sc.time / U.T_FF, ratio_alpha1=r15["ratio"], C_kin=r15.get("C_kin"),
                            div_fit_s=r15["div_fit_s"], slopes=both_readings(rows),
                            structure=radial_structure(ps), rows=rows, wall_s=time.time() - t0))
            print(json.dumps(dict(N=n, step=st, ratio=r15["ratio"], C=r15.get("C_kin"), slopes=out[-1]["slopes"],
                                  wall=round(time.time() - t0, 1))), flush=True)
        if st == max(STEPS):
            break
        continuum_step(sc, dt, alpha_visc=1.0)
    led = sc.energy_ledger()
    return dict(N=n, spacing_m=d, h_m=1.5 * d, two_h_over_R=3 * d / U.R, samples=out, energy_ledger_end=led,
                wall_s=time.time() - t0)


if __name__ == "__main__":
    part = sys.argv[1]
    res = {"note": "POST-HOC resolution diagnostics, v2", "part": part, "mentat_sources": U.SRC}
    if part == "small":
        res["S1"] = [s1(64, HFINE), s1(512, HSET)]
        for r in res["S1"]:
            print("S1", r["N"], r["slopes"], flush=True)
        res["S2"] = [s2(64), s2(128), s2(256)]
        jdump(res, "u13_supp_small_v2.json")
    else:
        res["S2"] = [s2(512)]
        jdump(res, "u13_supp_big_v2.json")
    print("done", flush=True)
