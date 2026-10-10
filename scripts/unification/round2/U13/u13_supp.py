"""U13 supplementary diagnostics -- POST-HOC (written after Run A/B showed h-slope 0.18-0.32).

S1  operator check: an exactly homologous velocity field v = -(k) r imposed on the seed
    (N = 64, 512); h-scan at alpha = 1.  For v_ij = -k r_ij the pair ratio Pi_ij / (P_i/rho_i^2 + P_j/rho_j^2)
    is alpha k h / c_s x (density weighting) -> slope ~ 1 by construction.  If S1 gives ~1, the
    h-slope failure of Run A/B is the FLOW (velocity structure on the kernel scale), not the operator.
S2  resolution: the real collapse (dt = t_ff/600, alpha = 1) at N = 64, 128, 256 to step 50 and 300,
    N = 512 to step 50; ratio and h-slope vs N (h = 1.5 (V/N)^(1/3) shrinks with N).
"""
import copy
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import u13_av_share as U  # noqa: E402  (module-level Mentat leaf calls; sources kept in U.SRC)
from sigma_ground.dynamics.seeding import seed_rotating_cloud, to_continuum  # noqa: E402
from sigma_ground.dynamics.continuum import continuum_step  # noqa: E402
from sigma_ground.dynamics.vec import Vec3  # noqa: E402


def scene_N(n):
    d = (4.0 / 3.0 * math.pi * U.R ** 3 / n) ** (1.0 / 3.0)
    bodies = seed_rotating_cloud(n, cloud_mass_kg=U.M, cloud_radius_m=U.R, omega_rad_s=0.0,
                                 particle_density_kg_m3=U.RHO_FE, jitter_frac=0.0, seed=U.SEED)
    sc = to_continuum(bodies, material_key="iron", G=U.G, smoothing_h=1.5 * d, gravity_softening_m=U.EPS,
                      eos="isothermal", sound_speed_m_s=U.CS)
    return sc, d


def hscan_N(ps_live, d, alpha=1.0):
    rows = []
    for hm in U.HMULTS:
        ps = copy.deepcopy(ps_live)
        rows.append(dict(h_mult=hm, h_m=hm * d, **U.evaluate(ps, hm * d, alpha)))
    return rows


def radial_structure(ps):
    m = np.array([p.mass for p in ps])
    x = np.array([[p.position.x, p.position.y, p.position.z] for p in ps])
    v = np.array([[p.velocity.x, p.velocity.y, p.velocity.z] for p in ps])
    xc = (m[:, None] * x).sum(0) / m.sum()
    vc = (m[:, None] * v).sum(0) / m.sum()
    dx, dv = x - xc, v - vc
    r = np.sqrt((dx ** 2).sum(1))
    vr = (dx * dv).sum(1) / r
    order = np.argsort(r)
    half = len(r) // 2
    inner, outer = order[:half], order[half:]
    return dict(frac_outward=float((vr > 0).mean()),
                k_inner_s=float(-(vr[inner] / r[inner]).mean()), k_outer_s=float(-(vr[outer] / r[outer]).mean()),
                vr_over_cs_outer_mean=float(vr[outer].mean() / U.CS))


def s1(n, k=1.0e-5):
    sc, d = scene_N(n)
    ps = sc.continuum_parcels()
    m = np.array([p.mass for p in ps])
    x = np.array([[p.position.x, p.position.y, p.position.z] for p in ps])
    xc = (m[:, None] * x).sum(0) / m.sum()
    for p in ps:
        p.velocity = Vec3(-k * (p.position.x - xc[0]), -k * (p.position.y - xc[1]), -k * (p.position.z - xc[2]))
    rows = hscan_N(ps, d)
    return dict(N=n, k_imposed_s=k, two_h_over_R_at_1p5=3 * d / U.R, rows=rows,
                h_slope=U.slope([r["h_mult"] for r in rows], [r["ratio"] for r in rows]),
                ratio_over_alpha_k_h_over_2cs=[r["ratio"] / (1.0 * k * r["h_m"] / (2 * U.CS)) for r in rows])


def s2(n, steps):
    sc, d = scene_N(n)
    dt = U.T_FF / 600.0
    out, t0 = [], time.time()
    for st in range(max(steps) + 1):
        if st in steps:
            ps = sc.continuum_parcels()
            rows = hscan_N(ps, d)
            out.append(dict(step=st, t_over_tff=sc.time / U.T_FF,
                            ratio_alpha1=[r for r in rows if r["h_mult"] == 1.5][0]["ratio"],
                            h_slope=U.slope([r["h_mult"] for r in rows], [r["ratio"] for r in rows]),
                            structure=radial_structure(ps), rows=rows))
            print(json.dumps(dict(N=n, step=st, ratio=out[-1]["ratio_alpha1"], h_slope=out[-1]["h_slope"],
                                  structure=out[-1]["structure"], wall=time.time() - t0)), flush=True)
        if st == max(steps):
            break
        continuum_step(sc, dt, alpha_visc=1.0)
    return dict(N=n, h_m=1.5 * d, two_h_over_R=3 * d / U.R, samples=out, wall_s=time.time() - t0)


if __name__ == "__main__":
    res = {"note": "POST-HOC supplementary diagnostics (not pre-registered)", "mentat_sources": U.SRC}
    res["S1"] = [s1(64), s1(512)]
    for r in res["S1"]:
        print("S1", r["N"], r["h_slope"], r["ratio_over_alpha_k_h_over_2cs"], flush=True)
    res["S2"] = [s2(64, (50, 300)), s2(128, (50, 300)), s2(256, (50, 300)), s2(512, (50,))]
    with open(os.path.join(U.OUT, "u13_supp_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("done", flush=True)
