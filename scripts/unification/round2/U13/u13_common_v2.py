"""U13 v2 shared helpers.  Imports the ORIGINAL runner module (u13_av_share) so that the Mentat leaves,
the scene geometry and evaluate() are byte-for-byte the ones the skeptics reproduced.

Every physical leaf lives in U.SRC (Mentat tool + args + value + source).  Nothing in the repo is modified.
"""
import copy
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import u13_av_share as U  # noqa: E402

from sigma_ground.dynamics.seeding import seed_rotating_cloud, to_continuum  # noqa: E402
from sigma_ground.dynamics.continuum import continuum_step, cfl_timestep  # noqa: E402
from sigma_ground.dynamics.vec import Vec3  # noqa: E402

OUT = HERE
# The two readings of the design's "vary h (1.0x, 1.5x, 2.0x)":
#   reading S (runner's): h = spacing x {1.0, 1.5, 2.0}   (default h0 = 1.5 spacing is the middle point)
#   reading H (alternative): h = h0 x {1.0, 1.5, 2.0} = spacing x {1.5, 2.25, 3.0}
READ_S = [1.0, 1.5, 2.0]
READ_H = [1.5, 2.25, 3.0]
HSET = sorted(set(READ_S + READ_H))            # h / spacing values evaluated at every h-scan state
HFINE = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]


def jdump(obj, name):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)


def spacing(n):
    return (4.0 / 3.0 * math.pi * U.R ** 3 / n) ** (1.0 / 3.0)


def scene_N(n=U.N, seed=U.SEED):
    d = spacing(n)
    bodies = seed_rotating_cloud(n, cloud_mass_kg=U.M, cloud_radius_m=U.R, omega_rad_s=0.0,
                                 particle_density_kg_m3=U.RHO_FE, jitter_frac=0.0, seed=seed)
    sc = to_continuum(bodies, material_key="iron", G=U.G, smoothing_h=1.5 * d, gravity_softening_m=U.EPS,
                      eos="isothermal", sound_speed_m_s=U.CS)
    return sc, d


def hscan(ps_live, d, mults=HSET, alpha=1.0):
    """Evaluate the virial sums at a FIXED particle state for each h = mult x spacing."""
    rows = []
    for hm in mults:
        ps = copy.deepcopy(ps_live)
        rows.append(dict(h_over_spacing=hm, h_m=hm * d, **U.evaluate(ps, hm * d, alpha)))
    return rows


def slope_of(rows, mults, key="ratio"):
    sel = [r for r in rows if r["h_over_spacing"] in mults]
    sel.sort(key=lambda r: r["h_over_spacing"])
    return U.slope([r["h_over_spacing"] for r in sel], [r[key] for r in sel])


def both_readings(rows):
    return {"reading_S_spacing_x_1_1.5_2": slope_of(rows, READ_S),
            "reading_H_h0_x_1_1.5_2": slope_of(rows, READ_H)}


def row_at(rows, hm):
    return [r for r in rows if r["h_over_spacing"] == hm][0]


def state_arrays(ps):
    m = np.array([p.mass for p in ps])
    x = np.array([[p.position.x, p.position.y, p.position.z] for p in ps])
    v = np.array([[p.velocity.x, p.velocity.y, p.velocity.z] for p in ps])
    return m, x, v


def radial_structure(ps):
    m, x, v = state_arrays(ps)
    xc = (m[:, None] * x).sum(0) / m.sum()
    vc = (m[:, None] * v).sum(0) / m.sum()
    dx, dv = x - xc, v - vc
    r = np.sqrt((dx ** 2).sum(1))
    vr = (dx * dv).sum(1) / r
    order = np.argsort(r)
    half = len(r) // 2
    inner, outer = order[:half], order[half:]
    k_fit = -float((m * (dx * dv).sum(1)).sum() / (m * r ** 2).sum())        # v = -k r fit
    v_tan = np.sqrt(np.maximum((dv ** 2).sum(1) - vr ** 2, 0.0))
    v_res = dv + k_fit * dx                                                  # residual about homologous fit
    return dict(frac_outward=float((vr > 0).mean()),
                k_inner_s=float(-(vr[inner] / r[inner]).mean()), k_outer_s=float(-(vr[outer] / r[outer]).mean()),
                k_fit_s=k_fit,
                v_hom_at_R_over_cs=k_fit * U.R / U.CS,
                v_hom_at_rmax_over_cs=k_fit * float(r.max()) / U.CS,
                v_rms_over_cs=float(np.sqrt((m * (dv ** 2).sum(1)).sum() / m.sum())) / U.CS,
                v_residual_rms_over_cs=float(np.sqrt((m * (v_res ** 2).sum(1)).sum() / m.sum())) / U.CS,
                v_tan_rms_over_cs=float(np.sqrt((m * v_tan ** 2).sum() / m.sum())) / U.CS,
                rms_radius_over_R=float(np.sqrt((m * r ** 2).sum() / m.sum())) / U.R)
