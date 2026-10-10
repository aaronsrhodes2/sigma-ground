"""U7 repair pass (v2) — traceable numbers the skeptics asked for. Writes u7_repairs_v2.json. Engine unmodified.

A. 7^3 literal lattice, uniform bulk mass, t = 0: summation-density residual rho/rho0 - 1 (= P/K for the engine's
   linear Tait) at face centre, edge centre, corner, and the surface-layer mean.          [VERIFIED numerics, engine W]
B. Surface-layer IC masses of the 27x11x11 bar (and the v2 15x15 / 21x21 bars): range, massless and heavy counts.
C. Lindemann melting flag: material_properties('water_ice', T) and ContinuumParcel.phase at 200 / 250 / 300 K against
   the cascade's own PHASE_DATA T_melt_K and Mentat melting_point(water).
D. Shear-free Tait speed against the cascade's longitudinal speed: c_Tait / c_L.
E. Heat, literal 1e-3 threshold: continuum crossing times at d = 2, 3, 4 (point source, D = 1) against the measured
   ones, and the continuum absolute-threshold exponent p = 2 / (1 - 3/(2L)), derived symbolically by Mentat.
F. Plane- and point-source signal size against the reference background on the axis of the 27x11x11 bar,
   including the axis summation density (P/K) of the reference run at every step.
"""
import json, math, sys
import numpy as np
from scipy.optimize import brentq
from scipy.spatial import cKDTree
REPO = r"D:/Aaron/development/sigma-ground"
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + "/tools")
OUT = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U7"
sys.path.insert(0, OUT)
from mentat_client import mentat
from sigma_ground.dynamics.fluid.kernel import W
from sigma_ground.dynamics.continuum import material_properties, ContinuumParcel
from sigma_ground.dynamics.vec import Vec3
from sigma_ground.field.interface.phase_transition import PHASE_DATA
import u7_sph_analyse as SA

res = {}
s = 0.01; h = 1.5 * s

# ── A. 7^3 uniform-mass lattice at t = 0 ──
pr = material_properties("water_ice", 300.0)
rho0 = pr["density_kg_m3"]
R = int(math.ceil(2 * h / s)) + 1
m_bulk = rho0 / sum(W(s * math.sqrt(i * i + j * j + k * k), h)
                    for i in range(-R, R + 1) for j in range(-R, R + 1) for k in range(-R, R + 1))   # = u7_sph.bulk_mass
n7 = 7
ijk7 = np.array([(i, j, k) for i in range(n7) for j in range(n7) for k in range(n7)])
P7 = ijk7 * s
tree = cKDTree(P7)
rho7 = np.zeros(len(P7))
for a in range(len(P7)):
    for b in tree.query_ball_point(P7[a], 2 * h):
        rho7[a] += m_bulk * W(float(np.linalg.norm(P7[a] - P7[b])), h)
res7 = rho7 / rho0 - 1
depth7 = np.minimum.reduce([ijk7[:, 0], n7 - 1 - ijk7[:, 0], ijk7[:, 1], n7 - 1 - ijk7[:, 1], ijk7[:, 2], n7 - 1 - ijk7[:, 2]])
def at(i, j, k):
    return float(res7[np.nonzero(np.all(ijk7 == (i, j, k), axis=1))[0][0]])
res["A_lit7_uniform_mass_t0"] = dict(
    m_bulk=m_bulk, face_centre=at(3, 3, 0), edge_centre=at(3, 0, 0), corner=at(0, 0, 0), centre=at(3, 3, 3),
    surface_layer_mean=float(res7[depth7 == 0].mean()), surface_layer_min=float(res7[depth7 == 0].min()),
    depth1_mean=float(res7[depth7 == 1].mean()), depth2_mean=float(res7[depth7 == 2].mean()),
    note="rho/rho0 - 1 = P/K for the engine's linear Tait P = K(rho/rho0 - 1); the 7^3 lattice has one bulk parcel "
         "(depth 3) and it reads exactly rho0.")

# ── B. IC masses ──
def mass_stats(tag):
    m = np.load(f"{OUT}/masses_{tag}.npy")
    info = json.load(open(f"{OUT}/masses_{tag}.json"))
    r = m / info["m_bulk"]
    return dict(dims=info["dims"], n=int(len(m)), min_over_m_bulk=float(r.min()), max_over_m_bulk=float(r.max()),
                n_exactly_zero=int((r == 0).sum()), n_below_1e_6=int((r < 1e-6).sum()), n_below_1e_3=int((r < 1e-3).sum()),
                n_below_0p1=int((r < 0.1).sum()), n_above_2=int((r > 2).sum()), n_above_5=int((r > 5).sum()),
                res_max=info["res_max"], uniform_mass_res_max=info["uniform_mass_res_max"])
res["B_ic_masses"] = {t: mass_stats(t) for t in ("bar27_water_ice", "v2_bar27x15_water_ice", "v2_bar27x21_water_ice")}

# ── C. Lindemann flag ──
Tm_mentat, Tm_src = mentat("melting_point", {"material": "water"})
lind = {}
for T in (200.0, 250.0, 300.0):
    p = ContinuumParcel("water_ice", 0.005, temperature=T, position=Vec3(0, 0, 0), velocity=Vec3(0, 0, 0))
    p.update_properties()
    lind[f"{T:g}K"] = dict(melting_point_K_cascade=p.props["melting_point_K"], phase_flag=p.phase,
                           real_phase_at_1atm=("solid (ice)" if T < Tm_mentat else "liquid"))
res["C_lindemann_water_ice"] = dict(by_T=lind, PHASE_DATA_T_melt_K=PHASE_DATA["water_ice"]["T_melt_K"],
                                    mentat_melting_point_water=dict(value=Tm_mentat, source=Tm_src),
                                    note="continuum.material_properties (continuum.py:122) takes T_melt from "
                                         "lindemann_melting_estimate (phase_transition.py:422) although PHASE_DATA "
                                         "(phase_transition.py:306-308) carries the measured 273.15 K")

# ── D. c_Tait / c_L ──
res["D_shear_free_speed"] = {}
for mat in ("water_ice", "iron"):
    q = material_properties(mat, 300.0)
    ct = math.sqrt(q["bulk_modulus_Pa"] / q["density_kg_m3"])
    res["D_shear_free_speed"][mat] = dict(c_tait=ct, c_L=q["sound_speed_m_s"], c_tait_over_c_L=ct / q["sound_speed_m_s"],
                                         c_L_over_c_tait=q["sound_speed_m_s"] / ct,
                                         c_shear=math.sqrt(q["shear_modulus_Pa"] / q["density_kg_m3"]))

# ── E. heat, literal threshold: continuum expectation and the Mentat-derived exponent ──
H = json.load(open(f"{OUT}/heat_results.json"))
HR = json.load(open(f"{OUT}/heat_refine_results.json"))
G3 = lambda d, t: (4 * math.pi * t) ** -1.5 * math.exp(-d * d / (4 * t))
cont = {}
for d in (2, 3, 4):
    tc = brentq(lambda t: G3(d, t) - 1e-3, 1e-4, d * d / 6.0)
    cont[d] = dict(t_continuum=tc, L=d * d / (4 * tc), p_local_continuum=2.0 / (1 - 3.0 / (2 * d * d / (4 * tc))),
                   t_measured_slices=H["literal"]["arrivals"][str(d)],
                   t_measured_every_substep=H["threshold_sweep"]["0.001"]["t"][str(d)])
dd = np.array([2.0, 3.0, 4.0])
p_cont_234 = float(np.polyfit(np.log(dd), np.log([cont[d]["t_continuum"] for d in (2, 3, 4)]), 1)[0])
F = "log(Am/th) - 3*log(4*pi*Dc*exp(u))/2 - exp(2*v)/(4*Dc*exp(u))"      # u = ln t, v = ln d, implicit F = 0
Fv, sFv = mentat("differentiate_expr", {"expression": F, "variable": "v"})
Fu, sFu = mentat("differentiate_expr", {"expression": F, "variable": "u"})
pexpr, sp = mentat("simplify_expr", {"expression": f"-({Fv})/({Fu})"})
# substitute exp(2v) = 4 Dc L exp(u), i.e. L = d^2/(4 D t), and compare with 2/(1 - 3/(2L)); 0 means identical
chk2, schk2 = mentat("simplify_expr", {"expression": f"({pexpr.replace('exp(2*v)', '(4*Dc*L*exp(u))')}) - 2/(1 - 3/(2*L))"})
res["E_heat_literal"] = dict(
    continuum_crossings=cont, p_continuum_d2_4=p_cont_234,
    p_measured_slices=H["literal"]["p"], p_measured_every_substep=H["threshold_sweep"]["0.001"]["p"],
    analyst_expectation_prerun="p ~ 3.4 from continuum crossings at t ~ 0.18, 0.57, 1.8",
    half_peak_d2=dict(dx1=HR["dx=1"]["rows"]["2"]["t_half"], dx05=HR["dx=0.5"]["rows"]["2"]["t_half"],
                      continuum=HR["continuum"]["2"]["t_half"]),
    mentat_derivation=dict(F_equals_zero=F, dF_dv=dict(value=Fv, source=sFv), dF_du=dict(value=Fu, source=sFu),
                           p_equals_minus_Fv_over_Fu=dict(value=pexpr, source=sp),
                           p_minus_2_over_1_minus_3_over_2L=dict(value=chk2, source=schk2)))

# ── F. signal vs reference background on the axis, 27x11x11 bar ──
def axis_density_series(tag, dmax=10):
    z = np.load(f"{OUT}/sph_{tag}.npz")
    X, m = z["X"], z["masses"]
    ijk = [tuple(int(v) for v in x) for x in z["ijk"]]; c = tuple(int(v) for v in z["centre"])
    ks = [ijk.index((c[0] + d, c[1], c[2])) for d in range(0, dmax + 1)]
    out = np.zeros((X.shape[0], len(ks)))
    for n in range(X.shape[0]):
        tr = cKDTree(X[n])
        for a, k in enumerate(ks):
            out[n, a] = sum(m[b] * W(float(np.linalg.norm(X[n, k] - X[n, b])), h) for b in tr.query_ball_point(X[n, k], 2 * h))
    return out / rho0 - 1.0
bg = axis_density_series("bar_w_ref")
z_r = np.load(f"{OUT}/sph_bar_w_ref.npz"); z_pl = np.load(f"{OUT}/sph_bar_w_plane.npz"); z_pt = np.load(f"{OUT}/sph_bar_w_pert.npz")
ijk = [tuple(int(v) for v in x) for x in z_r["ijk"]]; c = tuple(int(v) for v in z_r["centre"])
sig = {}
for d in range(1, 11):
    k = ijk.index((c[0] + d, c[1], c[2]))
    ypl = np.abs(z_pl["V"][:, k, 0] - z_r["V"][:, k, 0]); ypt = np.abs(z_pt["V"][:, k, 0] - z_r["V"][:, k, 0])
    jpl, jpt = SA.first_peak(ypl), SA.first_peak(ypt)
    sig[d] = dict(plane_first_peak_m_s=float(ypl[jpl]), point_first_peak_m_s=float(ypt[jpt]),
                  ref_speed_max_to_plane_peak=float(np.linalg.norm(z_r["V"][: jpl + 1, k], axis=1).max()))
res["F_background_27x11x11_a1"] = dict(
    axis_P_over_K_ref_min=float(bg.min()), axis_P_over_K_ref_max=float(bg.max()),
    axis_P_over_K_ref_t0=bg[0].tolist(),
    ref_axis_speed_max=float(max(np.linalg.norm(z_r["V"][:, ijk.index((c[0] + d, c[1], c[2]))], axis=1).max() for d in range(0, 11))),
    signal_by_d=sig)

json.dump(res, open(f"{OUT}/u7_repairs_v2.json", "w"), indent=1, default=float)
print(json.dumps({k: (v if k != "F_background_27x11x11_a1" else {kk: vv for kk, vv in v.items() if kk != "axis_P_over_K_ref_t0"})
                  for k, v in res.items()}, indent=1, default=float))
