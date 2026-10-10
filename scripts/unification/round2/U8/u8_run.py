"""U8 - Range and extensivity (design_synthesis.json experiments_ranked id U8).

Part A  extensivity at fixed density: |U(N)| (phase_e.diagnostics pe_j, unsoftened pair sum)
        and Mentat gravitational_binding_energy vs pressure energy E_P = M c_s^2 and the
        engine's own sum m P/rho (isothermal; Tait exploratory); short-range control =
        kernel pair energy sum_i m_i rho_i^SPH.  N = 32..512, seed 7 (+ seeds 1..10).
Part B  range: central parcel mass x 1.1 at N = 256 (design) and N = 2048 (declared
        supplement); gravity and SPH pressure acceleration responses vs r.
Part C  virial-balance crossing N* of the two laws (the Jeans locus up to an O(1) factor).

Read-only on the repo; outputs only in this folder.  Run from the repo root.
"""
import sys, os, json, math, time
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
sys.path.insert(0, r"D:/Aaron/development/sigma-ground")
os.environ.setdefault("MSYS_NO_PATHCONV", "1")
import numpy as np
from mentat_client import mentat

from sigma_ground.dynamics.seeding import seed_rotating_cloud, to_continuum
from sigma_ground.dynamics.continuum import (_sph_density_sum, _sph_pressure_force,
                                             _pressure_isothermal, _pressure_from_eos)
from sigma_ground.dynamics.phase_e import diagnostics, _sph_state
from sigma_ground.field.interface.mechanical import bulk_modulus as engine_bulk_modulus

HERE = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U8"
T0 = time.time()
SRC = {}


def M(tool, args, key):
    v, s = mentat(tool, args)
    SRC[key] = {"tool": tool, "args": args, "value": v, "source": s}
    return v


# -- Mentat leaves -------------------------------------------------------------
G = float(M("lookup_constant", {"name": "G"}, "G"))
RHO = float(M("density", {"material": "iron"}, "rho_iron"))
STRAIN = -1e-3
ea = M("elastic_analysis", {"material_key": "iron", "strain": STRAIN}, "elastic_analysis_iron")
K_MENTAT = -float(ea["hydrostatic_stress_Pa"]) / STRAIN        # P = -K dV/V  (elasticity.py:151)
K_ENGINE = engine_bulk_modulus("iron")                          # what the Tait EOS uses
CS = float(M("simplify_expr", {"expression": f"sqrt({K_MENTAT:.10e}/{RHO})"}, "c_s_bulk"))
R_EARTH = float(M("solar_system_body", {"body_name": "Earth"}, "earth")["radius_m"])

M_P = 1.0                                   # kg per parcel  [Choice]
KH = 1.2                                    # h = 1.2 d      [engine default k]
D = (M_P / RHO) ** (1.0 / 3.0)              # mean spacing at fixed density
H = KH * D
EPS_DEFAULT = 0.5 * H                       # ContinuumScene default softening
NS = [32, 64, 128, 256, 512]
SEEDS = list(range(1, 11))


def cloud_R(N):
    return (3.0 * N * M_P / (4.0 * math.pi * RHO)) ** (1.0 / 3.0)


def make(N, seed):
    bodies = seed_rotating_cloud(N, cloud_mass_kg=N * M_P, cloud_radius_m=cloud_R(N),
                                 omega_rad_s=0.0, particle_density_kg_m3=RHO, seed=seed)
    scene = to_continuum(bodies, material_key="iron", G=G, smoothing_h=H,
                         eos="isothermal", sound_speed_m_s=CS)
    return bodies, scene


def fit(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.abs(np.asarray(y, float)))
    A_ = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A_, y, rcond=None)
    n = len(x)
    s2 = float(np.sum((y - A_ @ coef) ** 2) / max(n - 2, 1))
    se = math.sqrt(s2 / float(np.sum((x - x.mean()) ** 2)))
    return float(coef[0]), float(coef[1]), se


# -- Part A: extensivity -------------------------------------------------------
A = {"N": NS, "R_m": [cloud_R(n) for n in NS], "two_h_over_R": [2 * H / cloud_R(n) for n in NS]}
U_eng, U_soft, U_men, EP, sumPrho_iso, sumPrho_tait, kern = [], [], [], [], [], [], []
for N in NS:
    bodies, scene = make(N, 7)
    ms, ps, vs = _sph_state(scene)
    U_eng.append(diagnostics(ms, ps, vs, G)["pe_j"])
    U_soft.append(scene.energy_ledger()["self_gravity"])           # eps = 0.5 h (engine default)
    U_men.append(float(M("gravitational_binding_energy",
                         {"mass_kg": N * M_P, "radius_m": cloud_R(N)}, f"U_mentat_N{N}")))
    EP.append(sum(ms) * CS * CS)
    par = scene.continuum_parcels()
    _sph_density_sum(par, H)
    for p in par:
        _pressure_isothermal(p, CS)
    sumPrho_iso.append(sum(p.mass * p.pressure / p.sph_density for p in par))
    kern.append(sum(p.mass * p.sph_density for p in par))
    for p in par:
        _pressure_from_eos(p)
    sumPrho_tait.append(sum(p.mass * p.pressure / p.sph_density for p in par))
    print(f"A N={N}: U_eng={U_eng[-1]:.4e} U_men={U_men[-1]:.4e} EP={EP[-1]:.4e} "
          f"kern={kern[-1]:.4e} tait={sumPrho_tait[-1]:.4e}  t={time.time()-T0:.1f}s", flush=True)

A.update({"U_engine_pe_j": U_eng, "U_engine_softened_0.5h": U_soft, "U_mentat": U_men,
          "E_P_Mcs2": EP, "sum_mP_over_rho_isothermal": sumPrho_iso,
          "sum_mP_over_rho_tait": sumPrho_tait, "kernel_pair_energy_sum_m_rho": kern,
          "U_engine_over_mentat": [abs(a) / b for a, b in zip(U_eng, U_men)],
          "expected_ratio_(N-1)/N": [(n - 1) / n for n in NS],
          "isothermal_sum_rel_dev_from_Mcs2": [s / e - 1 for s, e in zip(sumPrho_iso, EP)]})
for key, arr in (("U_engine_pe_j", U_eng), ("U_engine_softened_0.5h", U_soft), ("U_mentat", U_men),
                 ("E_P_Mcs2", EP), ("sum_mP_over_rho_isothermal", sumPrho_iso),
                 ("kernel_pair_energy_sum_m_rho", kern), ("sum_mP_over_rho_tait", sumPrho_tait)):
    if all(a != 0 for a in arr) and (all(a > 0 for a in arr) or all(a < 0 for a in arr)):
        s, b, se = fit(NS, arr)
        A[f"slope_{key}"] = s
        A[f"slope_se_{key}"] = se
        A[f"local_slopes_{key}"] = [math.log(abs(arr[i + 1] / arr[i])) / math.log(NS[i + 1] / NS[i])
                                    for i in range(len(NS) - 1)]
    else:
        A[f"slope_{key}"] = None
        A[f"sign_change_{key}"] = arr

# multi-seed noise of the U slope and the control slope
ms_slopes, ms_kern_slopes, ms_ratio = [], [], []
for sd in SEEDS:
    ue, kk, rr = [], [], []
    for N in NS:
        bodies = seed_rotating_cloud(N, cloud_mass_kg=N * M_P, cloud_radius_m=cloud_R(N),
                                     omega_rad_s=0.0, particle_density_kg_m3=RHO, seed=sd)
        ue.append(diagnostics([b.mass_kg for b in bodies], [b.position_m for b in bodies],
                              [b.velocity_m_s for b in bodies], G)["pe_j"])
        _, scene = make(N, sd)
        par = scene.continuum_parcels()
        _sph_density_sum(par, H)
        kk.append(sum(p.mass * p.sph_density for p in par))
        rr.append(abs(ue[-1]) / U_men[NS.index(N)])
    ms_slopes.append(fit(NS, ue)[0])
    ms_kern_slopes.append(fit(NS, kk)[0])
    ms_ratio.append(rr)
print(f"multi-seed done t={time.time()-T0:.1f}s", flush=True)
A["multiseed"] = {"seeds": SEEDS, "slope_U": ms_slopes, "slope_U_mean": float(np.mean(ms_slopes)),
                  "slope_U_sd": float(np.std(ms_slopes, ddof=1)),
                  "slope_kernel_control": ms_kern_slopes,
                  "slope_kernel_mean": float(np.mean(ms_kern_slopes)),
                  "slope_kernel_sd": float(np.std(ms_kern_slopes, ddof=1)),
                  "U_ratio_mean_per_N": [float(v) for v in np.mean(ms_ratio, axis=0)],
                  "U_ratio_sd_per_N": [float(v) for v in np.std(ms_ratio, axis=0, ddof=1)]}


# -- Part B: range -------------------------------------------------------------
def vsub(a, b):
    return (a.x - b.x, a.y - b.y, a.z - b.z)


def range_run(N):
    bodies, scene = make(N, 7)
    par = scene.continuum_parcels()
    n = len(par)
    X = np.array([[p.position.x, p.position.y, p.position.z] for p in par])
    mass = np.array([p.mass for p in par])
    bc = (mass[:, None] * X).sum(0) / mass.sum()
    c = int(np.argmin(np.linalg.norm(X - bc, axis=1)))
    r = np.linalg.norm(X - X[c], axis=1)
    out = {"N": N, "R_m": cloud_R(N), "h_m": H, "R_over_h": cloud_R(N) / H,
           "central_index": c,
           "central_offset_from_barycentre_over_h": float(np.linalg.norm(X[c] - bc) / H),
           "r_over_h": [float(v) for v in r / H], "max_r_over_h": float(r.max() / H),
           "n_beyond_2h": int(np.sum(r > 2 * H)), "n_beyond_4h": int(np.sum(r > 4 * H))}
    m0 = par[c].mass
    dm = 0.1 * m0
    pres_store = {}
    for tag, eps in (("eps0", None), ("eps0.5h", EPS_DEFAULT)):
        res = {}
        for state in ("base", "pert"):
            par[c].mass = m0 * (1.1 if state == "pert" else 1.0)
            g = _sph_density_sum(par, H, gravity_G=G, softening=eps)
            for p in par:
                _pressure_isothermal(p, CS)
            a_iso = _sph_pressure_force(par, H)
            for p in par:
                _pressure_from_eos(p)
            a_tait = _sph_pressure_force(par, H)
            res[state] = (g, a_iso, a_tait)
        par[c].mass = m0
        dg = np.array([vsub(res["pert"][0][i], res["base"][0][i]) for i in range(n)])
        diso = np.array([vsub(res["pert"][1][i], res["base"][1][i]) for i in range(n)])
        dtait = np.array([vsub(res["pert"][2][i], res["base"][2][i]) for i in range(n)])
        mag_g = np.linalg.norm(dg, axis=1)
        e2 = (eps or 0.0) ** 2
        mask = np.arange(n) != c
        analytic = np.zeros(n)
        analytic[mask] = G * dm * r[mask] / (r[mask] ** 2 + e2) ** 1.5
        w = mask & (r >= 2 * H) & (r <= cloud_R(N))
        s, b, se = fit(r[w], mag_g[w])
        # log-binned medians (design: "bin delta-a by r")
        edges = np.geomspace(2 * H, cloud_R(N), 7)
        bc_r, bc_a = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = w & (r >= lo) & (r < hi)
            if sel.sum() >= 3:
                bc_r.append(float(np.median(r[sel])))
                bc_a.append(float(np.median(mag_g[sel])))
        sb = fit(bc_r, bc_a)[0] if len(bc_r) >= 3 else None
        w4 = mask & (r >= 4 * H) & (r <= cloud_R(N))
        s4 = fit(r[w4], mag_g[w4]) if w4.sum() >= 5 else None
        rel = float(np.max(np.abs(mag_g[mask] / analytic[mask] - 1)))
        res_out = {"gravity_slope_2h_R": s, "gravity_slope_se": se, "n_fit": int(w.sum()),
                   "gravity_slope_binned_2h_R": sb, "binned_r": bc_r, "binned_dA": bc_a,
                   "gravity_slope_4h_R": (s4[0] if s4 else None), "n_fit_4h": int(w4.sum()),
                   "gravity_max_rel_dev_from_analytic_pair_term": rel,
                   "gravity_dA": [float(v) for v in mag_g]}
        for name, d in (("isothermal", diso), ("tait", dtait)):
            mag = np.linalg.norm(d, axis=1)
            nz = mag != 0.0
            far2, far4 = r > 2 * H, r > 4 * H
            res_out[f"pressure_{name}"] = {
                "max_dA": float(mag.max()),
                "n_nonzero_total": int(nz.sum()),
                "n_nonzero_beyond_2h": int(np.sum(nz & far2)),
                "n_beyond_2h": int(np.sum(far2)),
                "max_dA_beyond_2h_over_max": (float(mag[far2].max() / mag.max()) if np.any(far2) else None),
                "n_nonzero_beyond_4h": int(np.sum(nz & far4)),
                "n_beyond_4h": int(np.sum(far4)),
                "max_r_nonzero_over_h": float(r[nz].max() / H),
                "identically_zero_beyond_2h": bool(not np.any(nz & far2)),
                "identically_zero_beyond_4h": bool(not np.any(nz & far4)),
                "dA": [float(v) for v in mag]}
            pres_store.setdefault(name, []).append(mag)
        out[tag] = res_out
        pi_ = res_out["pressure_isothermal"]
        print(f"B N={N} {tag}: grav slope [2h,R]={s:.5f} binned={sb} [4h,R]={s4[0] if s4 else None} "
              f"rel_dev={rel:.2e} | iso nz>2h={pi_['n_nonzero_beyond_2h']}/{pi_['n_beyond_2h']} "
              f"nz>4h={pi_['n_nonzero_beyond_4h']}/{pi_['n_beyond_4h']} "
              f"rmax_nz={pi_['max_r_nonzero_over_h']:.3f}h  t={time.time()-T0:.1f}s", flush=True)
    out["pressure_identical_across_eps"] = {k: bool(np.array_equal(v[0], v[1])) for k, v in pres_store.items()}
    out["dm_kg"] = dm
    return out


B = {"N256": range_run(256), "N2048": range_run(2048)}

# -- Part C: virial-balance crossing ------------------------------------------
lnA_U = float(np.mean(np.log(np.abs(U_eng)) - (5.0 / 3.0) * np.log(NS)))   # fixed-slope prefactor
A_P = M_P * CS * CS
N_star_engine = math.exp((math.log(A_P) - lnA_U) / (2.0 / 3.0))
M("solve_equation", {"equation": "4*pi*G*rho*R^2/5 = c^2", "variable": "R"}, "Rstar_symbolic")
R_star = float(M("simplify_expr",
                 {"expression": f"N(sqrt(5)*{CS:.10e}/(2*sqrt(pi*{G:.10e}*{RHO})))"}, "Rstar_value"))
ratio_J = M("simplify_expr", {"expression": "sqrt(5/(4*pi))/sqrt(pi)"}, "Rstar_over_lambdaJ")
N_star_formula = (4.0 / 3.0) * math.pi * RHO * R_star ** 3 / M_P
C = {"A_U_fixed_slope": math.exp(lnA_U), "A_P": A_P,
     "N_star_from_engine_fits": N_star_engine, "N_star_from_Mentat_formula": N_star_formula,
     "N_star_ratio_engine_over_formula": N_star_engine / N_star_formula,
     "R_star_m": R_star, "R_star_over_R_earth": R_star / R_EARTH,
     "R_star_over_lambda_J": ratio_J,
     "note": "crossing of the extensive (M c_s^2) and non-extensive ((3/5)GM^2/R) laws: an iron sphere whose "
             "self-gravity energy equals its bulk-sound pressure energy; Jeans length times sqrt(5)/(2 pi)"}

# -- verdicts against the pre-registered criteria -----------------------------
pre = json.load(open(os.path.join(HERE, "u8_prereg.json"), encoding="utf8"))
sU = A["slope_U_engine_pe_j"]
sP = A["slope_E_P_Mcs2"]
b256, b2048 = B["N256"], B["N2048"]
orig = {
    "U_slope_1.667pm0.03": abs(sU - 5 / 3) <= 0.03,
    "EP_slope_1.00pm0.01": abs(sP - 1.0) <= 0.01,
    "gravity_-2.00pm0.05_N256_eps0": abs(b256["eps0"]["gravity_slope_2h_R"] + 2) <= 0.05,
    "gravity_-2.00pm0.05_N256_eps0.5h": abs(b256["eps0.5h"]["gravity_slope_2h_R"] + 2) <= 0.05,
    "pressure_identically_0_beyond_2h_N256_isothermal":
        b256["eps0"]["pressure_isothermal"]["identically_zero_beyond_2h"],
    "pressure_identically_0_beyond_2h_N256_tait": b256["eps0"]["pressure_tait"]["identically_zero_beyond_2h"],
}
same = abs(sU - sP) <= 0.1
orig_outcome = ("SEPARATE" if all(orig.values())
                else ("SAME" if same else "NEITHER (a SEPARATE clause fails; SAME fails)"))
corr = {
    "U_slope_seed7": abs(sU - 5 / 3) <= 0.03,
    "U_slope_10seed_mean": abs(A["multiseed"]["slope_U_mean"] - 5 / 3) <= 0.03,
    "EP_slope": abs(sP - 1.0) <= 0.01,
    "gravity_N256_eps0_2h_R": abs(b256["eps0"]["gravity_slope_2h_R"] + 2) <= 0.05,
    "gravity_N2048_eps0.5h_4h_R": abs(b2048["eps0.5h"]["gravity_slope_4h_R"] + 2) <= 0.05,
    "pressure_zero_beyond_4h_N2048_iso": (b2048["eps0"]["pressure_isothermal"]["identically_zero_beyond_4h"]
                                          and b2048["eps0"]["pressure_isothermal"]["n_beyond_4h"] > 0),
    "pressure_zero_beyond_4h_N2048_tait": (b2048["eps0"]["pressure_tait"]["identically_zero_beyond_4h"]
                                           and b2048["eps0"]["pressure_tait"]["n_beyond_4h"] > 0),
    "pressure_nonzero_in_2h_4h_N2048": b2048["eps0"]["pressure_isothermal"]["n_nonzero_beyond_2h"] > 0,
}
corr_outcome = "SEPARATE" if all(corr.values()) else ("SAME" if same else "MIXED")

out = {"id": "U8", "prereg_file": os.path.join(HERE, "u8_prereg.json"),
       "prereg_written_utc": pre["written_utc"],
       "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "choices": {"particle_mass_kg": M_P, "rho_cloud_kg_m3": RHO, "d_m": D, "h_m": H, "k": KH,
                   "eps_default_m": EPS_DEFAULT, "c_s_m_s": CS, "K_mentat_Pa": K_MENTAT,
                   "K_engine_Pa": K_ENGINE, "seed": 7, "omega": 0.0, "N_set": NS,
                   "control_groups": {"2h/R": A["two_h_over_R"], "eps/d": EPS_DEFAULT / D}},
       "mentat_sources": SRC, "partA": A, "partB": B, "partC": C,
       "verdict_original": {"clauses": orig, "outcome": orig_outcome, "same_clause_abs_dslope_le_0.1": same},
       "verdict_corrected": {"clauses": corr, "outcome": corr_outcome},
       "runtime_s": time.time() - T0}
json.dump(out, open(os.path.join(HERE, "u8_results.json"), "w", encoding="utf8"), indent=1)
summary = {k: v for k, v in A.items() if k.startswith("slope_") or k.startswith("U_engine_over")
           or k.startswith("local_slopes")}
print(json.dumps(summary, indent=1))
print(json.dumps(A["multiseed"], indent=1))
print(json.dumps(C, indent=1))
print(json.dumps(out["verdict_original"], indent=1))
print(json.dumps(out["verdict_corrected"], indent=1))
print(f"runtime {time.time()-T0:.1f}s")
