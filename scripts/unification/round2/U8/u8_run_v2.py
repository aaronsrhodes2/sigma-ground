"""U8 v2 - Range and extensivity, repaired after three skeptic reviews.

Re-runs every pre-registered measurement of u8_run.py unchanged (same seeds, same choices,
same criteria read from u8_prereg.json) and adds, clearly labelled POST HOC (skeptic-requested,
not pre-registered):
  A+  fit-SE vs seed-SD labelling; check of the short-range-control prediction as written.
  A2  extension of the engine-measured short-range energies to N = 8192 (seed 7) and 4096
      (seeds 1, 2): kernel pair energy local slopes; Tait sum m P/rho decomposed into an
      interior (bulk) part and a surface part; interior rho_SPH/rho0 vs the Mentat self-term.
  B+  exact comparison of the measured non-zero pressure-response set with the predicted
      two-hop neighbour set (engine neighbour lists), and the fraction of parcels reached.
  C   renamed: ENERGY-EQUALITY crossing |U| = M c_s^2 (not virial balance); the isothermal
      virial crossing |U| = 3 M c_s^2 added (Mentat); proxy caveat recorded.
  V   literal-design verdict under every reading (eps = 0, eps = 0.5h, E_P read as the Tait
      sum), plus the pre-registered corrected verdict.
Read-only on the repo; outputs only in this folder.  Run from the repo root.
"""
import sys, os, json, math, time, subprocess
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
REPO = r"D:/Aaron/development/sigma-ground"
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
SELF_TERM = float(M("simplify_expr", {"expression": "N(1/(pi*(6/5)**3))"}, "kernel_self_term_mW0_over_rho0"))

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


def local_slopes(Ns, arr):
    return [math.log(abs(arr[i + 1] / arr[i])) / math.log(Ns[i + 1] / Ns[i]) for i in range(len(Ns) - 1)]


# -- Part A: extensivity (pre-registered, unchanged) ----------------------------
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
          "isothermal_sum_rel_dev_from_Mcs2": [s / e - 1 for s, e in zip(sumPrho_iso, EP)],
          "tags": {"U_engine_pe_j": "MEASURED (engine pair sum)", "U_mentat": "BY CONSTRUCTION (formula)",
                   "E_P_Mcs2": "BY CONSTRUCTION (formula M c_s^2; computed in this script)",
                   "sum_mP_over_rho_isothermal": "BY CONSTRUCTION (P = rho c_s^2 makes it identical to M c_s^2)",
                   "kernel_pair_energy_sum_m_rho": "MEASURED (engine; includes the extensive self term m^2 W(0))",
                   "sum_mP_over_rho_tait": "MEASURED (engine; exploratory, no criterion)"}})
for key, arr in (("U_engine_pe_j", U_eng), ("U_engine_softened_0.5h", U_soft), ("U_mentat", U_men),
                 ("E_P_Mcs2", EP), ("sum_mP_over_rho_isothermal", sumPrho_iso),
                 ("kernel_pair_energy_sum_m_rho", kern), ("sum_mP_over_rho_tait", sumPrho_tait)):
    if all(a != 0 for a in arr) and (all(a > 0 for a in arr) or all(a < 0 for a in arr)):
        s, b, se = fit(NS, arr)
        A[f"slope_{key}"] = s
        A[f"slope_se_{key}"] = se
        A[f"local_slopes_{key}"] = local_slopes(NS, arr)
    else:
        A[f"slope_{key}"] = None
        A[f"sign_change_{key}"] = arr
A["slope_se_meaning"] = ("slope_se_* is the least-squares standard error of a 5-point fit within ONE seed. "
                         "The five N-point sets are nested rescalings of one RNG stream, so their residuals are "
                         "correlated and this SE understates the real uncertainty; use the seed-to-seed SD "
                         "(multiseed.slope_U_sd) as the realistic uncertainty of a single-seed exponent.")
A["kernel_self_term_fraction_N512"] = (NS[-1] * M_P * M_P * SELF_TERM * RHO) / kern[-1]

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
                  "slope_U_se_of_mean": float(np.std(ms_slopes, ddof=1) / math.sqrt(len(SEEDS))),
                  "slope_kernel_control": ms_kern_slopes,
                  "slope_kernel_mean": float(np.mean(ms_kern_slopes)),
                  "slope_kernel_sd": float(np.std(ms_kern_slopes, ddof=1)),
                  "slope_kernel_se_of_mean": float(np.std(ms_kern_slopes, ddof=1) / math.sqrt(len(SEEDS))),
                  "U_ratio_mean_per_N": [float(v) for v in np.mean(ms_ratio, axis=0)],
                  "U_ratio_sd_per_N": [float(v) for v in np.std(ms_ratio, axis=0, ddof=1)],
                  "U_ratio_se_per_N": [float(v) / math.sqrt(len(SEEDS)) for v in np.std(ms_ratio, axis=0, ddof=1)],
                  "pm_meaning": "slope_*_sd = SD across the 10 seeds; *_se_of_mean = SD/sqrt(10)"}

# short-range control prediction, checked as written in u8_prereg.json
lk = A["local_slopes_kernel_pair_energy_sum_m_rho"]
A["control_prediction_check"] = {
    "prediction_verbatim": "between 1.0 and 1.25, local slopes falling toward 1 (bulk N + negative surface N^(2/3) term)",
    "global_slope": A["slope_kernel_pair_energy_sum_m_rho"],
    "global_slope_in_1.0_1.25": bool(1.0 <= A["slope_kernel_pair_energy_sum_m_rho"] <= 1.25),
    "local_slopes": lk,
    "all_local_slopes_le_1.25": bool(all(v <= 1.25 for v in lk)),
    "local_slopes_monotonically_falling": bool(all(lk[i + 1] <= lk[i] for i in range(len(lk) - 1))),
    "would_pass_1.00pm0.01_as_E_P": bool(abs(A["slope_kernel_pair_energy_sum_m_rho"] - 1.0) <= 0.01),
}
cpc = A["control_prediction_check"]
cpc["reading"] = ("global slope PASS; 'local slopes falling toward 1' NOT cleanly met at N <= 512 "
                  "(first local slope above 1.25, sequence not monotonic)"
                  if cpc["global_slope_in_1.0_1.25"] and not (cpc["all_local_slopes_le_1.25"]
                                                             and cpc["local_slopes_monotonically_falling"])
                  else "as predicted" if cpc["global_slope_in_1.0_1.25"] else "global slope outside 1.0-1.25")


# -- Part A2 (POST HOC): engine short-range energies at larger N ---------------
def ext(N, seed):
    _, scene = make(N, seed)
    par = scene.continuum_parcels()
    _sph_density_sum(par, H)
    k_ = sum(p.mass * p.sph_density for p in par)
    for p in par:
        _pressure_from_eos(p)
    X = np.array([[p.position.x, p.position.y, p.position.z] for p in par])
    mass = np.array([p.mass for p in par])
    rho = np.array([p.sph_density for p in par])
    P = np.array([p.pressure for p in par])
    bc = (mass[:, None] * X).sum(0) / mass.sum()
    rc = np.linalg.norm(X - bc, axis=1)
    inner = rc < cloud_R(N) - 2 * H
    term = mass * P / rho
    tait = float(sum(p.mass * p.pressure / p.sph_density for p in par))   # same summation order as Part A
    return {"N": N, "seed": seed, "kernel_sum_m_rho": k_, "tait_sum_mP_over_rho_J": tait,
            "tait_over_Mcs2": tait / (N * M_P * CS * CS),
            "n_interior": int(inner.sum()),
            "fraction_within_2h_of_surface": float(1 - inner.mean()),
            "interior_mean_rho_over_rho0": (float(rho[inner].mean() / RHO) if inner.any() else None),
            "all_mean_rho_over_rho0": float(rho.mean() / RHO),
            "fraction_P_negative": float((P < 0).mean()),
            "mean_P_over_K": float((P / K_ENGINE).mean()),
            "tait_interior_part_J": float(term[inner].sum()),
            "tait_surface_part_J": float(term[~inner].sum())}


EXT = {"label": "POST HOC (skeptic-requested), not pre-registered", "per_seed": {},
       "interior_definition": "distance from the mass barycentre < R - 2h (R = seeded cloud radius)"}
for seed, Ns in ((7, [256, 512, 1024, 2048, 4096, 8192]), (1, [256, 512, 1024, 2048, 4096]),
                 (2, [256, 512, 1024, 2048, 4096])):
    rows = []
    for N in Ns:
        rows.append(ext(N, seed))
        r_ = rows[-1]
        print(f"A2 seed {seed} N={N}: kern={r_['kernel_sum_m_rho']:.5e} tait={r_['tait_sum_mP_over_rho_J']:.4e} "
              f"(int {r_['tait_interior_part_J']:.3e}, surf {r_['tait_surface_part_J']:.3e}) "
              f"rho_int/rho0={r_['interior_mean_rho_over_rho0']} t={time.time()-T0:.0f}s", flush=True)
    kk = [r_["kernel_sum_m_rho"] for r_ in rows]
    tt = [r_["tait_sum_mP_over_rho_J"] for r_ in rows]
    sign_change = [(Ns[i], Ns[i + 1]) for i in range(len(Ns) - 1) if tt[i] < 0 <= tt[i + 1]]
    EXT["per_seed"][str(seed)] = {"N": Ns, "rows": rows, "kernel_local_slopes": local_slopes(Ns, kk),
                                  "tait_sign_change_between": sign_change}
s7 = EXT["per_seed"]["7"]
EXT["consistency_with_partA_seed7"] = {
    "kernel_N256_N512_equal": bool(s7["rows"][0]["kernel_sum_m_rho"] == kern[3] and s7["rows"][1]["kernel_sum_m_rho"] == kern[4]),
    "tait_N256_N512_equal": bool(s7["rows"][0]["tait_sum_mP_over_rho_J"] == sumPrho_tait[3]
                                 and s7["rows"][1]["tait_sum_mP_over_rho_J"] == sumPrho_tait[4])}
EXT["kernel_last_local_slope_within_0.03_of_1"] = {s: bool(abs(v["kernel_local_slopes"][-1] - 1) <= 0.03)
                                                   for s, v in EXT["per_seed"].items()}
EXT["self_term_expected_interior_rho_over_rho0"] = 1 + SELF_TERM


# -- Part B: range (pre-registered) + two-hop set check (post hoc) --------------
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
           "n_beyond_2h": int(np.sum(r > 2 * H)), "n_beyond_4h": int(np.sum(r > 4 * H)),
           "scene_h_equals_H": bool(abs(scene.h - H) <= 1e-15 * H),
           "scene_default_softening_over_h": float(scene.gravity_softening_m / scene.h)}
    m0 = par[c].mass
    dm = 0.1 * m0
    pres_store, nz_store = {}, {}
    nbrs = None
    for tag, eps in (("eps0", None), ("eps0.5h", EPS_DEFAULT)):
        res = {}
        for state in ("base", "pert"):
            par[c].mass = m0 * (1.1 if state == "pert" else 1.0)
            g = _sph_density_sum(par, H, gravity_G=G, softening=eps)
            if state == "base" and nbrs is None:
                nbrs = [list(p.neighbors) for p in par]          # engine neighbour lists (r < 2h)
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
            nz_store[(tag, name)] = nz
        out[tag] = res_out
        pi_ = res_out["pressure_isothermal"]
        print(f"B N={N} {tag}: grav slope [2h,R]={s:.5f} binned={sb} [4h,R]={s4[0] if s4 else None} "
              f"rel_dev={rel:.2e} | iso nz>2h={pi_['n_nonzero_beyond_2h']}/{pi_['n_beyond_2h']} "
              f"nz>4h={pi_['n_nonzero_beyond_4h']}/{pi_['n_beyond_4h']} "
              f"rmax_nz={pi_['max_r_nonzero_over_h']:.3f}h  t={time.time()-T0:.1f}s", flush=True)
    out["pressure_identical_across_eps"] = {k: bool(np.array_equal(v[0], v[1])) for k, v in pres_store.items()}
    out["dm_kg"] = dm
    # POST HOC: predicted two-hop set from the engine's own neighbour lists
    near_c = set([c]) | set(nbrs[c])
    pred = np.array([(i in near_c) or any(j in near_c for j in nbrs[i]) for i in range(n)])
    th = {"label": "POST HOC exact check of the pre-registered two-hop mechanism",
          "definition": "{c} U N(c) U {i : N(i) meets {c} U N(c)}, N = engine neighbour list (r < 2h)",
          "n_predicted": int(pred.sum()), "n_total": n, "fraction_of_parcels_reached": float(pred.mean()),
          "max_r_predicted_over_h": float(r[pred].max() / H)}
    for (tag, name), nz in nz_store.items():
        th[f"{tag}_{name}"] = {"n_nonzero": int(nz.sum()), "sets_equal": bool(np.array_equal(nz, pred)),
                               "n_predicted_but_zero": int(np.sum(pred & ~nz)),
                               "n_nonzero_not_predicted": int(np.sum(nz & ~pred))}
    th["all_sets_equal"] = bool(all(v["sets_equal"] for k, v in th.items() if isinstance(v, dict)))
    out["two_hop_set_check"] = th
    print(f"B N={N} two-hop: predicted {th['n_predicted']}/{n} ({th['fraction_of_parcels_reached']:.3f}), "
          f"all sets equal {th['all_sets_equal']}", flush=True)
    return out


B = {"N256": range_run(256), "N2048": range_run(2048)}

# -- Part C: ENERGY-EQUALITY crossing (renamed) and the isothermal virial crossing ----
lnA_U = float(np.mean(np.log(np.abs(U_eng)) - (5.0 / 3.0) * np.log(NS)))   # fixed-slope prefactor
A_P = M_P * CS * CS
N_star_engine = math.exp((math.log(A_P) - lnA_U) / (2.0 / 3.0))
M("solve_equation", {"equation": "4*pi*G*rho*R^2/5 = c^2", "variable": "R"}, "Rstar_equality_symbolic")
M("solve_equation", {"equation": "4*pi*G*rho*R^2/5 = 3*c^2", "variable": "R"}, "R_virial_symbolic")
R_star = float(M("simplify_expr",
                 {"expression": f"N(sqrt(5)*{CS:.10e}/(2*sqrt(pi*{G:.10e}*{RHO})))"}, "Rstar_equality_value"))
R_vir = float(M("simplify_expr",
                {"expression": f"N(sqrt(15)*{CS:.10e}/(2*sqrt(pi*{G:.10e}*{RHO})))"}, "R_virial_value"))
LAM_J = float(M("simplify_expr", {"expression": f"N({CS:.10e}*sqrt(pi/({G:.10e}*{RHO})))"}, "lambda_J_value"))
ratio_J = M("simplify_expr", {"expression": "sqrt(5/(4*pi))/sqrt(pi)"}, "Rstar_over_lambdaJ")
ratio_Jv = M("simplify_expr", {"expression": "sqrt(15/(4*pi))/sqrt(pi)"}, "Rvirial_over_lambdaJ")
N_star_formula = (4.0 / 3.0) * math.pi * RHO * R_star ** 3 / M_P
N_vir_formula = (4.0 / 3.0) * math.pi * RHO * R_vir ** 3 / M_P
C = {"label": "[DERIVED] illustration, not a test",
     "energy_equality_crossing": {
         "definition": "|U| = (3/5) G M^2 / R equals E_P = M c_s^2 (energy equality, NOT virial balance)",
         "A_U_fixed_slope": math.exp(lnA_U), "A_P": A_P,
         "N_star_from_engine_fits": N_star_engine, "N_star_from_Mentat_formula": N_star_formula,
         "N_star_ratio_engine_over_formula": N_star_engine / N_star_formula,
         "R_star_m": R_star, "R_star_over_R_earth": R_star / R_EARTH,
         "R_star_over_lambda_J_symbolic": ratio_J, "R_star_over_lambda_J": R_star / LAM_J},
     "isothermal_virial_crossing": {
         "definition": "2K + W = 0 with 2K = 3 * integral(P dV) = 3 M c_s^2, i.e. |U| = 3 M c_s^2",
         "R_virial_m": R_vir, "R_virial_over_R_star": R_vir / R_star,
         "R_virial_over_R_earth": R_vir / R_EARTH,
         "R_virial_over_lambda_J_symbolic": ratio_Jv, "R_virial_over_lambda_J": R_vir / LAM_J,
         "N_virial_from_Mentat_formula": N_vir_formula, "N_virial_over_N_star": N_vir_formula / N_star_formula},
     "lambda_J_m": LAM_J,
     "proxy_caveat": ("E_P = M c_s^2 with iron's zero-pressure bulk sound speed is a dimensional proxy for a "
                      "pressure energy. It is not the compressional energy of a solid at its reference density, "
                      "which is about zero. The crossing marks where the two scaling laws meet, nothing more.")}

# -- verdicts against the pre-registered criteria -----------------------------
pre = json.load(open(os.path.join(HERE, "u8_prereg.json"), encoding="utf8"))
sU = A["slope_U_engine_pe_j"]
sP = A["slope_E_P_Mcs2"]
sT = A["slope_sum_mP_over_rho_tait"]
sK = A["slope_kernel_pair_energy_sum_m_rho"]
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
pres2h = orig["pressure_identically_0_beyond_2h_N256_isothermal"] and orig["pressure_identically_0_beyond_2h_N256_tait"]
tait_powerlaw = sT is not None and not any(v["tait_sign_change_between"] for v in EXT["per_seed"].values())
tait_ok = bool(sT is not None and abs(sT - 1.0) <= 0.01 and tait_powerlaw)
readings = {
    "eps0_EP_as_Mcs2": {"U": orig["U_slope_1.667pm0.03"], "E_P": orig["EP_slope_1.00pm0.01"],
                        "gravity": orig["gravity_-2.00pm0.05_N256_eps0"], "pressure_0_beyond_2h": pres2h},
    "eps0.5h_EP_as_Mcs2": {"U": orig["U_slope_1.667pm0.03"], "E_P": orig["EP_slope_1.00pm0.01"],
                           "gravity": orig["gravity_-2.00pm0.05_N256_eps0.5h"], "pressure_0_beyond_2h": pres2h},
    "eps0_EP_as_Tait_sum": {"U": orig["U_slope_1.667pm0.03"], "E_P": tait_ok,
                            "gravity": orig["gravity_-2.00pm0.05_N256_eps0"], "pressure_0_beyond_2h": pres2h},
    "eps0.5h_EP_as_Tait_sum": {"U": orig["U_slope_1.667pm0.03"], "E_P": tait_ok,
                               "gravity": orig["gravity_-2.00pm0.05_N256_eps0.5h"], "pressure_0_beyond_2h": pres2h},
}
for k, v in readings.items():
    fails = [c_ for c_, ok in v.items() if not ok]
    v["failing_clauses"] = fails
    v["outcome"] = "SEPARATE" if not fails else ("SAME" if same else "NEITHER")
literal_outcomes = sorted(set(v["outcome"] for v in readings.values()))
same_any = {"|U - E_P|": abs(sU - sP), "|U - kernel control|": abs(sU - sK),
            "|U - Tait (N<=512 fit)|": (abs(sU - sT) if sT is not None else None)}
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
informational = {
    "gravity_N2048_eps0.5h_2h_R": b2048["eps0.5h"]["gravity_slope_2h_R"],
    "gravity_N2048_eps0.5h_2h_R_inside_original_band": abs(b2048["eps0.5h"]["gravity_slope_2h_R"] + 2) <= 0.05,
    "note": ("not a pre-registered clause: with the longer N = 2048 window even the default softening passes "
             "-2.00 +/- 0.05 over [2h, R]; the N = 256 failure is a short-window effect (R/h = 3.28)")}


# -- post-hoc blocks produced by their own scripts ------------------------------
def load(name):
    p = os.path.join(HERE, name)
    return json.load(open(p, encoding="utf8")) if os.path.exists(p) else None


posthoc = {"ratio_200_seeds_u8_posthoc_ratio.json": load("u8_posthoc_ratio.json"),
           "ratio_large_samples_u8_posthoc_ratio_v2.json": load("u8_posthoc_ratio_v2.json"),
           "note": ("POST HOC, not pre-registered. In the 200-seed block 'se' is the standard error of the mean "
                    "(SD/sqrt(200)). The v2 block (4000/2000/500 fresh seeds) is the test of whether the "
                    "same-sign 200-seed z values were noise.")}

git_now = subprocess.run(["git", "-C", REPO, "status", "--short"], capture_output=True, text=True).stdout
git_before = open(os.path.join(HERE, "git_status_before_v2.txt"), encoding="utf8").read()

out = {"id": "U8", "version": "v2 (repair after skeptic review)",
       "prereg_file": os.path.join(HERE, "u8_prereg.json"),
       "prereg_written_utc": pre["written_utc"],
       "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "choices": {"particle_mass_kg": M_P, "rho_cloud_kg_m3": RHO, "d_m": D, "h_m": H, "k": KH,
                   "eps_default_m": EPS_DEFAULT, "c_s_m_s": CS, "K_mentat_Pa": K_MENTAT,
                   "K_engine_Pa": K_ENGINE, "seed": 7, "omega": 0.0, "N_set": NS,
                   "control_groups": {"2h/R": A["two_h_over_R"], "eps/d": EPS_DEFAULT / D}},
       "mentat_sources": SRC, "partA": A, "partA2_posthoc_extension": EXT, "partB": B, "partC": C,
       "verdict_original": {"clauses": orig, "outcome": orig_outcome, "same_clause_abs_dslope_le_0.1": same,
                            "readings": readings, "literal_outcomes_over_all_readings": literal_outcomes,
                            "same_clause_dslopes": same_any,
                            "tait_sum_demoted_to_exploratory_before_run": pre["predictions"]["tait_sum_mP_over_rho"]},
       "verdict_corrected": {"clauses": corr, "outcome": corr_outcome},
       "outcome_summary": (f"{corr_outcome} under the pre-registered corrected criterion; "
                           f"{' / '.join(literal_outcomes)} under the design's literal criterion (every reading)"),
       "informational": informational,
       "posthoc_not_preregistered": posthoc,
       "lit_refs_verified": [
           "Padmanabhan T. 1990, Statistical mechanics of gravitating systems, Phys. Rep. 188(5), 285-362, "
           "doi:10.1016/0370-1573(90)90051-3 (web search: INSPIRE 284425, IAS repository 25522)",
           "Campa A., Dauxois T., Ruffo S. 2009, Statistical mechanics and dynamics of solvable models with "
           "long-range interactions, Phys. Rep. 480, 57-159, doi:10.1016/j.physrep.2009.07.001, arXiv:0907.0323 "
           "(web search: arXiv, UNIFI FLORE)"],
       "files": {"prereg_script": "u8_prereg.py", "prereg_json": "u8_prereg.json",
                 "run_script_v1": "u8_run.py", "run_script": "u8_run_v2.py", "log": "u8_run_v2.log",
                 "posthoc_scripts": ["u8_posthoc_ratio.py", "u8_posthoc_ratio_v2.py"],
                 "figure_script": "u8_figure_v2.py", "figure": "u8_range_extensivity_v2.png",
                 "results_v1": "u8_results.json"},
       "repo_untouched": {"git_status_unchanged_during_v2": git_now == git_before,
                          "git_status_short": git_now.splitlines()},
       "runtime_s": time.time() - T0}
json.dump(out, open(os.path.join(HERE, "u8_results_v2.json"), "w", encoding="utf8"), indent=1)
summary = {k: v for k, v in A.items() if k.startswith("slope_") or k.startswith("U_engine_over")
           or k.startswith("local_slopes")}
print(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in A["multiseed"].items() if k not in ("slope_U", "slope_kernel_control")}, indent=1))
print(json.dumps(A["control_prediction_check"], indent=1))
print(json.dumps({s: {"kernel_local_slopes": v["kernel_local_slopes"], "tait_sign_change": v["tait_sign_change_between"],
                      "tait_over_Mcs2": [r_["tait_over_Mcs2"] for r_ in v["rows"]],
                      "rho_int": [r_["interior_mean_rho_over_rho0"] for r_ in v["rows"]]}
                  for s, v in EXT["per_seed"].items()}, indent=1))
print(json.dumps(EXT["consistency_with_partA_seed7"]))
print(json.dumps({"N256": B["N256"]["two_hop_set_check"], "N2048": B["N2048"]["two_hop_set_check"]}, indent=1))
print(json.dumps(C, indent=1))
print(json.dumps(out["verdict_original"]["readings"], indent=1))
print(json.dumps(out["verdict_corrected"], indent=1))
print(out["outcome_summary"], "| repo unchanged:", out["repo_untouched"]["git_status_unchanged_during_v2"])
print(f"runtime {time.time()-T0:.1f}s")
