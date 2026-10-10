"""U10 repair (v2) analysis. Re-scores the SAME pre-registered criteria (u10_prereg.txt, unchanged) on the SAME solver outputs
(u10_n{32,40,48}.json + u10_fields_n*.npz from u10_run.py; reproduced bitwise from fresh processes by the reproduce skeptic), and adds
what the three skeptics required:
  1. an exact DISCRETE solve (7-point operator, same Dirichlet walls, sparse CG) of every source, so each stored field splits into
     spatial (stencil) error + split-kick bias + n_tau = 8 transient;
  2. the corrected split-kick fixed point: m explicit-Euler substeps of h then a kick m*h*s ->
     T* - v_d = ((m-1)/2) h s + ((m^2-1)/12) h^2 (-L s) + ...   (Mentat series; exactly 0 for m = 1)
     (the original analysis used (m h / 2) s, which is wrong);
  3. C1/C2/Q_T/Q_N split into cell-sum quadrature / Phi_N spatial error / Phi_N split-kick parts;
  4. the power of C5's primary metric against a first-order spatial defect;
  5. the comparison with the earlier in-session P7b run (same solver, same 40^3 grid, written before the prereg);
  6. C4 with the prereg's own normalisation, max(T_an - T_inf) over non-wall cells (original used 2 a^2).
Writes u10_results_v2.json and u10_figure_v2.png. Run from the repo root:  python -X utf8 <this file>"""
import json, math, os, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import quad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Patch

REPO = r"D:/Aaron/development/sigma-ground"; sys.path.insert(0, REPO)
from sigma_ground.dynamics.fields.heat import stable_dt, diffuse_fvm
OUT = os.path.dirname(os.path.abspath(__file__))
P7B = os.path.join(os.path.dirname(os.path.dirname(OUT)), "sigma-bridge-1pn", "p7_restricted_results.json")
LV = json.load(open(os.path.join(OUT, "u10_leaves.json")))
LV2 = json.load(open(os.path.join(OUT, "u10_leaves_v2.json")))
G, C = LV["G"]["value"], LV["c"]["value"]
XI = LV["XI"]["value"]
PRED = {f"{p['w']:g}": p for p in LV["predictions"]}
NS = [32, 40, 48]; WS = ["0.001", "0.01"]; PI = math.pi
RUN = {n: json.load(open(os.path.join(OUT, f"u10_n{n}.json"))) for n in NS}
OLD = json.load(open(os.path.join(OUT, "u10_results.json"), encoding="utf-8"))


# ---------- analytic potentials of the dimensionless sources (a = 1, rho_c = 1), as in u10_analyze.py ----------
def th(x):
    return math.sin(x) / x if x > 1e-12 else 1.0


def src(kind, w):
    mf = (lambda x: 1 + (4 * w / 3) * (1 + th(x))) if kind.endswith("PN") else (lambda x: 1.0)
    sgn = +1 if kind.startswith("T") else -1
    return lambda x: mf(x) * (th(x) + sgn * w * th(x) ** 2)


def pot(s, x):
    inner = quad(lambda y: s(y) * y * y, 0, x, epsabs=0, epsrel=1e-13, limit=200)[0] / x if x > 0 else 0.0
    outer = quad(lambda y: s(y) * y, x, PI, epsabs=0, epsrel=1e-13, limit=200)[0]
    return inner + outer


def deficit_an(w, x, pn=True):
    k = "PN" if pn else ""
    return 1 - pot(src("S" + k, w), x) / pot(src("T" + k, w), x)


def fit_order(dxs, errs):
    dxs, errs = np.asarray(dxs, float), np.abs(np.asarray(errs, float))
    p, lnC = np.polyfit(np.log(dxs), np.log(errs), 1)
    pair = [float(np.log(errs[i] / errs[i + 1]) / np.log(dxs[i] / dxs[i + 1])) for i in range(len(dxs) - 1)]
    return dict(order_lsq=float(p), pairwise=pair)


# ---------- exact discrete Dirichlet Poisson solve: (6 v_i - sum v_nb)/dx^2 = s_i on non-wall cells ----------
class Discrete:
    def __init__(self, wall, r, dx):
        n = wall.shape[0]; self.wall, self.r, self.dx = wall, r, dx
        idx = -np.ones((n, n, n), int); idx[~wall] = np.arange(int((~wall).sum()))
        I = idx[~wall]; rows, cols, vals = [I], [I], [np.full(I.size, 6.0)]; self.shifts = []
        for ax in range(3):
            for sh in (+1, -1):
                nb = np.roll(idx, -sh, axis=ax); nbw = np.roll(wall, -sh, axis=ax); msk = (~wall) & (~nbw)
                rows.append(idx[msk]); cols.append(nb[msk]); vals.append(-np.ones(int(msk.sum()))); self.shifts.append((ax, sh))
        N = I.size
        self.A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)) / dx**2

    def solve(self, s):
        Q = float(s.sum() * self.dx**3); vwall = Q / (4 * np.pi * np.maximum(self.r, self.dx / 2))   # poisson_fvm's own wall rule
        b = np.zeros_like(s)
        for ax, sh in self.shifts:
            b += np.where(np.roll(self.wall, -sh, axis=ax), np.roll(vwall, -sh, axis=ax), 0.0)
        rhs = s[~self.wall] + b[~self.wall] / self.dx**2
        x, info = spla.cg(self.A, rhs, rtol=1e-13, atol=0.0, maxiter=50000)
        assert info == 0, info
        v = vwall.copy(); v[~self.wall] = x
        return v, float(np.abs(self.A @ x - rhs).max() / np.abs(rhs).max())


def neg_lap(f, dx):
    """-(discrete 7-point Laplacian) of f on interior cells (walls left 0)."""
    out = np.zeros_like(f); c = f[1:-1, 1:-1, 1:-1]
    out[1:-1, 1:-1, 1:-1] = (6 * c - f[2:, 1:-1, 1:-1] - f[:-2, 1:-1, 1:-1] - f[1:-1, 2:, 1:-1] - f[1:-1, :-2, 1:-1]
                             - f[1:-1, 1:-1, 2:] - f[1:-1, 1:-1, :-2]) / dx**2
    return out


# ---------- per grid ----------
grid = {}; KEEP = {}
for n in NS:
    t0 = time.time()
    d = RUN[n]; F = np.load(os.path.join(OUT, f"u10_fields_n{n}.npz"))
    r, wall, theta = F["r"], F["wall"], F["theta"]; dx, a, R = d["dx_m"], d["a_m"], d["R_m"]
    m_ = ~wall; inner = r < 0.9 * dx
    v_an = np.where(r < R, a * a * (1 + theta), PI * a**3 / np.maximum(r, 1e-12)); v_c = float(v_an[inner].mean())
    peak_prereg = float(v_an[m_].max()); peak_2a2 = 2 * a * a
    D = Discrete(wall, r, dx)
    vd_N, resN = D.solve(theta)
    vN, vH = F["v_N"], F["v_heat"]
    assert np.allclose(vN[wall], vd_N[wall], rtol=1e-13, atol=0)          # same wall rule as poisson_fvm
    # substeps per diffuse_fvm call, read from the engine itself
    h = stable_dt(1.0, dx); probe = np.zeros((3, 3, 3)); one = np.ones((3, 3, 3))
    m_sub = diffuse_fvm(probe, one, one, dx, total_time=h * 20, max_substeps=10**6)[1]["substeps"]
    m_sub1 = diffuse_fvm(probe, one, one, dx, total_time=h, max_substeps=10**6)[1]["substeps"]
    mu_nom = h / a**2
    nl = neg_lap(theta, dx)
    bias_claimed = (m_sub * h / 2) * theta                                      # original (wrong) model, dt_split/2 * s
    bias_ee1 = ((m_sub - 1) / 2) * h * theta                                    # corrected, first order
    bias_ee2 = bias_ee1 + ((m_sub**2 - 1) / 12) * h * h * nl                    # corrected incl. O(mu)
    # transient of the n_tau = 8 design solve, measured on the heat-law run's own centre history (same operator, same start)
    H = np.array(d["T_leg_heat"]["centre_history_t_over_tau"]); Tfin = d["T_leg_heat"]["centre_T_K"]; Tinf = d["T_leg_heat"]["T_inf_K"]
    eps8 = float((Tfin - np.interp(8.0, H[:, 0], H[:, 1])) / (Tfin - Tinf))          # original estimate (linear interp at 8 tau)
    # refined: the design run's true end time (poisson_fvm loop replayed) and log-linear interpolation of the decaying residual
    tau_d = (n * dx)**2 / (3 * np.pi**2); dts = h * 20; t_end = 0.0
    while t_end < 8 * tau_d:
        t_end += dts
    keep = H[:, 1] < Tfin
    eps_end = float(np.exp(np.interp(t_end / tau_d, H[keep, 0], np.log((Tfin - H[keep, 1]) / (Tfin - Tinf)))))
    cen = lambda f: float(f[inner].mean())
    tr_abs = eps_end * cen(vH)
    # transient field = the exact discrete fundamental Dirichlet mode of the box (walls at cells 0 and n-1), unit mean on the inner cells
    s1 = np.sin(np.pi * np.arange(n) / (n - 1)); phi = s1[:, None, None] * s1[None, :, None] * s1[None, None, :]
    phi = phi / float(phi[inner].mean()); tr_field = -tr_abs * phi
    d_design = cen(vN - vd_N); d_heat = cen(vH - vd_N)
    g = dict(n=n, dx_km=dx / 1e3, h_s=h, substeps_per_20x_kick=int(m_sub), substeps_per_1x_kick=int(m_sub1),
             mu_nominal_h_over_a2=mu_nom, mu_effective_centre=float(h * cen(nl) / cen(theta)), cg_rel_residual_N=resN,
             v_centre_analytic=v_c, peak_prereg_max_Tan=peak_prereg, peak_2a2=peak_2a2)
    # --- spatial / split / transient decomposition (units of v_c at the centre) ---
    g["heat_leg_minus_discrete_centre_over_vc"] = d_heat / v_c
    g["heat_leg_minus_discrete_maxabs_over_vc"] = float(np.abs(vH - vd_N)[m_].max() / v_c)
    g["design_leg_minus_discrete_centre_over_vc"] = d_design / v_c
    g["bias_pred_claimed_dt_split_over_2_centre_over_vc"] = cen(bias_claimed) / v_c
    g["bias_pred_corrected_first_order_centre_over_vc"] = cen(bias_ee1) / v_c
    g["bias_pred_corrected_with_O_mu_centre_over_vc"] = cen(bias_ee2) / v_c
    g["transient_eps8_from_heat_history"] = eps8
    g["design_run_end_time_over_tau"] = t_end / tau_d
    g["transient_at_design_end_loginterp"] = eps_end
    g["transient_expected_exp_minus_8_n_over_n_minus_1_sq"] = math.exp(-8 * (n / (n - 1))**2)
    for k, b in (("claimed", bias_claimed), ("corrected_first_order", bias_ee1), ("corrected_with_O_mu", bias_ee2)):
        g[f"design_minus_discrete_minus_bias_{k}_centre_over_vc"] = (d_design - cen(b)) / v_c                      # should = -transient
        g[f"closure_after_bias_{k}_and_transient_centre_over_vc"] = (d_design - cen(b) + tr_abs) / v_c          # should = 0
        g[f"design_minus_discrete_minus_bias_{k}_maxabs_over_vc"] = float(np.abs(vN - vd_N - b)[m_].max() / v_c)
        g[f"fullfield_closure_after_bias_{k}_and_transient_mode_maxabs_over_vc"] = float(np.abs(vN - vd_N - b - tr_field)[m_].max() / v_c)
    g["spatial_error_discrete_minus_analytic_centre_over_vc"] = cen(vd_N - v_an) / v_c
    g["spatial_error_discrete_minus_analytic_max_over_peak_prereg"] = float(np.abs(vd_N - v_an)[m_].max() / peak_prereg)
    # split ratios: raw (as stored) / post-hoc transient-corrected / corrected-formula prediction (1 + (m+1) mu/6)
    sc = d["split_check"]
    g["split_ratio_raw"] = sc["centre_diff_measured"] / sc["centre_diff_predicted"]
    g["split_ratio_transient_corrected_POST_HOC"] = sc["centre_diff_measured"] / (sc["centre_diff_predicted"] - eps8 * cen(vH))
    g["split_ratio_predicted_by_O_mu_term"] = cen(bias_ee2) / cen(bias_ee1)
    # --- C4 per prereg normalisation (max over non-wall cells of T_an - T_inf) and the original 2 a^2 ---
    g["C4_design_max_prereg_norm"] = float(np.abs(vN - v_an)[m_].max() / peak_prereg)
    g["C4_heat_max_prereg_norm"] = float(np.abs(vH - v_an)[m_].max() / peak_prereg)
    g["C4_design_max_2a2_norm_original"] = d["T_leg_design_err"]["max_abs_rel_to_peak"]
    g["C4_heat_max_2a2_norm_original"] = d["T_leg_heat_err"]["max_abs_rel_to_peak"]
    g["C4_design_split_part_max"] = g["C4_design_max_prereg_norm"] - g["C4_heat_max_prereg_norm"]
    # --- 1PN pair: quadrature decomposition + discrete solves of the very same sources ---
    QN = float(theta.sum()); g["w"] = {}
    for wk in WS:
        w = float(wk); p = PRED[wk]; row = d["w"][wk]
        rho_c = w * C**2 / (6 * math.pi * G * a * a); K = 2 * math.pi * G * a * a
        rho = rho_c * theta; P = K * rho**2
        mf_eng = 1 + 2 * 4 * math.pi * G * rho_c * vN / C**2       # exactly the run's arithmetic
        mf_dis = 1 + 2 * 4 * math.pi * G * rho_c * vd_N / C**2
        mf_an = 1 + (4 * w / 3) * v_an / (a * a)                   # analytic 1 - 2 Phi_N/c^2
        sTe, sSe = mf_eng * (rho + 3 * P / C**2) / rho_c, mf_eng * (rho - 3 * P / C**2) / rho_c
        assert abs(sTe.sum() * dx**3 - row["T"]["Q_quad"]) < 1e-12 * abs(row["T"]["Q_quad"])
        q = {}
        for lab, mf in (("analytic_mf", mf_an), ("discrete_mf", mf_dis), ("engine", mf_eng)):
            QT = float((mf * (theta + w * theta**2)).sum()); QS = float((mf * (theta - w * theta**2)).sum())
            q[lab] = dict(C1_dev=QS / QT - p["ratio_pressure_only"], C2_literal_rel=((QT - QS) / QN - w) / w, QT_over_QN=QT / QN)
        q["engine_stored"] = dict(C1_dev=row["Q_S_over_Q_T"] - p["ratio_pressure_only"], C2_literal_rel=(row["dQ_over_QN"] - w) / w,
                                  QT_over_QN=row["QT_over_QN"])
        ex = dict(C1_dev=p["ratio_weighted_minus_pressure_only"], C2_literal_rel=p["dQ_over_QN_weighted_rel_offset_from_w"],
                  QT_over_QN=p["QT_over_QN_weighted_exact"])
        decomp = {}
        for key in ("C1_dev", "C2_literal_rel", "QT_over_QN"):
            tot = q["engine_stored"][key] - ex[key]
            decomp[key] = dict(engine_minus_exact=tot, cell_quadrature=q["analytic_mf"][key] - ex[key],
                               PhiN_spatial_via_mf=q["discrete_mf"][key] - q["analytic_mf"][key],
                               PhiN_split_and_transient_via_mf=q["engine_stored"][key] - q["discrete_mf"][key],
                               fraction_from_cell_quadrature=(q["analytic_mf"][key] - ex[key]) / tot if tot else None)
        vdT, _ = D.solve(sTe); vdS, _ = D.solve(sSe)
        vdT2, _ = D.solve(mf_dis * (rho + 3 * P / C**2) / rho_c); vdS2, _ = D.solve(mf_dis * (rho - 3 * P / C**2) / rho_c)
        x_in = d["r_inner_m"] / a
        df_an = deficit_an(w, x_in) / w
        df_eng = row["deficit_over_w_weighted"]
        df_dis = float(np.mean(1 - vdS[inner] / vdT[inner]) / w)
        df_dis2 = float(np.mean(1 - vdS2[inner] / vdT2[inner]) / w)
        sig = lambda v: float(np.mean((2 * XI / C**2) * 4 * math.pi * G * rho_c * v[inner]))
        sig_model = 2 * XI * (2 * w / 3) * pot(src("SPN", w), x_in)
        g["w"][wk] = dict(quadrature=q, exact_derived=ex, decomposition=decomp,
                          C3_engine=df_eng, C3_discrete_same_sources=df_dis, C3_discrete_discrete_mf=df_dis2,
                          C3_analytic_at_inner_cells=df_an, C3_analytic_r0=p["deficit_over_w_weighted_centre_exact"],
                          C3_engine_minus_analytic=df_eng - df_an, C3_split_part=df_eng - df_dis, C3_spatial_part=df_dis2 - df_an,
                          C3_split_fraction_of_excess=(df_eng - df_dis) / (df_eng - df_an),
                          sigma_centre_engine=row["sigma_centre"], sigma_centre_discrete=sig(vdS), sigma_centre_model_inner_cells=sig_model,
                          sigma_engine_over_model_minus_1=row["sigma_centre"] / sig_model - 1,
                          sigma_discrete_over_model_minus_1=sig(vdS) / sig_model - 1,
                          C2c_weighted_rel=OLD["grids"][str(n)]["w"][wk]["C2c_weighted_rel"],
                          C2c_unweighted_rel=OLD["grids"][str(n)]["w"][wk]["C2c_unweighted_rel"])
    if n == 40:
        KEEP = dict(vd_N=vd_N, bias_ee2=bias_ee2, bias_claimed=bias_claimed, F=F, v_an=v_an, tr_abs=tr_abs, tr_field=tr_field)
    g["seconds"] = round(time.time() - t0, 1)
    grid[n] = g
    print(n, "done", g["seconds"], "s", flush=True)

# ---------- orders (pre-registered grids) ----------
dxs = [grid[n]["dx_km"] for n in NS]
orders = dict(
    primary_design_literal_T_leg_max=fit_order(dxs, [grid[n]["C4_design_max_prereg_norm"] for n in NS]),
    heat_law_T_leg_max_SPATIAL_SENTINEL=fit_order(dxs, [grid[n]["C4_heat_max_prereg_norm"] for n in NS]),
    discrete_exact_spatial_error_max=fit_order(dxs, [grid[n]["spatial_error_discrete_minus_analytic_max_over_peak_prereg"] for n in NS]),
    split_part_design_minus_heat_max=fit_order(dxs, [grid[n]["C4_design_split_part_max"] for n in NS]),
    deficit_err_engine_w1e_2=fit_order(dxs, [grid[n]["w"]["0.01"]["C3_engine_minus_analytic"] for n in NS]),
    deficit_err_discrete_exact_w1e_2_SPATIAL=fit_order(dxs, [grid[n]["w"]["0.01"]["C3_spatial_part"] for n in NS]),
    deficit_err_discrete_exact_w1e_3_SPATIAL=fit_order(dxs, [grid[n]["w"]["0.001"]["C3_spatial_part"] for n in NS]),
    QT_over_QN_engine_err_w1e_2=fit_order(dxs, [grid[n]["w"]["0.01"]["decomposition"]["QT_over_QN"]["engine_minus_exact"] for n in NS]))

# ---------- power of C5's primary metric against a first-order spatial defect ----------
dxa = np.array(dxs); des = np.array([grid[n]["C4_design_max_prereg_norm"] for n in NS]); heat = np.array([grid[n]["C4_heat_max_prereg_norm"] for n in NS])
split = des - heat
ordr = lambda e: float(np.polyfit(np.log(dxa), np.log(e), 1)[0])
amps = np.geomspace(0.01, 1e4, 4000) * heat[1]
ords = np.array([ordr(split + A_ * dxa / dxa[1]) for A_ in amps])
fail_amp = float(amps[np.argmax(ords < 1.2)]) if np.any(ords < 1.2) else None
A_c4 = 0.01 - split[1]
power = dict(order_with_first_order_spatial_error_of_measured_size=ordr(split + heat[1] * dxa / dxa[1]),
             largest_first_order_amplitude_passing_C4_at_40=float(A_c4), order_at_that_amplitude=ordr(split + A_c4 * dxa / dxa[1]),
             amplitude_needed_for_primary_order_below_1p2_over_measured_spatial=(fail_amp / heat[1]) if fail_amp else None,
             C4_total_at_40_at_that_amplitude=(float(split[1] + fail_amp) if fail_amp else None),
             heat_law_sentinel_order_if_spatial_were_first_order=1.0,
             note="first-order spatial error A*(dx/dx_40) added to the measured split part (design - heat); judged as in C5 (lsq over 32/40/48)")

# ---------- earlier in-session P7b run (same poisson_fvm, same 40^3 grid) ----------
p7 = json.load(open(P7B)); p7cmp = {}
for prow, wk in zip(p7, WS):
    e = RUN[40]["w"][wk]; w = float(wk)
    p7_def = (1 - prow["sigma_centre_PN"] / prow["sigma_centre_from_Phi_T"]) / w
    p7cmp[wk] = dict(Q_S_over_Q_T=[e["Q_S_over_Q_T"], prow["Q_sigPN_over_Q_TPN"]], QT_over_QN=[e["QT_over_QN"], prow["Q_TPN_over_Q_N"]],
                     deficit_over_w=[e["deficit_over_w_weighted"], p7_def], sigma_centre=[e["sigma_centre"], prow["sigma_centre_PN"]],
                     C2_literal_rel_derivable_from_p7=(prow["Q_TPN_over_Q_N"] * (1 - prow["Q_sigPN_over_Q_TPN"]) - w) / w,
                     max_abs_rel_diff=max(abs(x / y - 1) for x, y in ([e["Q_S_over_Q_T"], prow["Q_sigPN_over_Q_TPN"]],
                                                                       [e["QT_over_QN"], prow["Q_TPN_over_Q_N"]],
                                                                       [e["deficit_over_w_weighted"], p7_def],
                                                                       [e["sigma_centre"], prow["sigma_centre_PN"]])))
mt = lambda p: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))
p7info = dict(file=P7B, written=mt(P7B), prereg_written=mt(os.path.join(OUT, "u10_prereg.txt")),
              first_u10_solver_output=mt(os.path.join(OUT, "u10_n24.json")), rows=p7cmp,
              reading="the 40^3 C1, C2, C3, Q_T/Q_N and sigma_centre values existed before the prereg; at 40^3 these are post-dictions")

# ---------- criteria at 40^3 (pre-registered, unchanged; C4 with the prereg's own normalisation) ----------
g40 = grid[40]; o40 = OLD["grids"]["40"]["w"]
crit = {
    "C1 |Q_s/Q_T - (1-w/2)/(1+w/2)| < 1e-6 (w=1e-3)": dict(value=abs(o40["0.001"]["C1_dev"]), threshold=1e-6, tag="[BY CONSTRUCTION] cell-sum quadrature (99 %) + ~1 % solver term; post-diction at 40^3 (P7b)"),
    "C1 |Q_s/Q_T - (1-w/2)/(1+w/2)| < 1e-4 (w=1e-2)": dict(value=abs(o40["0.01"]["C1_dev"]), threshold=1e-4, tag="[BY CONSTRUCTION] quadrature (99 %) + ~1 % solver; post-diction"),
    "C2 literal |(Q_T-Q_s)/Q_N - w|/w < 1e-3 (w=1e-3)": dict(value=abs(o40["0.001"]["C2_literal_rel"]), threshold=1e-3, tag="FAIL as predicted; design error (target w_c omits the metric weight on the pressure term)"),
    "C2 literal |(Q_T-Q_s)/Q_N - w|/w < 1e-3 (w=1e-2)": dict(value=abs(o40["0.01"]["C2_literal_rel"]), threshold=1e-3, tag="FAIL as predicted; design error"),
    "C2' weighted vs 2B/I1, rel < 1e-3 (w=1e-3)": dict(value=abs(o40["0.001"]["C2c_weighted_rel"]), threshold=1e-3, tag="[BY CONSTRUCTION] quadrature; corrected criterion"),
    "C2' weighted vs 2B/I1, rel < 1e-3 (w=1e-2)": dict(value=abs(o40["0.01"]["C2c_weighted_rel"]), threshold=1e-3, tag="[BY CONSTRUCTION] quadrature; corrected criterion"),
    "C2' unweighted vs w, rel < 1e-3 (w=1e-3)": dict(value=abs(o40["0.001"]["C2c_unweighted_rel"]), threshold=1e-3, tag="[BY CONSTRUCTION] pure quadrature, no solver"),
    "C2' unweighted vs w, rel < 1e-3 (w=1e-2)": dict(value=abs(o40["0.01"]["C2c_unweighted_rel"]), threshold=1e-3, tag="[BY CONSTRUCTION] pure quadrature, no solver"),
    "C3 |deficit/w - 1.219| < 0.02 (w=1e-3)": dict(value=abs(g40["w"]["0.001"]["C3_engine"] - 1.219), threshold=0.02, tag="[VERIFIED numerics] solver; post-diction at 40^3 (P7b)"),
    "C3 |deficit/w - 1.219| < 0.02 (w=1e-2)": dict(value=abs(g40["w"]["0.01"]["C3_engine"] - 1.219), threshold=0.02, tag="[VERIFIED numerics] solver; post-diction at 40^3 (P7b)"),
    "C4 T leg max <= 1 % (design-literal, 40^3)": dict(value=g40["C4_design_max_prereg_norm"], threshold=0.01, tag="[VERIFIED numerics] 0.63 % split-kick + 0.09 % stencil - 0.02 % transient"),
    "C4 T leg max <= 1 % (heat law 1x kick, 40^3)": dict(value=g40["C4_heat_max_prereg_norm"], threshold=0.01, tag="[VERIFIED numerics] = exact discrete solution; pure stencil error"),
    "C5 order >= 1.2 (primary: design-literal T leg)": dict(value=orders["primary_design_literal_T_leg_max"]["order_lsq"], threshold=1.2, kind="min", tag="[BY CONSTRUCTION] split bias ~ dt_split ~ dx^2 (CFL); no power vs a first-order spatial defect that passes C4"),
    "C5 order >= 1.2 (secondary: heat-law T leg = spatial sentinel)": dict(value=orders["heat_law_T_leg_max_SPATIAL_SENTINEL"]["order_lsq"], threshold=1.2, kind="min", tag="[VERIFIED numerics] the real spatial-order test"),
}
for k, c_ in crit.items():
    c_["pass"] = bool(c_["value"] >= c_["threshold"]) if c_.get("kind") == "min" else bool(c_["value"] <= c_["threshold"] if "<=" in k else c_["value"] < c_["threshold"])
lit = [k for k in crit if not k.startswith("C2'")]; cor = [k for k in crit if not k.startswith("C2 literal")]
summary = dict(all_literal_pass=all(crit[k]["pass"] for k in lit), all_corrected_pass=all(crit[k]["pass"] for k in cor),
               literal_failures=[k for k in lit if not crit[k]["pass"]], corrected_failures=[k for k in cor if not crit[k]["pass"]])

prereg_vs_measured = {
    "C1 (40^3)": dict(prereg="~2.4e-7 / ~1.6e-5", measured=[o40["0.001"]["C1_dev"], o40["0.01"]["C1_dev"]],
                      note="matches the P7b 40^3 measurement (00:51), not the [DERIVED] metric offset (-1.56e-7 / -1.52e-5): post-diction"),
    "C2 literal (40^3)": dict(prereg="~2.2e-3 / ~2.2e-2 FAIL", measured=[o40["0.001"]["C2_literal_rel"], o40["0.01"]["C2_literal_rel"]],
                              derived=[PRED["0.001"]["dQ_over_QN_weighted_rel_offset_from_w"], PRED["0.01"]["dQ_over_QN_weighted_rel_offset_from_w"]],
                              note="fails as predicted; measured 4 % / 1 % above the derived 2.157 w (excess = cell quadrature)"),
    "C3 (40^3)": dict(prereg="~1.22-1.23 / ~1.21-1.225", measured=[g40["w"]["0.001"]["C3_engine"], g40["w"]["0.01"]["C3_engine"]],
                      note="derivable from the P7b sigma_centre ratio: post-diction at 40^3"),
    "C4 design-literal": dict(prereg="~0.7 % at 40^3 (PASS), ~1.05 % at 32^3", measured=[grid[40]["C4_design_max_prereg_norm"], grid[32]["C4_design_max_prereg_norm"]],
                              note="close, but built on the wrong bias formula (0.665 % claimed vs 0.635 % true) that happened to offset the omitted 0.094 % stencil error"),
    "C4 heat-law": dict(prereg="~0.05 % (bias 1/20 of the 20x bias)", measured=grid[40]["C4_heat_max_prereg_norm"],
                        note="MISSED by ~1.9x: the 1x leg has ZERO split bias; its 0.094 % is pure 7-point stencil error (equals the discrete solve)"),
    "C5": dict(prereg="order ~2", measured=orders["primary_design_literal_T_leg_max"]["order_lsq"], note="true but [BY CONSTRUCTION] for the primary metric"),
}

results = dict(experiment="U10", version="v2 (repair)", prereg_file="u10_prereg.txt (unchanged)",
               inputs="u10_n{32,40,48}.json + u10_fields_n*.npz from u10_run.py (unchanged; bitwise-reproduced by verify_reproduce/)",
               grids=grid, orders=orders, C5_power=power, criteria_at_40=crit, summary=summary, prereg_vs_measured=prereg_vs_measured,
               earlier_P7b_run=p7info,
               mentat_sources=dict(original={k: v for k, v in LV.items() if isinstance(v, dict) and "source" in v},
                                   repair_v2=LV2),
               engine_leaves=dict(XI=LV["XI"], solver_under_test="sigma_ground.dynamics.fields.heat.diffuse_fvm / stable_dt",
                                  pilot_wrapper="scripts/unification/pilot_p7_sigma_three_fields.py poisson_fvm (lines 20-31)"),
               predictions_original=LV["predictions"])
json.dump(results, open(os.path.join(OUT, "u10_results_v2.json"), "w"), indent=1)

# ======================================= figure =======================================
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"]
plt.rcParams.update({"axes.unicode_minus": True, "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK})
F = KEEP["F"]; r, wall, theta = F["r"], F["wall"], F["theta"]; d40 = RUN[40]; a, R = d40["a_m"], d40["R_m"]
v_an = KEEP["v_an"]; peak = grid[40]["peak_prereg_max_Tan"]; mm = ~wall
bins = np.linspace(0, r[mm].max(), 34); idx = np.digitize(r[mm], bins); rcen = 0.5 * (bins[1:] + bins[:-1])
binned = lambda f: np.array([np.mean(f[idx == i]) if np.any(idx == i) else np.nan for i in range(1, len(bins))])
fig, axs = plt.subplots(2, 3, figsize=(19.5, 10.6), dpi=150); fig.patch.set_facecolor(SURF)
for ax in axs.flat:
    ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)

# (a) radial T-leg residuals at 40^3
ax = axs[0, 0]
ax.plot(rcen / R, 100 * binned(((F["v_N"] - v_an) / peak)[mm]), color=S[0], lw=2.2, label="design-literal T leg (poisson_fvm, kick every 20 h)")
ax.plot(rcen / R, 100 * binned(((F["v_heat"] - v_an) / peak)[mm]), color=S[1], lw=2.2, label="heat-law T leg (kick every h, 16 τ)")
ax.plot(rcen / R, 100 * binned(((KEEP["vd_N"] - v_an) / peak)[mm]), color=INK, lw=1.0, ls=(0, (2, 2)), label="exact discrete Poisson solve (stencil error only)")
ax.plot(rcen / R, 100 * binned(((KEEP["vd_N"] + KEEP["bias_ee2"] + KEEP["tr_field"] - v_an) / peak)[mm]), color=S[2], lw=1.4, ls="--",
        label="discrete + ((m−1)/2)·h·s·[1+(m+1)μ/6] − n_τ = 8 transient  (m = 20)")
ax.plot(rcen / R, 100 * binned(((KEEP["vd_N"] + KEEP["bias_claimed"] + KEEP["tr_field"] - v_an) / peak)[mm]), color=S[3], lw=1.2, ls=":",
        label="discrete + (dt_split/2)·s − transient  (original formula, wrong)")
ax.axhline(1.0, color=INK2, lw=1, ls=":"); ax.text(1.5, 1.02, "1 % criterion (C4)", color=INK2, fontsize=8, va="bottom")
ax.set_ylim(-0.08, 1.2); ax.axvline(1.0, color=GRID, lw=1.2)
ax.set_xlabel("r / R"); ax.set_ylabel("(T − T_analytic) / peak rise  (%)")
ax.set_title("(a) 40³: the 1× heat-law leg IS the discrete solution (0.094 % stencil error);\nthe design leg adds a 0.63 % split-kick bias ((m−1)/2)·h·s, not (dt_split/2)·s", fontsize=9.5)
ax.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.76))

# (b) split-kick discrimination at the centre
ax = axs[0, 1]
for key, col, lab in (("claimed", S[3], "after (dt_split/2)·s  (original)"), ("corrected_first_order", S[2], "after ((m−1)/2)·h·s"),
                      ("corrected_with_O_mu", S[0], "after ((m−1)/2)·h·s·[1+(m+1)μ/6]")):
    ax.plot(dxs, [100 * grid[n][f"design_minus_discrete_minus_bias_{key}_centre_over_vc"] for n in NS], color=col, marker="o", lw=1.8,
            label=f"design − discrete − bias {lab}")
ax.plot(dxs, [-100 * grid[n]["transient_at_design_end_loginterp"] for n in NS], color=INK2, marker="s", lw=1.2, ls="--",
        label="− transient at the design run's end, read off the heat-law run's history")
ax.plot(dxs, [100 * grid[n]["heat_leg_minus_discrete_centre_over_vc"] for n in NS], color=S[1], marker="D", lw=1.4,
        label="heat-law leg − discrete (no split bias)")
ax.axhline(0, color=INK, lw=0.8)
ax.set_xlabel("dx (km)   [n = 48, 40, 32 in a 40 km box]"); ax.set_ylabel("centre residual (% of v_centre)")
ax.set_title("(b) only the explicit-Euler fixed point leaves just the n_τ = 8 transient;\nthe original (dt_split/2)·s over-subtracts by 0.02–0.05 % of v_centre", fontsize=9.5)
ax.set_ylim(-0.115, 0.008); ax.legend(frameon=False, fontsize=7, loc="lower center")

# (c) convergence
ax = axs[0, 2]; dxa = np.array(dxs)
series = [("heat-law T leg (spatial sentinel)", [grid[n]["C4_heat_max_prereg_norm"] for n in NS], orders["heat_law_T_leg_max_SPATIAL_SENTINEL"], S[1], "-"),
          ("|deficit/w − exact|, discrete solve, w_c = 0.01", [abs(grid[n]["w"]["0.01"]["C3_spatial_part"]) for n in NS], orders["deficit_err_discrete_exact_w1e_2_SPATIAL"], S[2], "-"),
          ("design-literal T leg (C5 primary; split-dominated)", [grid[n]["C4_design_max_prereg_norm"] for n in NS], orders["primary_design_literal_T_leg_max"], S[0], "--"),
          ("|deficit/w − exact|, engine, w_c = 0.01 (split-dominated)", [abs(grid[n]["w"]["0.01"]["C3_engine_minus_analytic"]) for n in NS], orders["deficit_err_engine_w1e_2"], S[3], "--")]
for lab, e, o, col, ls in series:
    ax.loglog(dxa, e, color=col, marker="o", lw=1.6, ls=ls, label=f"{lab}: order {o['order_lsq']:.2f}")
ref = np.array([dxa.min(), dxa.max()])
ax.loglog(ref, series[0][1][1] * ref ** 2, color=INK2, lw=1, ls=":", label="slope 2 reference")
ax.loglog(ref, series[0][1][1] * ref ** 1, color=INK2, lw=1, ls="-.", label="slope 1 reference")
ax.set_xticks(dxs); ax.set_xticklabels([f"{x:.3g}" for x in dxs]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("dx (km)"); ax.set_ylabel("error (fraction)")
ax.set_title("(c) spatial order = 2.06 (heat-law leg, = discrete solve); the primary\nmetric's 2.10 is the CFL time step (dt_split ∝ dx²), not the operator", fontsize=9.5)
ax.set_ylim(1.2e-4, 2.5e-2); ax.legend(frameon=False, fontsize=6.8, loc="lower right")

# (d) deficit profile
ax = axs[1, 0]; xs_an = np.linspace(0.02, 1.25, 50)
for j, wk in enumerate(WS):
    w = float(wk); num = (1 - F[f"v_S_{wk}"] / F[f"v_T_{wk}"]) / w
    ax.plot(rcen / R, binned(num[mm]), color=S[j], lw=0, marker="o", ms=3.3, label=f"engine, w_c = {w:g}")
    yy = [deficit_an(w, x * PI) / w if x < 1 else (1 - pot(src('SPN', w), PI) / pot(src('TPN', w), PI)) / w for x in xs_an]
    ax.plot(xs_an, yy, color=S[j], lw=1.3, label=f"design recipe, analytic, w_c = {w:g}")
ax.axhspan(1.219 - 0.02, 1.219 + 0.02, xmin=0, xmax=0.06, color=S[3], alpha=0.35, lw=0)
ax.text(0.08, 1.229, "design band 1.219 ± 0.02 at the centre (C3)", color=S[3], fontsize=8)
ax.axvline(1.0, color=GRID, lw=1.2); ax.set_xlabel("r / R"); ax.set_ylabel("(1 − Ψ_σ / Φ_Tolman) / w_c")
ax.set_title("(d) σ sits 1.22·w_c below Φ_Tolman at the centre; the ±3P/c² coupling\nis an input of the sources, so this is bookkeeping, not a finding", fontsize=9.5)
ax.legend(frameon=False, fontsize=7.2, loc="center right")

# (e) C1 decomposition
ax = axs[1, 1]
cats = ["derived w² offset\n(ρ = total energy)", "derived w² offset\n(ρ = rest + u = P)", "cell-sum quadrature\n(analytic metric factor)", "engine\n(C1 as scored)"]
xb = np.arange(len(cats)); wdt = 0.36
for j, wk in enumerate(WS):
    w = float(wk); g = grid[40]["w"][wk]
    ie = float(LV2[f"C1_offset_rho_rest_plus_internal_energy_exact_{'w1e-3' if wk == '0.001' else 'w1e-2'}"]["value"])
    vals = [g["exact_derived"]["C1_dev"], ie, g["quadrature"]["analytic_mf"]["C1_dev"], g["quadrature"]["engine_stored"]["C1_dev"]]
    ax.bar(xb + (j - 0.5) * wdt, [v / w**2 for v in vals], wdt, color=S[j], label=f"w_c = {w:g}")
ax.axhline(-1.0, color=INK2, lw=1, ls=":"); ax.text(3.45, -0.97, "C1 threshold\n(= w² at both w_c)", color=INK2, fontsize=7, ha="right", va="bottom")
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(xb); ax.set_xticklabels(cats, fontsize=7.5); ax.set_ylabel("(Q_σ/Q_T − (1 − w/2)/(1 + w/2)) / w²")
ax.set_ylim(-1.1, 0.15)
ax.set_title("(e) C1 is source quadrature: the cell sum alone gives 99 % of the engine value;\nthe 'exact' w² term flips sign if ρ is rest density plus internal energy", fontsize=9.5)
ax.legend(frameon=False, fontsize=7.5, loc="lower left")

# (f) scorecard
ax = axs[1, 2]; labels, ratios, cols, hatches = [], [], [], []
for k, c_ in crit.items():
    rr = c_["threshold"] / c_["value"] if c_.get("kind") == "min" else c_["value"] / c_["threshold"]
    kk_ = k.replace("-", "−").replace("design−literal", "design-literal").replace("heat−law", "heat-law").replace("^3", "³").replace("<=", "≤").replace(">=", "≥")
    labels.append(kk_.replace(" (", "\n(", 1) if len(kk_) > 44 else kk_); ratios.append(max(rr, 1e-6))
    if k.startswith("C2 literal"):
        cols.append(S[1])
    elif "BY CONSTRUCTION" in c_["tag"]:
        cols.append(S[2])
    else:
        cols.append(S[0])
yy = np.arange(len(labels))[::-1]
ax.barh(yy, ratios, color=cols, height=0.62); ax.set_xscale("log"); ax.axvline(1.0, color=INK, lw=1.2)
ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=6.0)
ax.set_xlabel("measured ÷ allowed  (left of 1 = pass; C5 as 1.2 ÷ order)")
ax.set_title("(f) criteria at 40³: all pass except C2 as sketched (fails as predicted);\nmost passes are by construction (quadrature, dt ∝ dx²)", fontsize=9.5)
ax.legend(handles=[Patch(color=S[0], label="pass, tests the solver"), Patch(color=S[2], label="pass, by construction"),
                   Patch(color=S[1], label="C2 as sketched: fail, design error")], frameon=False, fontsize=7.2, loc="lower right")
fig.suptitle("U10 (repaired): the repo's FVM heat law solves the n = 1 Tolman/σ/T problem to second order (heat-law leg = exact discrete solve, 0.094 %);\n"
             "the pilot's 0.70 % is a ((m−1)/2)·h·s split-kick bias, C2 as sketched misses w_c by a relative 2.16·w_c as predicted — engine validation, zero evidential weight",
             fontsize=10.5, color=INK, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig(os.path.join(OUT, "u10_figure_v2.png"), facecolor=SURF)

# ---------- console summary ----------
print(json.dumps(dict(criteria={k: (round(v["value"], 10), v["pass"]) for k, v in crit.items()}, summary=summary,
                      orders={k: round(v["order_lsq"], 4) for k, v in orders.items()}, power=power), indent=1))
for n in NS:
    g = grid[n]
    print(n, {k: (round(v, 8) if isinstance(v, float) else v) for k, v in g.items() if k != "w"})
    for wk in WS:
        gw = g["w"][wk]
        print("  ", wk, {k: gw[k] for k in ("C3_engine", "C3_discrete_same_sources", "C3_discrete_discrete_mf", "C3_analytic_at_inner_cells",
                                            "C3_split_fraction_of_excess", "sigma_centre_engine", "sigma_centre_discrete", "sigma_centre_model_inner_cells",
                                            "sigma_engine_over_model_minus_1", "sigma_discrete_over_model_minus_1")})
        print("     decomposition", json.dumps(gw["decomposition"]))
print(json.dumps(p7info, indent=1))
