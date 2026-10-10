"""U10 solver run at one resolution: python -X utf8 u10_run.py <n>   (run from the repo root, read-only on the repo).
Fixed physical box L = 40 km, a = 5 km (n = 40 -> dx = 1 km, the design grid). n = 1 polytrope.
Solves with the pilot's poisson_fvm (the repo's dynamics.fields.heat.diffuse_fvm run to steady state, kick every 20 stable_dt):
  v_N   : rho/rho_c                          (Phi_N = -4 pi G rho_c v_N)        -- also the design-literal T leg (called twice)
  v_P   : 3P/(c^2 rho_c) / w                 (pressure shape theta^2; unweighted pair by linearity)
  v_T^w, v_s^w : mf*(rho +- 3P/c^2)/rho_c    (mf = 1 - 2 Phi_N/c^2), per w_c in {1e-3, 1e-2}
plus a separate steady-heat run in heat units (k = 1 W/m/K, q = q0 rho/rho_c, T_inf = 300 K, kick every 1 stable_dt, n_tau = 16)."""
import json, math, os, sys, time
import numpy as np
REPO = r"D:/Aaron/development/sigma-ground"; sys.path.insert(0, REPO)
OUT = os.path.dirname(os.path.abspath(__file__))
from sigma_ground.dynamics.fields.heat import diffuse_fvm, stable_dt
from sigma_ground.inventory.core.sigma import XI
ns = {}   # the pilot's own solver, exactly as the design sketch: imports + poisson_fvm only
exec(compile(open(os.path.join(REPO, "scripts/unification/pilot_p7_sigma_three_fields.py"), encoding="utf-8").read().split("\nn, dx, a_cells")[0],
             "pilot_p7_solver", "exec"), ns)
poisson_fvm = ns["poisson_fvm"]
LV = json.load(open(os.path.join(OUT, "u10_leaves.json")))
G, C = LV["G"]["value"], LV["c"]["value"]          # Mentat lookup_constant (source sigma_ground.field.constants)

n = int(sys.argv[1]); Lbox, a = 40000.0, 5000.0; dx = Lbox / n; R = math.pi * a
c0 = (n - 1) / 2; xs = (np.arange(n) - c0) * dx
X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij"); r = np.sqrt(X**2 + Y**2 + Z**2)
xi = np.maximum(r / a, 1e-12); theta = np.where(r < R, np.sin(xi) / xi, 0.0)
wall = np.zeros((n, n, n), bool); wall[0] = wall[-1] = True; wall[:, 0] = wall[:, -1] = True; wall[:, :, 0] = wall[:, :, -1] = True
inner = r < 0.9 * dx                                  # the 8 cells around the centre (r = sqrt(3)/2 dx)
# exact n = 1 potential of the source theta: a^2 (1 + theta) inside, pi a^3 / r outside (Mentat simplify_expr: Laplacian residual 0)
v_an_N = np.where(r < R, a * a * (1 + theta), math.pi * a**3 / np.maximum(r, 1e-12))
timing = {}


def gauss_flux(v, hw_cells):
    """Discrete Gauss monopole: dx * sum over the faces of the cube |x|,|y|,|z| < hw_cells*dx of (v_in - v_out)."""
    ins = (np.abs(X) < hw_cells * dx) & (np.abs(Y) < hw_cells * dx) & (np.abs(Z) < hw_cells * dx)
    q = 0.0
    for ax in range(3):
        lo = tuple(slice(None, -1) if b == ax else slice(None) for b in range(3))
        hi = tuple(slice(1, None) if b == ax else slice(None) for b in range(3))
        a_in, b_in = ins[lo], ins[hi]; d = v[lo] - v[hi]
        q += float(d[a_in & ~b_in].sum() - d[~a_in & b_in].sum())
    return q * dx


hw = [n // 2 - 1, n // 2 - 2]   # cube half-widths in cells; both > R = 15.71 km at every n used
assert all(h * dx > R for h in hw)


def solve(key, s):
    t0 = time.time(); v, _, Q = poisson_fvm(s, dx); timing[key] = round(time.time() - t0, 1)
    ext = (r > R + 1.5 * dx) & ~wall & (np.abs(X) < hw[1] * dx) & (np.abs(Y) < hw[1] * dx) & (np.abs(Z) < hw[1] * dx)
    qr = 4 * np.pi * r[ext] * v[ext]
    return v, dict(Q_quad=Q, Q_flux=[gauss_flux(v, h) for h in hw], Q_exterior_4pirv_median=float(np.median(qr)),
                   Q_exterior_4pirv_minmax=[float(qr.min()), float(qr.max())])


def errs(v, v_an, norm):
    m = ~wall; d = (v - v_an)[m]
    return dict(max_abs_rel_to_peak=float(np.abs(d).max() / norm), rms_rel_to_peak=float(np.sqrt(np.mean(d**2)) / norm),
                centre_signed_rel=float(np.mean((v - v_an)[inner]) / np.mean(v_an[inner])), argmax_r_m=float(r[m][np.argmax(np.abs(d))]))


res = dict(n=n, dx_m=dx, a_m=a, R_m=R, L_box_m=Lbox, a_cells=a / dx, cube_half_widths_m=[h * dx for h in hw],
           r_inner_m=float(r[inner].mean()), n_inner=int(inner.sum()), theta_inner=float(theta[inner].mean()))
fields = {}
v_N, res["N"] = solve("N", theta)
v_Tleg_design, res["T_leg_design"] = solve("T_leg_design", theta)       # the sketch's Tf = poisson_fvm(rho/rho_c): same call as N
res["T_leg_design_identical_to_N_bitwise"] = bool(np.array_equal(v_N, v_Tleg_design))
v_P, res["P_shape"] = solve("P_shape", theta**2)
res["T_leg_design_err"] = errs(v_Tleg_design, v_an_N, 2 * a * a)
res["N_flux_over_quad_minus_1"] = [qf / res["N"]["Q_quad"] - 1 for qf in res["N"]["Q_flux"]]
res["N_quad_over_analytic_minus_1"] = res["N"]["Q_quad"] / (4 * math.pi**2 * a**3) - 1
fields.update(r=r, v_N=v_N, v_P=v_P, theta=theta, wall=wall)

# --- the heat-law T leg in heat units, kick every 1 stable_dt, n_tau = 16 ---
k_W, rcp, q0, T_inf = 1.0, 1.0, 2e-6, 300.0
alpha = k_W / rcp; q = q0 * theta
T_wall = T_inf + (q0 / k_W) * float(theta.sum() * dx**3) / (4 * np.pi * np.maximum(r, dx / 2))
T = np.full((n, n, n), T_inf); kk = np.full((n, n, n), k_W); rc = np.full((n, n, n), rcp)
tau = (n * dx)**2 / (3 * np.pi**2 * alpha); dt1 = stable_dt(alpha, dx); t = 0.0; t0 = time.time(); hist = []
while t < 16 * tau:
    T, _ = diffuse_fvm(T, kk, rc, dx, total_time=dt1, ambient_mask=wall, ambient_T=T_wall, max_substeps=10**6)
    T += q * dt1 / rcp; t += dt1
    if not hist or t - hist[-1][0] * tau >= tau / 4:
        hist.append((t / tau, float(np.mean(T[inner]))))
timing["T_leg_heat_1x"] = round(time.time() - t0, 1)
T_an = T_inf + (q0 / k_W) * v_an_N
res["T_leg_heat"] = dict(k_W_mK=k_W, rho_cp_J_m3K=rcp, q0_W_m3=q0, T_inf_K=T_inf, kick_dt_s=dt1, n_tau=16,
                         centre_T_K=float(np.mean(T[inner])), centre_T_analytic_K=float(np.mean(T_an[inner])),
                         Q_flux_over_quad_minus_1=[gauss_flux(T - T_inf, h) / float(q.sum() * dx**3 / k_W) - 1 for h in hw],
                         centre_history_t_over_tau=hist)
res["T_leg_heat_err"] = errs(T - T_inf, T_an - T_inf, 2 * a * a * q0 / k_W)
dt20 = stable_dt(1.0, dx) * 20
d_split = v_Tleg_design - (T - T_inf) * k_W / q0
res["split_check"] = dict(centre_diff_measured=float(np.mean(d_split[inner])),
                          centre_diff_predicted=float((dt20 - dt1) / 2 * np.mean(theta[inner])),
                          centre_diff_over_v_centre=float(np.mean(d_split[inner]) / np.mean(v_an_N[inner])),
                          note="units of v (alpha = 1 in both runs); prediction (dt_20 - dt_1)/2 * s from the split-kick fixed point T* = v + (dt/2) s")
fields["v_heat"] = (T - T_inf) * k_W / q0

# --- the order-consistent 1PN pair, per w_c ---
res["w"] = {}
for w_c in (1e-3, 1e-2):
    rho_c = w_c * C**2 / (6 * math.pi * G * a * a); K = 2 * math.pi * G * a * a
    rho = rho_c * theta; P = K * rho**2
    mf = 1 + 2 * 4 * math.pi * G * rho_c * v_N / C**2          # 1 - 2 Phi_N / c^2, Phi_N from the solver (design sketch)
    v_T, dT = solve(f"T_{w_c:g}", mf * (rho + 3 * P / C**2) / rho_c)
    v_S, dS = solve(f"S_{w_c:g}", mf * (rho - 3 * P / C**2) / rho_c)
    QN = res["N"]["Q_quad"]; QP = res["P_shape"]["Q_quad"]
    row = dict(w_c=w_c, rho_c_kg_m3=rho_c, M_kg=QN * rho_c, rs_over_R=2 * G * QN * rho_c / C**2 / R,
               w_check_3Pc_over_rhoc_c2=3 * K * rho_c / C**2, T=dT, S=dS,
               Q_S_over_Q_T=dS["Q_quad"] / dT["Q_quad"], dQ_over_QN=(dT["Q_quad"] - dS["Q_quad"]) / QN, QT_over_QN=dT["Q_quad"] / QN,
               Q_S_over_Q_T_flux=[s_ / t_ for s_, t_ in zip(dS["Q_flux"], dT["Q_flux"])],
               dQ_over_QN_flux=[(t_ - s_) / n_ for s_, t_, n_ in zip(dS["Q_flux"], dT["Q_flux"], res["N"]["Q_flux"])],
               QT_over_QN_flux=[t_ / n_ for t_, n_ in zip(dT["Q_flux"], res["N"]["Q_flux"])],
               unweighted_Q_S_over_Q_T=(QN - w_c * QP) / (QN + w_c * QP), unweighted_dQ_over_QN=2 * w_c * QP / QN,
               deficit_over_w_weighted=float(np.mean(1 - v_S[inner] / v_T[inner]) / w_c),
               deficit_over_w_unweighted=float(np.mean(1 - (v_N - w_c * v_P)[inner] / (v_N + w_c * v_P)[inner]) / w_c),
               sigma_centre=float(np.mean((2 * XI / C**2) * 4 * math.pi * G * rho_c * v_S[inner])),
               sigma_centre_if_sourced_like_Phi_T=float(np.mean((2 * XI / C**2) * 4 * math.pi * G * rho_c * v_T[inner])),
               sigma_surface_xi_rs_over_R=XI * 2 * G * QN * rho_c / C**2 / R)
    res["w"][f"{w_c:g}"] = row
    fields[f"v_T_{w_c:g}"] = v_T; fields[f"v_S_{w_c:g}"] = v_S
res["timing_s"] = timing
json.dump(res, open(os.path.join(OUT, f"u10_n{n}.json"), "w"), indent=1)
np.savez_compressed(os.path.join(OUT, f"u10_fields_n{n}.npz"), **fields)
print(json.dumps({k: v for k, v in res.items() if k != "T_leg_heat"}, indent=1)[:8000])
