"""Stage 6a — the decisive difference, measured properly: the heat capacity of a self-gravitating system.

The verification found that pilot P5 measured nothing (its two rows were one run under Newtonian similarity). This is the
design the referee asked for: fixed radius, GENUINELY different energies (initial virial ratio Q0 = 2K0/|U0| from 0.3 to
1.1 — all bound), several seeds, relax ≥ 10 t_ff, then time-average over the next 10 t_ff. Energy is the SOFTENED
Hamiltonian E = K + U_soft (what the integrator conserves; NBodySystem.total_energy() ignores softening — filed gap).

Prediction, decided before the run:
  self-gravitating, virialised:  2⟨K⟩ + ⟨W⟩ = 0  ⇒  ⟨K⟩ = −E  ⇒  d⟨K⟩/dE = −1  ⇒  C/C_gas = dE/d⟨K⟩ = −1
  ideal gas:                      K = E  ⇒  d⟨K⟩/dE = +1
with T_eff := 2⟨K⟩/(3N k_B), so C = dE/dT_eff = (3/2)N k_B · dE/d⟨K⟩, and C_gas = (3/2)N k_B (Mentat k_B).
PASS ("gravity's thermodynamics runs backwards") if the fitted slope d⟨K⟩/dE is negative at > 5σ.
Escapers (positive specific energy) are counted and reported; a second fit uses bound particles only.
"""
import json, math, os, sys, random
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from mentat_client import mentat
from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem

G, G_src = mentat("lookup_constant", {"name": "G"})
KB, KB_src = mentat("lookup_constant", {"name": "k_B"})
N, M, R = 64, 1.0e12, 5000.0
EPS = 0.02 * R
rho = 3 * M / (4 * math.pi * R ** 3); T_FF = math.sqrt(3 * math.pi / (32 * G * rho))

def u_soft(pos, m):
    d = pos[:, None, :] - pos[None, :, :]; r2 = (d ** 2).sum(-1) + EPS * EPS
    iu = np.triu_indices(len(m), 1)
    return float(-(G * np.outer(m, m)[iu] / np.sqrt(r2[iu])).sum())

def phi_i(pos, m):
    d = pos[:, None, :] - pos[None, :, :]; r2 = (d ** 2).sum(-1) + EPS * EPS
    inv = 1 / np.sqrt(r2); np.fill_diagonal(inv, 0.0)
    return -G * (inv * m[None, :]).sum(1)

def run(Q0, seed, t_relax=10.0, t_avg=10.0, steps_per_tff=200):
    rng = random.Random(seed); m = np.full(N, M / N); pos = []
    for _ in range(N):
        rr = R * rng.random() ** (1 / 3); ct = 2 * rng.random() - 1; st_ = math.sqrt(1 - ct * ct); ph = 2 * math.pi * rng.random()
        pos.append([rr * st_ * math.cos(ph), rr * st_ * math.sin(ph), rr * ct])
    pos = np.array(pos); pos -= pos.mean(0)
    vel = np.array([[rng.gauss(0, 1) for _ in range(3)] for _ in range(N)]); vel -= vel.mean(0)
    U0 = u_soft(pos, m); K = 0.5 * (m[:, None] * vel ** 2).sum(); vel *= math.sqrt(Q0 * abs(U0) / (2 * K))
    bodies = [CelestialBody(mass_kg=m[i], position_m=pos[i], velocity_m_s=vel[i], radius_m=1.0, love_number_k2=0.0) for i in range(N)]
    sysn = NBodySystem(bodies, softening_m=EPS)
    dt = T_FF / steps_per_tff
    def state():
        p = np.array([b.position_m for b in sysn.bodies]); v = np.array([b.velocity_m_s for b in sysn.bodies]); return p, v
    p, v = state(); E0 = 0.5 * (m[:, None] * v ** 2).sum() + u_soft(p, m)
    for _ in range(int(t_relax * steps_per_tff)):
        sysn.step(dt)
    Ks, Kb, Es, nesc = [], [], [], []
    for k in range(int(t_avg * steps_per_tff)):
        sysn.step(dt)
        if k % 20 == 0:
            p, v = state(); ke_i = 0.5 * m * (v ** 2).sum(1)
            e_i = ke_i + m * phi_i(p, m)
            bound = e_i < 0
            Ks.append(ke_i.sum()); Kb.append(ke_i[bound].sum()); Es.append(ke_i.sum() + u_soft(p, m)); nesc.append(int((~bound).sum()))
    return dict(Q0=Q0, seed=seed, E0=E0, E_mean=float(np.mean(Es)), E_drift=float(abs(Es[-1] / E0 - 1)),
                K_mean=float(np.mean(Ks)), K_bound_mean=float(np.mean(Kb)), escapers_mean=float(np.mean(nesc)),
                virial_Q=float(2 * np.mean(Ks) / abs(np.mean(Es) - np.mean(Ks))))

GRID = [(Q0, seed) for Q0 in (0.3, 0.5, 0.7, 0.9, 1.1) for seed in (11, 12, 13, 14)]

if __name__ == "__main__":
    # Usage:  (no args)               run the whole grid serially, then fit
    #         --one Q0 SEED           run one cell, print its JSON row (for parallel runs)
    #         --resume ROWS.jsonl     reuse rows already printed (one JSON object per line), run the missing cells, fit
    if len(sys.argv) >= 4 and sys.argv[1] == "--one":
        r = run(float(sys.argv[2]), int(sys.argv[3]))
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        sys.exit(0)
    done = {}
    if len(sys.argv) >= 3 and sys.argv[1] == "--resume":
        for line in open(sys.argv[2], encoding="utf-8"):
            line = line.strip()
            if line.startswith("{") and '"Q0"' in line:
                r = json.loads(line); done[(r["Q0"], r["seed"])] = r
    rows = []
    for Q0, seed in GRID:
        if (Q0, seed) in done:
            rows.append(done[(Q0, seed)]); continue
        r = run(Q0, seed); rows.append(r); print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    E = np.array([r["E0"] for r in rows]); K = np.array([r["K_mean"] for r in rows]); Kb = np.array([r["K_bound_mean"] for r in rows])
    def fit(x, y):
        A = np.vstack([x, np.ones_like(x)]).T; coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef; s2 = (resid ** 2).sum() / (len(x) - 2); cov = s2 * np.linalg.inv(A.T @ A)
        return float(coef[0]), float(math.sqrt(cov[0, 0]))
    s_all, e_all = fit(E, K); s_b, e_b = fit(E, Kb)
    out = dict(sources=dict(G=G_src, k_B=KB_src), N=N, T_ff_s=T_FF, softening_m=EPS, rows=rows,
               slope_dK_dE_all=s_all, slope_err_all=e_all, C_over_Cgas_all=1 / s_all,
               slope_dK_dE_bound=s_b, slope_err_bound=e_b, C_over_Cgas_bound=1 / s_b,
               significance_negative_sigma=abs(s_all) / e_all,
               C_gas_J_per_K=1.5 * N * KB, prediction_virial=-1.0, prediction_ideal_gas=+1.0)
    json.dump(out, open("misc/unification/stage6_heat_capacity.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
