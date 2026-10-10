"""U6 v2 (repair round; numerics IDENTICAL to u6_sigma_wave.py, only I/O changed) — sigma wave sourced by a pulsating n = 1 polytrope. NEW scratch utility (no wave solver exists in the tree).

    (1/c^2) sigma_tt - lap(sigma) = s(t, r),   s = (8 pi G xi / c^4)(rho c^2 - 3 P)      [box sigma = -xi R, perfect fluid]
    v = r sigma:   v_tt = c^2 v_rr + c^2 r s,   v(0) = 0 (regularity),   Sommerfeld v_N^{n+1} = v_{N-1}^n (exact at CFL 1)

Leapfrog at CFL = 1 (c dt = dr):  v_i^{n+1} = v_{i+1}^n + v_{i-1}^n - v_i^{n-1} + f_i^n,  f_i = dt^2 c^2 r_i s_i = dr^2 r_i s_i.
Two source discretisations:
  'point' - the design sketch: s sampled at the nodes.
  'cell'  - conservative: f_j = dr * Q_j / r_j with Q_j = exact integral of r^2 s over cell j (cell 1 absorbs [0, dr/2]);
            the discrete static Green's function is G_ij = min(i, j) dr, so the far-field monopole sum_j r_j^2 s_j dr = sum_j Q_j
            equals the continuum integral exactly -> mass conservation is exact on the grid.
All physical leaves (G, c, rho_c, r_s, ...) come from mentat_leaves.json (Mentat, PLATINUM 21); xi from sigma_ground.inventory.core.sigma.
Run from the repo root:  python -X utf8 u6_sigma_wave_v2.py --set A|B|C
v2 changes: reads mentat_leaves_v2.json (re-fetched, same values); writes results_v2_*.json / traces_v2_*.npz;
also stores FULL-RESOLUTION probe traces of the omega R/c = 0.01 runs (fulltraces_v2_*.npz) for the reviewer-requested S3 window sweep.
"""
import sys, os, json, math, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(os.path.join(HERE, "mentat_leaves_v2.json"), encoding="utf-8"))
G = float(L["G"]["value"]); C = float(L["c"]["value"]); XI = float(L["XI"]["value"])
CH = L["choices"]; a = CH["a_m"]; R = math.pi * a; W_C = CH["w_c"]; EPS = CH["eps"]
RHO_C = float(L["rho_c_kg_m3"]["value"]); K = 2 * math.pi * G * a * a          # n = 1: a^2 = K/(2 pi G)
COEF = 8 * math.pi * G * XI / C**4                                                 # s = COEF (rho c^2 - 3P)


def grid(N):
    dr = 40 * R / N
    r = np.arange(N + 1) * dr
    return r, dr


def fm(rr, b):   # antiderivative of r sin(r/b)  (Mentat integrate_expr: b^2 sin(r/b) - b r cos(r/b)), clipped at the surface pi b
    x = np.minimum(rr, math.pi * b)
    return b * b * np.sin(x / b) - b * x * np.cos(x / b)


def fp(rr):      # antiderivative of sin^2(r/a)  (Mentat integrate_expr: r/2 - a sin(r/a) cos(r/a)/2), clipped at R
    x = np.minimum(rr, R)
    return x / 2 - a * np.sin(x / a) * np.cos(x / a) / 2


class Source:
    def __init__(self, N, kind, mode, eps, om):
        self.r, self.dr = grid(N)
        self.kind, self.mode, self.eps, self.om = kind, mode, eps, om
        self.m = int(math.ceil(R * (1 + max(eps, 0.0)) / self.dr)) + 3
        j = np.arange(1, self.m + 1)
        self.rj = j * self.dr
        lo = (j - 0.5) * self.dr; lo[0] = 0.0
        hi = (j + 0.5) * self.dr
        self.lo, self.hi = lo, hi
        # frozen pressure part, cell integral of r^2 P = K rho_c^2 a^2 * int sin^2(r/a) dr
        self.Pcell = K * RHO_C**2 * a * a * (fp(hi) - fp(lo))
        x = np.maximum(self.rj / a, 1e-12)
        self.rho0_pt = RHO_C * np.where(self.rj < R, np.sin(x) / x, 0.0)
        self.P0_pt = K * self.rho0_pt**2
        self.Mcell0 = self.mass_cell(1.0)

    def mass_cell(self, lam):   # exact cell integral of r^2 rho for rho = rho0(r/lam)/lam^3
        b = a * lam
        return (RHO_C * b / lam**3) * (fm(self.hi, b) - fm(self.lo, b))

    def f(self, t):
        """f_j for j = 1..m (array of length m)."""
        sn = math.sin(self.om * t) if self.om > 0 else 0.0
        if self.kind == "cell":
            if self.mode == "mass":
                Mc = self.mass_cell(1 + self.eps * sn); Pc = self.Pcell
            else:
                Mc = self.Mcell0; Pc = self.Pcell * (1 + self.eps * sn)
            Q = COEF * (C * C * Mc - 3 * Pc)
            return self.dr * Q / self.rj
        # point sampling (design sketch)
        if self.mode == "mass":
            lam = 1 + self.eps * sn
            xx = np.maximum(self.rj / (a * lam), 1e-12)
            rho = RHO_C * np.where(self.rj < R * lam, np.sin(xx) / xx, 0.0) / lam**3
            P = self.P0_pt
        else:
            rho = self.rho0_pt; P = self.P0_pt * (1 + self.eps * sn)
        s = COEF * (rho * C * C - 3 * P)
        return self.dr * self.dr * self.rj * s


def run(N, kind, mode, wR, eps=EPS, periods=6, nsteps=None):
    r, dr = grid(N); dt = dr / C
    om = wR * C / R if wR > 0 else 0.0
    T = 2 * math.pi / om if om > 0 else None
    if nsteps is None:
        nsteps = int(math.ceil(periods * T / dt))
    src = Source(N, kind, mode, eps, om); m = src.m
    ip = int(round(30 * R / dr))
    v = np.zeros(N + 1); vo = np.zeros(N + 1); vn = np.zeros(N + 1)
    probe = np.zeros(nsteps + 1)
    t0 = time.time()
    const_f = None
    if om == 0.0 or eps == 0.0:
        const_f = src.f(0.0)
    for n in range(nsteps):
        fj = const_f if const_f is not None else src.f(n * dt)
        np.add(v[2:], v[:-2], out=vn[1:-1])
        vn[1:-1] -= vo[1:-1]
        vn[1:m + 1] += fj
        vn[0] = 0.0
        vn[-1] = v[-2]
        vo, v, vn = v, vn, vo
        probe[n + 1] = v[ip]
    wall = time.time() - t0
    return dict(N=N, kind=kind, mode=mode, wR=wR, eps=eps, dr=dr, dt=dt, ip=ip, r_probe=float(r[ip]), om=om, T=T,
                nsteps=nsteps, wall_s=wall, static_monopole_sumQ=float(np.sum(COEF * (C * C * src.Mcell0 - 3 * src.Pcell)))), probe


def fit_basis(tau, om, nh=4):
    cols = [np.ones_like(tau)]
    for k in range(1, nh + 1):
        cols += [np.sin(k * om * tau), np.cos(k * om * tau)]
    return np.vstack(cols).T


def analyse(meta, probe, A_pred=None):
    dt, om, T, ip = meta["dt"], meta["om"], meta["T"], meta["ip"]
    dr = meta["dr"]; rp = meta["r_probe"]
    n = np.arange(probe.size); t = n * dt
    iR = R / dr
    nz = np.nonzero(np.abs(probe) > 1e-30)[0]
    first = int(nz[0]) if nz.size else None
    out = dict(first_arrival_step=first, light_cone_steps=(ip - iR),
               arrival_offset_cells=(first - (ip - iR)) if first is not None else None,
               arrival_over_rmR_c=(first * dt) / ((rp - R) / C) if first is not None else None)
    if om == 0.0:
        return out
    tend = t[-1]
    w = t > tend - 3 * T
    tau = t[w] - rp / C
    coef, *_ = np.linalg.lstsq(fit_basis(tau, om), probe[w], rcond=None)
    A = float(math.hypot(coef[1], coef[2])); phase = float(math.atan2(coef[2], coef[1]))
    A_std = float(np.std(probe[w]) * math.sqrt(2))
    out.update(A_lockin=A, phase_rad=phase, A_2w=float(math.hypot(coef[3], coef[4])), dc=float(coef[0]), A_std_sqrt2=A_std,
               A_pred=A_pred, ratio_lockin=(A / A_pred) if A_pred else None, ratio_std=(A_std / A_pred) if A_pred else None)
    # S3: residual of the steady-state fit around the sonic arrival time of the fastest (central) sound speed
    cs_over_c = float(L["cs_centre_over_c"]["value"])
    t_cs = (rp - R) / (cs_over_c * C)
    t_ss = (rp + R) / C + 10 * R / C
    if t_cs - 20 * R / C > t_ss and t_cs + 20 * R / C < tend:
        allw = t > t_ss
        tau2 = t[allw] - rp / C
        res = probe[allw] - fit_basis(tau2, om) @ coef
        tt = t[allw]
        ws = np.abs(tt - t_cs) < 20 * R / C
        wc = np.abs(tt - (t_cs + 0.5 * (tend - t_cs))) < 20 * R / C
        out.update(t_cs_over_R_c=t_cs / (R / C), res_max_over_A_at_cs=float(np.max(np.abs(res[ws])) / A),
                   res_max_over_A_control=float(np.max(np.abs(res[wc])) / A), res_rms_over_A_all=float(np.sqrt(np.mean(res**2)) / A))
    else:
        out.update(t_cs_over_R_c=t_cs / (R / C), res_note="sonic arrival time outside the steady-state part of the run")
    return out


def A_pred(mode, wR):
    if mode == "pressure":
        return float(L["A_press_m"]["value"])
    return float(L[f"A_mass_m_wR{wR}"]["value"])


SETS = {
    "A": [(16000, "cell", md, q) for md in ("pressure", "mass") for q in (0.01, 0.02, 0.04, 0.08)],
    "B": [(16000, "point", md, q) for md in ("pressure", "mass") for q in (0.01, 0.02, 0.04, 0.08)],
    "C": [(Nn, kd, md, q) for (Nn, kd) in ((8000, "cell"), (8000, "point"), (4000, "cell"), (4000, "point"))
          for md in ("pressure", "mass") for q in (0.01, 0.02, 0.04, 0.08)],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True); args = ap.parse_args()
    results = []; traces = {}; full = {}
    if args.set == "C":
        # S2 ramp: static switch-on, eps = 0, N = 8000 and 16000 (cell source)
        for Nn in (8000, 16000):
            r, dr = grid(Nn); ip = int(round(30 * R / dr)); iR = int(round(R / dr))
            meta, probe = run(Nn, "cell", "pressure", 0.0, eps=0.0, nsteps=ip + iR + 600)
            meta.update(analyse(meta, probe)); meta["label"] = "ramp"
            results.append(meta); traces[f"ramp_N{Nn}"] = probe
            print(json.dumps({k: meta[k] for k in ("N", "first_arrival_step", "arrival_offset_cells", "wall_s")}), flush=True)
    for (Nn, kd, md, q) in SETS[args.set]:
        meta, probe = run(Nn, kd, md, q)
        meta.update(analyse(meta, probe, A_pred(md, q)))
        results.append(meta)
        dec = max(1, probe.size // 20000)
        traces[f"{kd}_{md}_{q}_N{Nn}"] = probe[::dec]; traces[f"{kd}_{md}_{q}_N{Nn}_dec"] = np.array([dec])
        if abs(q - 0.01) < 1e-12:
            full[f"{kd}_{md}_{q}_N{Nn}"] = probe
        print(json.dumps({k: meta.get(k) for k in ("N", "kind", "mode", "wR", "A_lockin", "ratio_lockin", "ratio_std", "phase_rad",
                                                    "arrival_offset_cells", "res_max_over_A_at_cs", "wall_s")}), flush=True)
        json.dump(results, open(os.path.join(HERE, f"results_v2_{args.set}.json"), "w", encoding="utf-8"), indent=1)
    json.dump(results, open(os.path.join(HERE, f"results_v2_{args.set}.json"), "w", encoding="utf-8"), indent=1)
    np.savez_compressed(os.path.join(HERE, f"traces_v2_{args.set}.npz"), **traces)
    np.savez_compressed(os.path.join(HERE, f"fulltraces_v2_{args.set}.npz"), **full)
    print("done", args.set, flush=True)
