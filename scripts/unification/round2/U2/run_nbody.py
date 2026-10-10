"""U2 leg A — N-body (field.interface.nbody velocity-Verlet, pure softened gravity) Loschmidt echo.
Forward T, reverse every velocity, integrate back T; eps_rec = rms|x(2T) - x(0)| / R.
Controls: dt-halving (t_ff/600 vs t_ff/1200), perturbation size delta at the reversal, N=32 smoke."""
import json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import u2_common as C
from sigma_ground.dynamics.seeding import to_nbody
from sigma_ground.dynamics.phase_e import _nbody_state

def echo(S, T_frac, n_per_tff, delta=0.0, keep_curve=False):
    R, tff, G, eps = S["R"], S["tff"], S["G"], S["eps"]
    dt = tff / n_per_tff; nF = int(round(T_frac * n_per_tff))
    s = to_nbody(S["bodies"], softening_m=eps)
    ms, p0, v0 = _nbody_state(s)
    E0 = sum(0.5 * m * (v[0]**2 + v[1]**2 + v[2]**2) for m, v in zip(ms, v0)) + C.u_soft(ms, p0, G, eps)
    traj = [np.array(p0)] if keep_curve else None
    t0 = time.time()
    for _ in range(nF):
        s.step(dt)
        if keep_curve: traj.append(np.array(_nbody_state(s)[1]))
    _, pT, vT = _nbody_state(s)
    ET = sum(0.5 * m * (v[0]**2 + v[1]**2 + v[2]**2) for m, v in zip(ms, vT)) + C.u_soft(ms, pT, G, eps)
    r_med_T = float(np.median(np.linalg.norm(np.array(pT) - np.average(pT, axis=0, weights=ms), axis=1)) / R)
    s.bodies = [b.replace(velocity_m_s=-b.velocity_m_s) for b in s.bodies]
    if delta:
        u = C.unit_vectors(len(s.bodies))
        s.bodies = [b.replace(position_m=b.position_m + delta * R * u[i]) for i, b in enumerate(s.bodies)]
    curve = []
    for j in range(1, nF + 1):
        s.step(dt)
        if keep_curve:
            curve.append(C.rms_rel(_nbody_state(s)[1], traj[nF - j], R))
    _, p2, v2 = _nbody_state(s)
    E2 = sum(0.5 * m * (v[0]**2 + v[1]**2 + v[2]**2) for m, v in zip(ms, v2)) + C.u_soft(ms, p2, G, eps)
    out = {"engine": "nbody", "N": S["N"], "T_tff": T_frac, "n_per_tff": n_per_tff, "dt_s": dt, "steps_each_leg": nF,
           "delta_R": delta, "eps_rec_R": C.rms_rel(p2, p0, R),
           "vel_rec_rel": float(np.sqrt(((np.array(v2) + np.array(v0)) ** 2).sum(1).mean()) /
                                np.sqrt((np.array(v0) ** 2).sum(1).mean())),
           "E_soft_J": [E0, ET, E2], "E_soft_drift_rel_fwd": abs(ET - E0) / abs(E0),
           "r_median_at_T_over_R": r_med_T, "wall_s": time.time() - t0}
    if keep_curve:
        out["backward_divergence_R"] = curve
    print(json.dumps({k: v for k, v in out.items() if k != "backward_divergence_R"}), flush=True)
    return out

if __name__ == "__main__":
    S = C.setup(64)
    runs = []
    for T in (0.3, 1.0):
        for n in (600, 1200):
            runs.append(echo(S, T, n, keep_curve=(T == 1.0 and n == 1200)))
    for d in (1e-13, 1e-11, 1e-9):
        runs.append(echo(S, 1.0, 600, delta=d))
    S32 = C.setup(32)
    runs.append(dict(echo(S32, 0.3, 600), smoke=True))
    with open(os.path.join(C.HERE, "res_nbody.json"), "w", encoding="utf-8") as f:
        json.dump({"leaves": S["leaves"], "choices": {k: S[k] for k in ("N", "M", "R", "cs", "h", "eps", "tff")},
                   "runs": runs}, f, indent=1)
    print("done")
