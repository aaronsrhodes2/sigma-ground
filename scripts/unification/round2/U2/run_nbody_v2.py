"""U2 repair round - N-body post-hoc diagnostics (predictions frozen in prereg_v2_addendum.json).

Same seed, engine (field.interface.nbody.NBodySystem.step, velocity-Verlet, Plummer softening) and dt = t_ff/600
as run_nbody.py. Modes:
  horizon_lo : forward to 4 t_ff with checkpoints at 1, 2, 3, 4 t_ff; roundoff-only echo back from each.
               The T = 1 echo must reproduce run_nbody.py's 4.687074276493767e-15 R bitwise (validation).
  horizon_hi : forward to 8 t_ff with checkpoints at 6, 8 t_ff; echo back from each.
  controls   : (A) delta = 1e-11 R injected at t = 0, full 1 t_ff echo (gain should be 1);
               (B) forward twin x0 vs x0 + delta, divergence and median radius every step to 4 t_ff;
               (C) delta = 1e-11 R injected at the reversal after 1 t_ff, divergence tracked along the backward leg;
               (D) the same after a 0.3 t_ff forward leg.
A checkpoint is the list of immutable CelestialBody records (NBodySystem.step keeps no other state between steps)."""
import json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import u2_common as C
from sigma_ground.dynamics.seeding import to_nbody
from sigma_ground.dynamics.phase_e import _nbody_state

N_PER = 600
DELTA = 1e-11


def pos(s):
    return np.array(_nbody_state(s)[1])


def r_median(p, ms, R):
    return float(np.median(np.linalg.norm(p - np.average(p, axis=0, weights=ms), axis=1)) / R)


def reverse(s):
    s.bodies = [b.replace(velocity_m_s=-b.velocity_m_s) for b in s.bodies]


def displace(s, delta, R, u):
    s.bodies = [b.replace(position_m=b.position_m + delta * R * u[i]) for i, b in enumerate(s.bodies)]


def horizon(S, T_list, T_max):
    R, tff, eps = S["R"], S["tff"], S["eps"]
    dt = tff / N_PER
    s = to_nbody(S["bodies"], softening_m=eps)
    ms, p0, _ = _nbody_state(s)
    p0 = np.array(p0)
    ck = {}
    marks = {int(round(T * N_PER)): T for T in T_list}
    rmed = []
    t0 = time.time()
    for k in range(1, int(round(T_max * N_PER)) + 1):
        s.step(dt)
        if k % 30 == 0:
            rmed.append((k / N_PER, r_median(pos(s), ms, R)))
        if k in marks:
            ck[marks[k]] = list(s.bodies)
    out = []
    for T in T_list:
        e = to_nbody(S["bodies"], softening_m=eps)
        e.bodies = list(ck[T])
        reverse(e)
        nB = int(round(T * N_PER))
        for _ in range(nB):
            e.step(dt)
        eps_rec = C.rms_rel(pos(e), p0, R)
        row = {"T_tff": T, "n_per_tff": N_PER, "steps_each_leg": nB, "eps_rec_R": eps_rec,
               "wall_s_cum": time.time() - t0}
        print(json.dumps(row), flush=True)
        out.append(row)
    return {"echo": out, "r_median_forward": rmed}


def controls(S):
    R, tff, eps = S["R"], S["tff"], S["eps"]
    dt = tff / N_PER
    u = C.unit_vectors(len(S["bodies"]))
    res = {}
    t0 = time.time()
    # (A) delta at t = 0, full echo over 1 t_ff
    s = to_nbody(S["bodies"], softening_m=eps)
    ms, p0, _ = _nbody_state(s)
    p0 = np.array(p0)
    displace(s, DELTA, R, u)
    for _ in range(N_PER):
        s.step(dt)
    reverse(s)
    for _ in range(N_PER):
        s.step(dt)
    eA = C.rms_rel(pos(s), p0, R)
    res["A_delta_at_t0_echo"] = {"delta_R": DELTA, "eps_rec_R": eA, "gain": eA / DELTA}
    print(json.dumps(res["A_delta_at_t0_echo"]), flush=True)
    # (B) forward twin to 4 t_ff
    a = to_nbody(S["bodies"], softening_m=eps)
    b = to_nbody(S["bodies"], softening_m=eps)
    displace(b, DELTA, R, u)
    tw = [(0.0, C.rms_rel(pos(a), pos(b), R) / DELTA, r_median(pos(a), ms, R))]
    for k in range(1, 4 * N_PER + 1):
        a.step(dt); b.step(dt)
        if k % 6 == 0:
            pa = pos(a)
            tw.append((k / N_PER, C.rms_rel(pa, pos(b), R) / DELTA, r_median(pa, ms, R)))
    res["B_forward_twin"] = {"delta_R": DELTA, "t_tff_gain_rmedian": tw}
    print(json.dumps({"B_gain_at": {str(t): g for t, g, _ in tw if abs(t * 10 - round(t * 10)) < 1e-9}}), flush=True)
    # (C) / (D) delta at the reversal, divergence tracked along the backward leg
    for lab, T in (("C_reversal_T1", 1.0), ("D_reversal_T0.3", 0.3)):
        nF = int(round(T * N_PER))
        ref = to_nbody(S["bodies"], softening_m=eps)
        for _ in range(nF):
            ref.step(dt)
        reverse(ref)
        per = to_nbody(S["bodies"], softening_m=eps)
        per.bodies = list(ref.bodies)
        displace(per, DELTA, R, u)
        curve = [(0.0, C.rms_rel(pos(ref), pos(per), R) / DELTA)]
        for k in range(1, nF + 1):
            ref.step(dt); per.step(dt)
            if k % 6 == 0 or k == nF:
                curve.append((k / N_PER, C.rms_rel(pos(ref), pos(per), R) / DELTA))
        eps_echo = C.rms_rel(pos(per), p0, R)
        res[lab] = {"T_tff": T, "delta_R": DELTA, "gain_end": curve[-1][1], "echo_eps_rec_R": eps_echo,
                    "echo_gain": eps_echo / DELTA, "t_back_tff_gain": curve}
        print(json.dumps({k_: v for k_, v in res[lab].items() if k_ != "t_back_tff_gain"}), flush=True)
    res["wall_s"] = time.time() - t0
    return res


if __name__ == "__main__":
    mode = sys.argv[1]
    S = C.setup(64)
    if mode == "horizon_lo":
        out = horizon(S, [1.0, 2.0, 3.0, 4.0], 4.0)
    elif mode == "horizon_hi":
        out = horizon(S, [6.0, 8.0], 8.0)
    elif mode == "controls":
        out = controls(S)
    else:
        raise SystemExit("mode?")
    out["mode"] = mode
    out["dt_s"] = S["tff"] / N_PER
    out["leaves_G"] = S["leaves"]["G"]
    out["leaves_t_ff"] = S["leaves"]["t_ff"]
    with open(os.path.join(C.HERE, f"res_nbody_v2_{mode}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("done", mode)
