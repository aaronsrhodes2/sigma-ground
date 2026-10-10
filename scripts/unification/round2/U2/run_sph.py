"""U2 legs B/C - SPH continuum (dynamics.continuum, committed first law) Loschmidt echo + entropy-production ledger.

Per step (ledger=True):
  * independent viscous ledger at the pre-step state (x_n, v_n): P_v = sum m v.a_visc with
    _sph_viscous_force(..., c_override = the engine's own signal speed), S_dot_led = -P_v / T;
    Q_led = trapezoid of -P_v dt.
  * the engine's own booking, decomposed pair by pair into pressure (PdV) and viscous shares, from the very
    pair-force lists continuum_step applied (scene._vv_pairs; the first len(neighbors) entries of each parcel list
    are pressure, the rest viscosity) with the step-averaged velocities, exactly as continuum_step books them.
  * scene.energy_ledger() every step (K, internal, bath, softened self-gravity).
  * isothermal free energy F = c_s^2 sum m ln rho_sph (the state function the isothermal PdV work integrates to).
Reversal: all velocities flipped after the forward leg; the velocity-Verlet force cache is invalidated (forces are
re-evaluated at the reversed state; bitwise irrelevant for alpha = 0, required for alpha = 1 because viscosity depends
on velocity). Optional displacement delta*R (seeded unit vectors) at the reversal for the perturbation-size control."""
import json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import u2_common as C
from sigma_ground.dynamics.seeding import to_continuum
from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.continuum import (continuum_step, _sph_density_sum, _sph_viscous_force,
    _sph_pressure_force, _pressure_isothermal, ContinuumParcel, ContinuumScene, _R_GAS)
from sigma_ground.dynamics.phase_e import _sph_state


def build_scene(S, eos):
    if eos == "isothermal":
        return to_continuum(S["bodies"], material_key="water_ice", G=S["G"], gravity_softening_m=S["eps"],
                            eos="isothermal", sound_speed_m_s=S["cs"], smoothing_h=S["h"])
    # eos="ideal": same parcel construction as seeding.to_continuum (which has no ideal-EOS pass-through)
    from sigma_ground.field.interface.surface import MATERIALS
    rho = MATERIALS["water_ice"]["density_kg_m3"]
    parcels = [ContinuumParcel("water_ice", (3.0 * b.mass_kg / (4.0 * math.pi * rho)) ** (1.0 / 3.0),
                               temperature=300.0, position=Vec3(*b.position_m), velocity=Vec3(*b.velocity_m_s),
                               label=b.label) for b in S["bodies"]]
    R_gas = S["leaves"]["R_gas"]["value"]
    M_mol = R_gas * 300.0 / S["cs"] ** 2   # Choice: isothermal sound speed at 300 K equals c_s -> same P(rho) at t=0
    sc = ContinuumScene(parcels, gravity=Vec3(0.0, 0.0, 0.0), ground=False, smoothing_h=S["h"],
                        self_gravity_G=S["G"], gravity_softening_m=S["eps"], eos="ideal",
                        ideal_molar_mass_kg_mol=M_mol, ideal_gamma=S["leaves"]["gamma_monatomic"]["value"])
    sc.heat_conduction = False              # adiabatic gas: Fourier conduction is the other leg
    return sc


def signal_speed(sc, ps):
    if sc.eos == "ideal":
        r_spec = _R_GAS / sc.ideal_molar_mass_kg_mol
        return math.sqrt(sc.ideal_gamma * r_spec * max(p.temperature for p in ps))
    return sc.sound_speed_m_s


def visc_power(sc, ps, alpha):
    _sph_density_sum(ps, sc.h)
    a = _sph_viscous_force(ps, sc.h, alpha, c_override=signal_speed(sc, ps))
    return sum(p.mass * (p.velocity.x * a[i].x + p.velocity.y * a[i].y + p.velocity.z * a[i].z)
               for i, p in enumerate(ps))


def replicate_pairs(sc, ps, alpha):
    """Pair lists continuum_step's _forces() records at the CURRENT state (used when its cache is empty)."""
    h = sc.h
    _sph_density_sum(ps, h)
    if sc.eos == "ideal":
        r_spec = _R_GAS / sc.ideal_molar_mass_kg_mol
        for p in ps:
            p.pressure = p.sph_density * r_spec * p.temperature
    else:
        for p in ps:
            _pressure_isothermal(p, sc.sound_speed_m_s)
    pairs = [[] for _ in ps]
    _sph_pressure_force(ps, h, pairs=pairs)
    _sph_viscous_force(ps, h, alpha, c_override=signal_speed(sc, ps), pairs=pairs)
    return pairs


def cache_valid(sc, ps):
    acc = getattr(sc, "_vv_accel", None)
    pr = getattr(sc, "_vv_pairs", None)
    return not (acc is None or len(acc) != len(ps) or pr is None or len(pr) != len(ps))


def echo(S, eos, alpha, T_frac, n_per_tff, delta=0.0, ledger=True, keep_curve=False, label=""):
    R, tff = S["R"], S["tff"]
    dt = tff / n_per_tff
    nF = int(round(T_frac * n_per_tff))
    sc = build_scene(S, eos)
    ps = sc.continuum_parcels()
    ms = np.array([p.mass for p in ps])
    _, p0, v0 = _sph_state(sc)
    cs = S["cs"]
    r_spec = (_R_GAS / sc.ideal_molar_mass_kg_mol) if eos == "ideal" else None
    c_v = (r_spec / (sc.ideal_gamma - 1.0)) if eos == "ideal" else None
    traj = [np.array(p0)] if keep_curve else None
    ser = {k: [] for k in ("t_s", "K", "U_soft", "bath", "internal", "total", "F_iso", "S_state", "Q_led",
                           "Sdot_led", "q_visc_eng", "q_press_eng", "dS_visc_eng")}
    t0 = time.time()

    def snapshot(t, Pv, Qled, qv, qp, dSv):
        led = sc.energy_ledger()
        ser["t_s"].append(t)
        ser["K"].append(led["kinetic"])
        ser["U_soft"].append(led["self_gravity"])
        ser["bath"].append(led["bath"])
        ser["internal"].append(led["internal"])
        ser["total"].append(led["total"])
        _sph_density_sum(ps, sc.h)
        ser["F_iso"].append(cs * cs * float(sum(p.mass * math.log(p.sph_density) for p in ps)))
        ser["S_state"].append(float(sum(p.mass * (c_v * math.log(p.temperature) - r_spec * math.log(p.sph_density))
                                        for p in ps)) if eos == "ideal" else 0.0)
        ser["Q_led"].append(Qled)
        T_ref = 300.0 if eos == "isothermal" else float(np.mean([p.temperature for p in ps]))
        ser["Sdot_led"].append(-Pv / T_ref)
        ser["q_visc_eng"].append(qv)
        ser["q_press_eng"].append(qp)
        ser["dS_visc_eng"].append(dSv)

    Qled = 0.0
    Qv_eng = 0.0
    Qp_eng = 0.0
    S_visc_eng = 0.0
    min_qv_step = math.inf
    n_neg_qv = 0
    book_err = 0.0
    Pv = visc_power(sc, ps, alpha) if ledger else 0.0
    if ledger:
        snapshot(0.0, Pv, 0.0, 0.0, 0.0, 0.0)
    t = 0.0
    curve = []
    marks = {}
    for k in range(2 * nF):
        if ledger:
            _sph_density_sum(ps, sc.h)            # state n: neighbours for the pressure/viscous split
            nb0 = [len(p.neighbors) for p in ps]
            pairs0 = sc._vv_pairs if cache_valid(sc, ps) else replicate_pairs(sc, ps, alpha)
            v_old = [(p.velocity.x, p.velocity.y, p.velocity.z) for p in ps]
            T_old = [p.temperature for p in ps]
            e_before = sc.energy_bath_J + sum(p.internal_energy for p in ps)
        continuum_step(sc, dt, alpha_visc=alpha)
        t += dt
        if ledger:
            pairs1 = sc._vv_pairs
            nb1 = [len(p.neighbors) for p in ps]
            vbar = [(0.5 * (vo[0] + p.velocity.x), 0.5 * (vo[1] + p.velocity.y), 0.5 * (vo[2] + p.velocity.z))
                    for vo, p in zip(v_old, ps)]
            qv = qp = dSv = 0.0
            for i, p in enumerate(ps):
                bi = vbar[i]
                wp = wv = 0.0
                for lst, nb in ((pairs0[i], nb0[i]), (pairs1[i], nb1[i])):
                    for idx, (j, gx, gy, gz) in enumerate(lst):
                        bj = vbar[j]
                        d = gx * (bi[0] - bj[0]) + gy * (bi[1] - bj[1]) + gz * (bi[2] - bj[2])
                        if idx < nb:
                            wp += d
                        else:
                            wv += d
                qvi = 0.25 * dt * wv * p.mass
                qpi = 0.25 * dt * wp * p.mass
                qv += qvi
                qp += qpi
                dSv += qvi / (0.5 * (T_old[i] + p.temperature))
            e_after = sc.energy_bath_J + sum(p.internal_energy for p in ps)
            book_err = max(book_err, abs((e_after - e_before) - (qv + qp)) / max(abs(qv) + abs(qp), 1e-300))
            Qv_eng += qv
            Qp_eng += qp
            S_visc_eng += dSv
            if alpha > 0:
                min_qv_step = min(min_qv_step, qv)
                n_neg_qv += int(qv < 0)
            Pv_end = visc_power(sc, ps, alpha)
            Qled += -0.5 * (Pv + Pv_end) * dt
            Pv = Pv_end
            snapshot(t, Pv, Qled, Qv_eng, Qp_eng, S_visc_eng)
        if keep_curve and k < nF:
            traj.append(np.array(_sph_state(sc)[1]))
        if keep_curve and k >= nF:
            curve.append(C.rms_rel(_sph_state(sc)[1], traj[2 * nF - 1 - k], R))
        if k == nF - 1:
            st = np.array(_sph_state(sc)[1])
            marks["T_index"] = len(ser["t_s"]) - 1
            marks["r_median_at_T_over_R"] = float(np.median(np.linalg.norm(
                st - np.average(st, axis=0, weights=ms), axis=1)) / R)
            marks["internal_at_T"] = sum(p.internal_energy for p in ps)
            marks["bath_at_T"] = sc.energy_bath_J
            marks["Q_led_at_T"] = Qled
            marks["Qv_eng_at_T"] = Qv_eng
            marks["Qp_eng_at_T"] = Qp_eng
            for p in ps:
                p.velocity = Vec3(-p.velocity.x, -p.velocity.y, -p.velocity.z)
            if delta:
                u = C.unit_vectors(len(ps))
                for i, p in enumerate(ps):
                    p.position = Vec3(p.position.x + delta * R * u[i, 0], p.position.y + delta * R * u[i, 1],
                                      p.position.z + delta * R * u[i, 2])
            sc._vv_accel = None                     # invalidate the velocity-Verlet cache at the reversal
            sc._vv_pairs = None
            if ledger:
                Pv = visc_power(sc, ps, alpha)      # trapezoid restarts at the reversed state
    _, p2, v2 = _sph_state(sc)
    out = {"engine": f"sph_{eos}", "alpha": alpha, "N": S["N"], "T_tff": T_frac, "n_per_tff": n_per_tff, "dt_s": dt,
           "steps_each_leg": nF, "delta_R": delta, "label": label,
           "eps_rec_R": C.rms_rel(p2, p0, R),
           "vel_rec_rel": float(np.sqrt(((np.array(v2) + np.array(v0)) ** 2).sum(1).mean()) /
                                np.sqrt((np.array(v0) ** 2).sum(1).mean())),
           "wall_s": time.time() - t0}
    out.update(marks)
    if ledger:
        U0 = ser["U_soft"][0]
        iT = marks["T_index"]
        mech = np.array(ser["K"]) + np.array(ser["U_soft"])
        loss_lit = -(mech - mech[0])
        loss_cor = loss_lit - (np.array(ser["F_iso"]) - ser["F_iso"][0])
        Q = np.array(ser["Q_led"])

        def rel(a, b):
            return float(abs(a - b) / abs(a)) if abs(a) > 0 else None
        out.update({
            "U_soft0_J": U0, "K0_J": ser["K"][0],
            "Q_led_end_J": float(Q[-1]), "Q_visc_eng_end_J": Qv_eng, "Q_press_eng_end_J": Qp_eng,
            "Q_led_over_absU0": float(Q[-1] / abs(U0)),
            "engine_booking_split_max_relerr": book_err,
            "total_drift_max_over_absU0": float(np.max(np.abs(np.array(ser["total"]) - ser["total"][0])) / abs(U0)),
            "total_drift_end_over_absU0": float(abs(ser["total"][-1] - ser["total"][0]) / abs(U0)),
            "closure_literal_end": rel(Q[-1], loss_lit[-1]), "closure_corrected_end": rel(Q[-1], loss_cor[-1]),
            "closure_literal_T": rel(Q[iT], loss_lit[iT]), "closure_corrected_T": rel(Q[iT], loss_cor[iT]),
            "closure_engine_vs_led_end": rel(Q[-1], Qv_eng),
            "loss_literal_end_over_absU0": float(loss_lit[-1] / abs(U0)),
            "loss_corrected_end_over_absU0": float(loss_cor[-1] / abs(U0)),
            "dF_iso_end_over_absU0": float((ser["F_iso"][-1] - ser["F_iso"][0]) / abs(U0)),
            "Sdot_led_min": float(np.min(ser["Sdot_led"])), "Sdot_led_max": float(np.max(ser["Sdot_led"])),
            "Sdot_led_n_negative": int(np.sum(np.array(ser["Sdot_led"]) < 0)),
            "n_ledger_points": len(ser["Sdot_led"]),
            "q_visc_eng_step_min_J": (min_qv_step if alpha > 0 else 0.0),
            "q_visc_eng_n_negative_steps": int(n_neg_qv),
            "S_led_cum_J_per_K_end": float(Q[-1] / 300.0) if eos == "isothermal" else None,
        })
        if eos == "ideal":
            Sst = np.array(ser["S_state"])
            out.update({"dS_state_end_J_per_K": float(Sst[-1] - Sst[0]), "S_visc_eng_end_J_per_K": S_visc_eng,
                        "dS_state_T_J_per_K": float(Sst[iT] - Sst[0]),
                        "dS_state_minstep": float(np.min(np.diff(Sst))),
                        "dS_state_max_dev_J_per_K": float(np.max(np.abs(Sst - Sst[0]))),
                        "sentinel_literal_uint_vs_Qled_T": rel(marks["Q_led_at_T"], marks["internal_at_T"]),
                        "sentinel_uint_vs_engine_split_T": rel(marks["internal_at_T"],
                                                               marks["Qv_eng_at_T"] + marks["Qp_eng_at_T"]),
                        "M_mol_kg_per_mol_choice": sc.ideal_molar_mass_kg_mol, "gamma": sc.ideal_gamma})
        stride = max(1, len(ser["t_s"]) // 600)
        out["series"] = {k: v[::stride] for k, v in ser.items()}
        out["series_stride"] = stride
    if keep_curve:
        out["backward_divergence_R"] = curve
    print(json.dumps({k: v for k, v in out.items() if k not in ("series", "backward_divergence_R")}), flush=True)
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    S = C.setup(64)
    runs = []
    if mode == "main":
        for a in (0.0, 1.0):
            for T in (0.3, 1.0):
                for n in (600, 1200):
                    runs.append(echo(S, "isothermal", a, T, n, keep_curve=(T == 1.0 and n == 1200)))
    elif mode == "delta":
        for a in (0.0, 1.0):
            for d in (1e-13, 1e-11, 1e-9):
                runs.append(echo(S, "isothermal", a, 1.0, 600, delta=d, ledger=False))
        S32 = C.setup(32)
        for a in (0.0, 1.0):
            runs.append(echo(S32, "isothermal", a, 0.3, 600, label="smoke_N32"))
    elif mode == "ideal":
        for a in (0.0, 1.0):
            for n in (600, 1200):
                runs.append(echo(S, "ideal", a, 1.0, n))
    with open(os.path.join(C.HERE, f"res_sph_{mode}.json"), "w", encoding="utf-8") as f:
        json.dump({"leaves": S["leaves"], "choices": {k: S[k] for k in ("N", "M", "R", "cs", "h", "eps", "tff")},
                   "runs": runs}, f, indent=1)
    print("done")
