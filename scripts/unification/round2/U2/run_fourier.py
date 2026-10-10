"""U2 Fourier leg - entropy production of conduction on dynamics.fields.heat.diffuse_fvm (32^3, insulated).

Per slice: E = sum rho_cp T dx^3, S = sum rho_cp dx^3 ln T, and the discrete entropy-production rate
S_dot = sum_faces k_face (T_hi - T_lo)^2 / (T_hi T_lo) * dx  (the face-by-face form of integral k|grad T|^2/T^2 dV).
No reversed run exists: the engine is asked for one (negative time; negative conductivity) and its answer recorded;
the conditioning of the inverse of the forward discrete map is computed for the grid's highest mode."""
import json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import u2_common as C
from sigma_ground.dynamics.fields.heat import diffuse_fvm, stable_dt, thermal_energy
from sigma_ground.dynamics.continuum import material_properties


def face_sdot(T, k, dx):
    s = 0.0
    for ax in range(3):
        lo = tuple(slice(None, -1) if a == ax else slice(None) for a in range(3))
        hi = tuple(slice(1, None) if a == ax else slice(None) for a in range(3))
        kA, kB = k[lo], k[hi]
        kf = 2.0 * kA * kB / (kA + kB)
        s += float((kf * (T[hi] - T[lo]) ** 2 / (T[hi] * T[lo])).sum()) * dx
    return s


if __name__ == "__main__":
    t0 = time.time()
    n, dx = 32, 0.01                                  # Choice: 32^3 cells of 1 cm
    Tb = 200.0                                         # Choice: background temperature (K)
    props = material_properties("water_ice", Tb)       # repo cascade (Mentat engine)
    kap, rho, cp = props["thermal_conductivity_W_mK"], props["density_kg_m3"], props["specific_heat_J_kgK"]
    k = np.full((n, n, n), kap)
    rcp = np.full((n, n, n), rho * cp)
    x = (np.arange(n) + 0.5) * dx
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    def blob(c, w):
        return np.exp(-((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) / (2 * w * w))
    T = Tb + 50.0 * blob((0.12, 0.14, 0.16), 4 * dx) - 30.0 * blob((0.22, 0.2, 0.18), 3 * dx)   # Choice: hot + cold blob
    alpha = kap / (rho * cp)
    dt_cfl = stable_dt(alpha, dx)
    t_slice = 40 * dt_cfl
    rows = []
    E0 = thermal_energy(T, rcp, dx)
    S_of = lambda TT: float((rcp * np.log(TT)).sum()) * dx ** 3
    total_sub = 0
    for s in range(21):
        rows.append({"t_s": s * t_slice, "E_J": thermal_energy(T, rcp, dx), "S_J_per_K": S_of(T),
                     "Sdot_W_per_K": face_sdot(T, k, dx), "Tmax": float(T.max()), "Tmin": float(T.min())})
        if s == 20:
            break
        T, info = diffuse_fvm(T, k, rcp, dx, total_time=t_slice)
        total_sub += info["substeps"]
        dt_sub = info["dt"]
    Sd = np.array([r["Sdot_W_per_K"] for r in rows])
    Ss = np.array([r["S_J_per_K"] for r in rows])
    Es = np.array([r["E_J"] for r in rows])
    # integral check: Delta S between slices vs trapezoid of S_dot
    dS = np.diff(Ss)
    trap = 0.5 * (Sd[1:] + Sd[:-1]) * t_slice
    # the reversed run: ask the engine
    Tneg, info_neg = diffuse_fvm(T, k, rcp, dx, total_time=-20 * t_slice)
    Tk, info_k = diffuse_fvm(T, -k, rcp, dx, total_time=20 * t_slice)
    lam = 12.0 * alpha * dt_sub / dx ** 2              # highest-mode decay per substep factor (1 - lam)
    out = {
        "choices": {"grid": n, "dx_m": dx, "T_background_K": Tb, "blobs": "+50 K (w=4dx), -30 K (w=3dx)",
                    "slices": 20, "t_slice_s": t_slice},
        "material": {"source": "sigma_ground.dynamics.continuum.material_properties('water_ice', 200 K) (repo cascade)",
                     "k_W_mK": kap, "rho_kg_m3": rho, "cp_J_kgK": cp, "alpha_m2_s": alpha},
        "substeps_total": total_sub, "dt_sub_s": dt_sub,
        "Sdot_min": float(Sd.min()), "Sdot_all_positive": bool((Sd > 0).all()),
        "Sdot_monotone_decreasing": bool((np.diff(Sd) < 0).all()),
        "S_monotone_increasing": bool((dS > 0).all()), "dS_min_J_per_K": float(dS.min()),
        "dS_vs_trapezoid_max_relerr": float(np.max(np.abs(dS - trap) / np.abs(trap))),
        "energy_drift_rel": float(np.max(np.abs(Es - E0)) / E0),
        "reversed_run_negative_time": {"substeps": info_neg["substeps"], "unchanged": bool(np.array_equal(Tneg, T))},
        "reversed_run_negative_k": {"substeps": info_k["substeps"], "unchanged": bool(np.array_equal(Tk, T))},
        "inverse_map_highest_mode_log10_amplification": float(-total_sub * math.log10(1.0 - lam)),
        "lam_per_substep": lam,
        "rows": rows, "wall_s": time.time() - t0,
    }
    with open(os.path.join(C.HERE, "res_fourier.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k_: v for k_, v in out.items() if k_ != "rows"}, indent=1))
