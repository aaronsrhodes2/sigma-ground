"""U13 v2 main runner (repair round).  N = 64 runs.

Same physics, same evaluate() as u13_av_share.py (imported).  Adds, per the three skeptic reviews:
  * both readings of the design h-scan (S: spacing x {1,1.5,2};  H: h0 x {1,1.5,2}) at every h-scan state,
    and a fine h-scan (1.0-3.0 spacing) at the Run B step-50 state with V_press and V_visc split;
  * Run A alpha-independence of the state at step 50 (positions and velocities, alpha = 0.25 vs 2);
  * Run B alpha = 1 continued to 1.5 t_ff (post hoc; the pre-registered window stays 0-1 t_ff);
  * flow-structure rows (rms radius, v_rms, homologous infall speed, residual non-homologous speed);
  * energy-ledger drift of every run;
  * seed robustness (seeds 7, 1, 2, 3) at alpha = 1, dt = t_ff/600;
  * the design's literal "step 50" with dt = t_ff/100 (100 steps span 1 t_ff) and its dt/2 control.
Pre-registration: u13_prereg_v2.json (written before this run) + the original PREREG (unchanged).
Run from the repo root:  python -X utf8 <this file>
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from u13_common_v2 import (U, OUT, HSET, HFINE, READ_S, READ_H, jdump, scene_N, hscan, both_readings,  # noqa: E402
                           slope_of, row_at, radial_structure, state_arrays, continuum_step, cfl_timestep)

ALPHAS = U.ALPHAS
T_FF = U.T_FF


def run(alpha, dt_mode, n_steps, every, extra_steps=(), hscan_steps=None, seed=U.SEED, keep_state=()):
    """hscan_steps: dict step -> list of h/spacing multipliers."""
    hscan_steps = hscan_steps or {}
    sc, d = scene_N(U.N, seed)
    led0 = sc.energy_ledger()
    rows, hs, states = [], {}, {}
    t0 = time.time()
    for st in range(n_steps + 1):
        on_grid = (st % every == 0) or (st in extra_steps)
        if on_grid or st in hscan_steps or st in keep_state:
            ps = sc.continuum_parcels()
            if on_grid:
                rows.append(dict(step=st, t_s=sc.time, t_over_tff=sc.time / T_FF,
                                 **U.evaluate(ps, U.H0, alpha), structure=radial_structure(ps)))
            if st in hscan_steps:
                hs[str(st)] = hscan(ps, d, hscan_steps[st], alpha)
            if st in keep_state:
                m, x, v = state_arrays(ps)
                states[str(st)] = dict(x=x.tolist(), v=v.tolist())
        if st == n_steps:
            break
        dt = cfl_timestep(sc) if dt_mode == "cfl" else dt_mode
        continuum_step(sc, dt, alpha_visc=alpha)
    led = sc.energy_ledger()
    return dict(alpha=alpha, seed=seed, dt_mode=("cfl_timestep" if dt_mode == "cfl" else dt_mode), rows=rows,
                hscan=hs, energy_ledger_start=led0, energy_ledger_end=led,
                energy_drift_over_W0=(led["total"] - led0["total"]) / abs(led0["self_gravity"]),
                wall_s=time.time() - t0, _states=states)


def ratio_at(r, step):
    return [row for row in r["rows"] if row["step"] == step][0]["ratio"]


def verdict(x):
    return "FORBID" if x > 0.1 else ("ALLOW" if x < 0.01 else "GRAY")


if __name__ == "__main__":
    res = {"id": "U13", "version": "v2", "mentat_sources": U.SRC, "choices": U.CHOICES,
           "prereg_original_unchanged": U.PREREG,
           "prereg_v2": json.load(open(os.path.join(OUT, "u13_prereg_v2.json"), encoding="utf-8")),
           "t_ff_s": T_FF}
    sc0, d0 = scene_N()
    res["dt_cfl_initial_s"] = cfl_timestep(sc0)
    res["material_sound_speed_iron_m_s"] = sc0.continuum_parcels()[0].sound_speed
    res["spacing_m"] = d0
    res["h0_m"] = U.H0

    # ---------------- Run A: design as written
    runA = [run(a, "cfl", 100, 50, hscan_steps=({50: HSET, 100: HSET} if a == 1.0 else {}),
                keep_state=(0, 50)) for a in ALPHAS]
    print("Run A done", flush=True)
    xa, va = np.array(runA[0]["_states"]["50"]["x"]), np.array(runA[0]["_states"]["50"]["v"])
    xb, vb = np.array(runA[3]["_states"]["50"]["x"]), np.array(runA[3]["_states"]["50"]["v"])
    x0 = np.array(runA[0]["_states"]["0"]["x"])
    disp = np.sqrt(((xa - x0) ** 2).sum(1))
    alpha_dep = dict(t_s=[row for row in runA[0]["rows"] if row["step"] == 50][0]["t_s"],
                     max_pos_diff_m=float(np.sqrt(((xa - xb) ** 2).sum(1)).max()),
                     max_displacement_m=float(disp.max()),
                     pos_diff_over_displacement=float(np.sqrt(((xa - xb) ** 2).sum(1)).max() / disp.max()),
                     rel_vel_diff=float(np.sqrt(((va - vb) ** 2).sum(1)).max() / np.sqrt((va ** 2).sum(1)).max()))

    # ---------------- Run B: corrected protocol (alpha = 1 continued to 1.5 t_ff, post hoc)
    dtB = T_FF / 600.0
    runB = []
    for a in ALPHAS:
        n = 900 if a == 1.0 else 600
        hsd = {50: HFINE, 300: HSET, 600: HSET} if a == 1.0 else {}
        runB.append(run(a, dtB, n, 30, extra_steps=(50,), hscan_steps=hsd))
        print("Run B alpha", a, "done", flush=True)
    ctrl = run(1.0, T_FF / 1200.0, 1200, 60, extra_steps=(100,))
    print("ctrl done", flush=True)

    # ---------------- seeds (alpha = 1, dt = t_ff/600)
    seeds = {}
    for sd in (1, 2, 3):
        r = run(1.0, dtB, 600, 300, extra_steps=(50,), seed=sd)
        seeds[str(sd)] = r
        print("seed", sd, [round(x["ratio"], 4) for x in r["rows"] if x["ratio"] is not None], flush=True)

    # ---------------- design's literal step 50 with dt = t_ff/100 (+ dt/2 control)
    k2a = run(1.0, T_FF / 100.0, 100, 50)
    k2b = run(1.0, T_FF / 200.0, 200, 100)
    print("K2 done", flush=True)

    for r in runA + runB + [ctrl, k2a, k2b] + list(seeds.values()):
        r.pop("_states", None)
    res.update(runA=runA, runB=runB, runB_dt_control_alpha1=ctrl, seeds_alpha1=seeds,
               literal_step50_dt_tff_over_100=k2a, literal_step50_dt_tff_over_200_control=k2b,
               runA_alpha_independence_step50=alpha_dep)

    # ---------------- analysis
    A = {}
    for st in (0, 50, 100):
        rs = [ratio_at(r, st) for r in runA]
        A[f"step{st}"] = dict(ratios=dict(zip(map(str, ALPHAS), rs)),
                              alpha_slope=(U.slope(ALPHAS, rs) if all(x and x > 0 for x in rs) else None))
    for st in ("50", "100"):
        A[f"h_slopes_step{st}"] = both_readings(runA[2]["hscan"][st])
    rA = ratio_at(runA[2], 50)
    A["ratio_alpha1_step50"] = rA
    A["t_step50_over_tff"] = [row for row in runA[2]["rows"] if row["step"] == 50][0]["t_over_tff"]
    A["t_step100_over_tff"] = [row for row in runA[2]["rows"] if row["step"] == 100][0]["t_over_tff"]
    A["scope_rule_design_step50"] = verdict(rA)
    A["alpha_slope_pass"] = abs(A["step50"]["alpha_slope"] - 1.0) <= 0.15
    A["h_slope_pass_reading_S"] = abs(A["h_slopes_step50"]["reading_S_spacing_x_1_1.5_2"] - 1.0) <= 0.3
    A["h_slope_pass_reading_H"] = abs(A["h_slopes_step50"]["reading_H_h0_x_1_1.5_2"] - 1.0) <= 0.3
    A["prediction_0p1alpha_measured_over_predicted"] = rA / 0.1

    rB1 = runB[2]["rows"]
    win = [row for row in rB1 if row["t_over_tff"] <= 1.0 + 1e-9]
    grid = [row for row in win if row["step"] % 30 == 0]
    B = {"steps": [row["step"] for row in grid], "t_over_tff": [row["t_over_tff"] for row in grid]}
    B["ratio_by_alpha"] = {str(r["alpha"]): [row["ratio"] for row in r["rows"] if row["step"] % 30 == 0
                                             and row["t_over_tff"] <= 1.0 + 1e-9] for r in runB}
    B["alpha_slope_vs_time"] = [U.slope(ALPHAS, [B["ratio_by_alpha"][str(a)][k] for a in ALPHAS])
                                if all(B["ratio_by_alpha"][str(a)][k] for a in ALPHAS) else None
                                for k in range(len(grid))]
    B["alpha_slope_first_below_0p85_t_over_tff"] = next(
        (t for t, x in zip(B["t_over_tff"], B["alpha_slope_vs_time"]) if x is not None and x < 0.85), None)
    rB50 = ratio_at(runB[2], 50)
    B["ratio_alpha1_step50"] = rB50
    B["t_step50_over_tff"] = 50 / 600
    B["scope_rule_design_step50"] = verdict(rB50)
    rmax = max(row["ratio"] for row in win if row["ratio"] is not None)
    B["max_ratio_alpha1_0_to_1tff"] = rmax
    B["scope_rule_whole_collapse_preregistered_ii"] = verdict(rmax)
    B["ratio_alpha1_at_1tff"] = ratio_at(runB[2], 600)
    B["ratio_still_rising_at_1tff"] = ratio_at(runB[2], 600) > ratio_at(runB[2], 570)

    def first_cross(rows, thr):
        for row in rows:
            if row["ratio"] is not None and row["ratio"] > thr:
                return row["t_over_tff"]
        return None
    B["t_first_sample_above_0p01"] = first_cross(grid, 0.01)
    B["t_first_sample_above_0p1"] = first_cross(grid, 0.1)
    B["ratio_alpha1_by_t_over_tff"] = {f"{row['t_over_tff']:.4f}": row["ratio"] for row in rB1}
    B["extension_post_hoc"] = dict(ratio_1p5tff=ratio_at(runB[2], 900),
                                   max_over_0_to_1p5tff=max(row["ratio"] for row in rB1 if row["ratio"] is not None),
                                   ratio_1p2tff=ratio_at(runB[2], 720), ratio_1p35tff=ratio_at(runB[2], 810))
    for st in ("50", "300", "600"):
        B[f"h_slopes_step{st}"] = both_readings(runB[2]["hscan"][st])
    fine = runB[2]["hscan"]["50"]
    B["fine_hscan_step50"] = dict(
        rows=[dict(h_over_spacing=r["h_over_spacing"], ratio=r["ratio"], V_press_J=r["V_press_J"],
                   V_visc_J=r["V_visc_J"], mean_neighbours=r["mean_neighbours"]) for r in fine],
        slopes={name: {k: slope_of(fine, sel, k) for k in ("ratio", "V_visc_J", "V_press_J")}
                for name, sel in (("reading_S", READ_S), ("reading_H", READ_H), ("all_1_to_3", HFINE),
                                  ("1.5_to_3", [1.5, 1.75, 2.0, 2.25, 2.5, 3.0]))})
    ctrl_map = {round(row["t_over_tff"], 6): row["ratio"] for row in ctrl["rows"]}
    diffs = [abs(row["ratio"] / ctrl_map[round(row["t_over_tff"], 6)] - 1.0) for row in win
             if row["ratio"] and ctrl_map.get(round(row["t_over_tff"], 6))]
    B["dt_control_max_rel_diff"] = max(diffs)
    B["energy_drift_over_W0"] = {str(r["alpha"]): r["energy_drift_over_W0"] for r in runB}
    B["structure_alpha1"] = {f"{row['t_over_tff']:.4f}": row["structure"] for row in rB1 if row["step"] % 150 == 0
                             or row["step"] == 50}
    B["rms_radius_by_alpha_at_1tff"] = {str(r["alpha"]): [row for row in r["rows"] if row["step"] == 600][0]
                                        ["structure"]["rms_radius_over_R"] for r in runB}

    S = {"7": {"50": rB50, "300": ratio_at(runB[2], 300), "600": ratio_at(runB[2], 600),
               "energy_drift_over_W0_to_1tff": None}}
    for sd, r in seeds.items():
        S[sd] = {"50": ratio_at(r, 50), "300": ratio_at(r, 300), "600": ratio_at(r, 600),
                 "energy_drift_over_W0_to_1tff": r["energy_drift_over_W0"],
                 "rms_radius_1tff": [row for row in r["rows"] if row["step"] == 600][0]["structure"]["rms_radius_over_R"]}
    S["range_step50"] = [min(S[s]["50"] for s in ("7", "1", "2", "3")), max(S[s]["50"] for s in ("7", "1", "2", "3"))]
    S["range_step600"] = [min(S[s]["600"] for s in ("7", "1", "2", "3")), max(S[s]["600"] for s in ("7", "1", "2", "3"))]
    S["verdicts_step50"] = {s: verdict(S[s]["50"]) for s in ("7", "1", "2", "3")}
    S["verdicts_1tff"] = {s: verdict(S[s]["600"]) for s in ("7", "1", "2", "3")}

    K2 = dict(dt_tff_over_100=dict(step50_ratio=ratio_at(k2a, 50), step50_t_over_tff=0.5,
                                   step100_ratio=ratio_at(k2a, 100), scope_rule_design_step50=verdict(ratio_at(k2a, 50))),
              dt_tff_over_200=dict(step100_ratio=ratio_at(k2b, 100), step200_ratio=ratio_at(k2b, 200)))
    K2["rel_diff_0p5tff"] = ratio_at(k2a, 50) / ratio_at(k2b, 100) - 1.0
    K2["literal_step50_rule_by_protocol"] = {"cfl_timestep (design)": [rA, verdict(rA)],
                                             "dt = t_ff/600 (corrected)": [rB50, verdict(rB50)],
                                             "dt = t_ff/100": [ratio_at(k2a, 50), verdict(ratio_at(k2a, 50))]}
    res["analysis"] = {"runA": A, "runB": B, "seeds": S, "literal_step50": K2,
                       "runA_alpha_independence_step50": alpha_dep}
    jdump(res, "u13_results_v2.json")
    print(json.dumps(res["analysis"], indent=1, default=str)[:12000])
