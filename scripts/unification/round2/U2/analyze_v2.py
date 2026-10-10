"""U2 analysis, repair round (v2). Evaluates the SAME frozen criteria (prereg.json, unchanged) on the SAME original
runs, applies the skeptics' corrections to the derived quantities and the reading, adds the post-hoc diagnostics of
prereg_v2_addendum.json, and writes u2_results_v2.json and u2_loschmidt_echo_v2.png.

Changes vs analyze.py (each traceable to a skeptic):
  * finite-time growth rate = ln(gain) / T_leg with T_leg = 1 t_ff (delta is injected AT the reversal, so it lives
    through one backward leg only); 'round-trip gain' renamed 'backward-leg gain'.            [reproduce, physics, skeptic]
  * the frozen corrected S8 sentinel (prereg C3, parts a and b) is evaluated and reported (FAIL).        [skeptic]
  * the original criterion is evaluated under both readings of 'shrink' (strict < 1, and 'halves').      [skeptic]
  * Fourier inverse-map amplification from the MEASURED per-substep factor of the highest insulated-grid mode. [physics]
  * new post-hoc diagnostics: delta-at-t0 echo, forward twin to 4 t_ff, backward-leg profile, roundoff horizon
    T = 1..8 t_ff, cache-invalidation check for both EOS, cloud-size tracks.
Mentat (simplify_expr) supplies the logarithms and closed forms; sources recorded."""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sympy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
from mentat_client import mentat


def J(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


MENTAT = {}


def mx(key, expr):
    v, s = mentat("simplify_expr", {"expression": expr})
    val = float(sympy.sympify(v).evalf(17))
    MENTAT[key] = {"value": str(v), "float": val, "source": s, "tool": "simplify_expr", "args": {"expression": expr}}
    return val


pre = J("prereg.json"); add2 = J("prereg_v2_addendum.json")
nb = J("res_nbody.json"); main = J("res_sph_main.json"); dl = J("res_sph_delta.json")
ideal = J("res_sph_ideal.json"); fr = J("res_fourier.json"); leaves = J("mentat_leaves.json")
hlo = J("res_nbody_v2_horizon_lo.json"); hhi = J("res_nbody_v2_horizon_hi.json"); ctl = J("res_nbody_v2_controls.json")
side = J("res_sph_v2_side.json"); fr2 = J("res_fourier_v2.json")
fr2["reversed_requests"]["negative_k"]["stable_dt_for_negative_alpha"] = str(
    fr2["reversed_requests"]["negative_k"]["stable_dt_for_negative_alpha"])   # 'inf' as a string: strict JSON


def pick(runs, **kw):
    out = [r for r in runs if all((r.get(k) == v) for k, v in kw.items())]
    assert len(out) == 1, (kw, len(out))
    return out[0]


NB = nb["runs"]; SM = main["runs"]; SD = dl["runs"]; SI = ideal["runs"]
eps = {}
for T in (0.3, 1.0):
    for n in (600, 1200):
        eps[("nbody", T, n)] = pick(NB, T_tff=T, n_per_tff=n, delta_R=0.0, N=64)["eps_rec_R"]
        for a in (0.0, 1.0):
            eps[(f"sph{int(a)}", T, n)] = pick(SM, alpha=a, T_tff=T, n_per_tff=n)["eps_rec_R"]
for n in (600, 1200):
    for a in (0.0, 1.0):
        eps[(f"ideal{int(a)}", 1.0, n)] = pick(SI, alpha=a, n_per_tff=n)["eps_rec_R"]
ratio = {k[0] + f"_T{k[1]}": eps[(k[0], k[1], 1200)] / eps[(k[0], k[1], 600)] for k in eps if k[2] == 600}

# ---------------- delta control (T = 1, dt = t_ff/600), delta injected AT THE REVERSAL
deltas = (1e-13, 1e-11, 1e-9)
dctl = {"nbody": [pick(NB, T_tff=1.0, n_per_tff=600, delta_R=d)["eps_rec_R"] for d in deltas],
        "sph0": [pick(SD, alpha=0.0, delta_R=d)["eps_rec_R"] for d in deltas],
        "sph1": [pick(SD, alpha=1.0, delta_R=d)["eps_rec_R"] for d in deltas]}
d0 = {"nbody": eps[("nbody", 1.0, 600)], "sph0": eps[("sph0", 1.0, 600)], "sph1": eps[("sph1", 1.0, 600)]}


def lin_slope(e, e0):
    pts = [(d, x) for d, x in zip(deltas, e) if 3 * e0 <= x <= 1e-2]
    if len(pts) < 2:
        return None
    return float(np.polyfit(np.log10([p[0] for p in pts]), np.log10([p[1] for p in pts]), 1)[0])


slopes = {k: lin_slope(v, d0[k]) for k, v in dctl.items()}
gain = {k: [x / d for x, d in zip(v, deltas)] for k, v in dctl.items()}
sph1_delta_indep = abs(dctl["sph1"][-1] / d0["sph1"] - 1.0)
g_med = {k: float(np.median(gain[k])) for k in ("nbody", "sph0")}
rate = {"nbody": mx("rate_nbody_per_tff", f"log({g_med['nbody']!r})/1"),
        "sph0": mx("rate_sph0_per_tff", f"log({g_med['sph0']!r})/1")}

# ---------------- ledgers (unchanged evaluation)
r1 = {n: pick(SM, alpha=1.0, T_tff=1.0, n_per_tff=n) for n in (600, 1200)}
r1s = {n: pick(SM, alpha=1.0, T_tff=0.3, n_per_tff=n) for n in (600, 1200)}
r0 = {n: pick(SM, alpha=0.0, T_tff=1.0, n_per_tff=n) for n in (600, 1200)}
i1 = {n: pick(SI, alpha=1.0, n_per_tff=n) for n in (600, 1200)}
i0 = {n: pick(SI, alpha=0.0, n_per_tff=n) for n in (600, 1200)}
sdot_min = min(r["Sdot_led_min"] for r in SM + SI if r["alpha"] == 1.0)
sdot_neg = sum(r["Sdot_led_n_negative"] for r in SM + SI)
qv_min = min(r["q_visc_eng_step_min_J"] for r in SM + SI if r["alpha"] == 1.0)
qv_neg = sum(r["q_visc_eng_n_negative_steps"] for r in SM + SI)


def eng_closure(r):
    U0 = abs(r["U_soft0_J"]); loss = r["loss_corrected_end_over_absU0"] * U0
    return abs(r["Q_visc_eng_end_J"] - loss) / r["Q_visc_eng_end_J"]


post_hoc = {
    "richardson_corrected_closure_T1": 2 * r1[1200]["closure_corrected_end"] - r1[600]["closure_corrected_end"],
    "richardson_corrected_closure_T0.3": 2 * r1s[1200]["closure_corrected_end"] - r1s[600]["closure_corrected_end"],
    "engine_own_viscous_booking_vs_mech_loss_T1": {n: eng_closure(r1[n]) for n in (600, 1200)},
    "engine_own_viscous_booking_vs_mech_loss_T0.3": {n: eng_closure(r1s[n]) for n in (600, 1200)},
    "Q_led_vs_Q_visc_engine_T1": {n: r1[n]["closure_engine_vs_led_end"] for n in (600, 1200)},
}

# ---------------- original criterion (verbatim clauses), BOTH readings of 'shrink'
T = 1.0
o = {}
o["hamiltonian_le_1e-8"] = all(eps[(e, T, n)] <= 1e-8 for e in ("nbody", "sph0") for n in (600, 1200))
o["viscous_ge_1e-3"] = all(eps[("sph1", T, n)] >= 1e-3 for n in (600, 1200))
o["min_Sdot_ge_0"] = sdot_min >= 0 and sdot_neg == 0
o["ledger_closure_le_1e-3_literal"] = r1[1200]["closure_literal_end"] <= 1e-3
o["SAME_all_within_10x"] = (max(eps[(e, T, 1200)] for e in ("nbody", "sph0", "sph1")) /
                            min(eps[(e, T, 1200)] for e in ("nbody", "sph0", "sph1"))) <= 10
readings = {}
for rd, shrink in (("strict_ratio_lt_1 (addendum_1, fixed AFTER the Hamiltonian ratios were seen)", lambda q: q < 1.0),
                   ("halves_ratio_0.4_to_0.6 (the design's 'halves with dt' language)", lambda q: 0.4 <= q <= 0.6)):
    cl = dict(o)
    cl["hamiltonian_shrink_under_dt_halving"] = all(shrink(eps[(e, T, 1200)] / eps[(e, T, 600)]) for e in ("nbody", "sph0"))
    cl["viscous_does_not_shrink"] = not shrink(eps[("sph1", T, 1200)] / eps[("sph1", T, 600)])
    sep = all(cl[k] for k in ("hamiltonian_le_1e-8", "hamiltonian_shrink_under_dt_halving", "viscous_ge_1e-3",
                              "viscous_does_not_shrink", "min_Sdot_ge_0", "ledger_closure_le_1e-3_literal"))
    readings[rd] = {"clauses": cl, "verdict": "SEPARATE" if sep else ("SAME" if cl["SAME_all_within_10x"] else "MIXED"),
                    "failed": [k for k in ("hamiltonian_le_1e-8", "hamiltonian_shrink_under_dt_halving",
                                           "viscous_ge_1e-3", "viscous_does_not_shrink", "min_Sdot_ge_0",
                                           "ledger_closure_le_1e-3_literal") if not cl[k]]}
verdict_original = sorted({v["verdict"] for v in readings.values()})

# S8 sentinel, literal (design) form
drift = {f"{lab}_n{n}": {"max_over_run": r["total_drift_max_over_absU0"], "at_end": r["total_drift_end_over_absU0"]}
         for lab, d in (("isothermal_a1_T1", r1), ("isothermal_a0_T1", r0), ("ideal_a1_T1", i1), ("ideal_a0_T1", i0))
         for n, r in d.items()}
s8_literal = {
    "isothermal_sum_m_uint_eq_Qcum": {"internal_at_T": r1[1200]["internal_at_T"], "Q_led_at_T": r1[1200]["Q_led_at_T"],
                                      "pass": abs(r1[1200]["internal_at_T"] - r1[1200]["Q_led_at_T"]) <= 1e-3 * r1[1200]["Q_led_at_T"]},
    "ideal_sum_m_uint_eq_Qcum_relerr": i1[1200]["sentinel_literal_uint_vs_Qled_T"],
    "ideal_pass": i1[1200]["sentinel_literal_uint_vs_Qled_T"] <= 1e-3,
    "K_Wsoft_Uint_conserved_1e-4": drift,
    "verdict": "FAIL"}
# S8 sentinel, FROZEN corrected form (prereg C3): (a) drift <= 1e-4 ; (b) engine viscous share == Q_led to 1e-3 at dt/1200
s8a_max = {k: v["max_over_run"] for k, v in drift.items() if k.endswith("n1200")}
s8a_end = {k: v["at_end"] for k, v in drift.items() if k.endswith("n1200")}
s8b = {"isothermal_a1_T1_n1200": r1[1200]["closure_engine_vs_led_end"],
       "ideal_a1_T1_n1200": i1[1200]["closure_engine_vs_led_end"]}
s8_frozen = {
    "text": pre["design_corrections_identified_before_running"][2],
    "part_a_drift_max_over_run_dt1200": s8a_max, "part_a_pass_max_reading": all(v <= 1e-4 for v in s8a_max.values()),
    "part_a_drift_end_dt1200": s8a_end, "part_a_pass_end_reading": all(v <= 1e-4 for v in s8a_end.values()),
    "part_a_drift_max_over_run_dt600": {k: v["max_over_run"] for k, v in drift.items() if k.endswith("n600")},
    "part_b_engine_viscous_share_vs_Q_led_dt1200": s8b, "part_b_pass": all(v <= 1e-3 for v in s8b.values()),
}
s8_frozen["verdict"] = "PASS" if (s8_frozen["part_a_pass_max_reading"] and s8_frozen["part_b_pass"]) else "FAIL"
s8_frozen["not_a_sentinel_by_construction_check"] = {
    "what": "the script's pair decomposition of continuum_step's own pair lists sums back to the engine's own "
            "internal-energy increment; it reproduces the engine's booking and could not fail",
    "isothermal_split_max_relerr": r1[1200]["engine_booking_split_max_relerr"],
    "ideal_uint_vs_engine_split_T": i1[1200]["sentinel_uint_vs_engine_split_T"]}

# ---------------- corrected criterion (frozen; unchanged)
c = {}
c["H1"] = o["hamiltonian_le_1e-8"]
c["H2"] = all(0.1 <= eps[(e, T, 1200)] / eps[(e, T, 600)] <= 10 for e in ("nbody", "sph0"))
c["D1"] = (o["viscous_ge_1e-3"] and abs(eps[("sph1", T, 1200)] / eps[("sph1", T, 600)] - 1) <= 0.1
           and sph1_delta_indep <= 0.1)
c["D2"] = o["min_Sdot_ge_0"] and qv_min >= 0 and qv_neg == 0
c["D3"] = r1[1200]["closure_corrected_end"] <= 1e-3
sep_c = all(c[k] for k in ("H1", "H2", "D1", "D2", "D3"))
verdict_corrected = "SEPARATE" if sep_c else ("SAME" if o["SAME_all_within_10x"] else "MIXED")
failed_c = [k for k in ("H1", "H2", "D1", "D2", "D3") if not c[k]]
could_fail = {
    "D1_delta_clause_gain_needed_to_fail": 0.1 * d0["sph1"] / deltas[-1],
    "D1_delta_clause_measured_viscous_response_gain": abs(dctl["sph1"][-1] - d0["sph1"]) / deltas[-1],
    "H2_note": "H2 ([0.1, 10]) also passes a first-order dt effect (ideal a=0 ratio 0.5005): it is not a discriminator",
    "H2_ideal_a0_ratio": ratio["ideal0_T1.0"]}

# ---------------- growth-rate diagnostics (POST-HOC)
A = ctl["A_delta_at_t0_echo"]
tw = np.array(ctl["B_forward_twin"]["t_tff_gain_rmedian"])
def at(arr, t):
    j = int(np.argmin(np.abs(arr[:, 0] - t))); return float(arr[j, 1])
tw_gain = {str(t): at(tw, t) for t in (0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)}
post = (tw[:, 0] >= 1.5) & (tw[:, 0] <= 4.0)
post_rate = float(np.polyfit(tw[post, 0], np.log(tw[post, 1]), 1)[0])
pre_rate = float(np.polyfit(tw[(tw[:, 0] >= 0.5) & (tw[:, 0] <= 1.0), 0],
                            np.log(tw[(tw[:, 0] >= 0.5) & (tw[:, 0] <= 1.0), 1]), 1)[0])
rmin_j = int(np.argmin(tw[:, 2]))
Cb = np.array(ctl["C_reversal_T1"]["t_back_tff_gain"])
frac_first_quarter = math.log(at(Cb, 0.25)) / math.log(Cb[-1, 1])
D = ctl["D_reversal_T0.3"]
growth = {
    "backward_leg_gain_delta_at_reversal_T1_dt600": g_med,
    "finite_time_rate_per_tff_ln_gain_over_1tff": rate,
    "as_reported_before_wrong_factor_2": {k: math.log(v) / 2.0 for k, v in g_med.items()},
    "A_delta_at_t0_full_echo_gain": A["gain"],
    "B_forward_twin_gain": tw_gain,
    "B_forward_twin_rate_0.5_to_1.0_tff": pre_rate,
    "B_forward_twin_rate_1.5_to_4.0_tff": post_rate,
    "B_r_median_min_R_and_time_tff": [float(tw[rmin_j, 2]), float(tw[rmin_j, 0])],
    "C_backward_leg_gain_end": float(Cb[-1, 1]),
    "C_fraction_of_ln_gain_in_first_0.25_tff_of_backward_leg": frac_first_quarter,
    "C_gain_after_0.25_tff_back": at(Cb, 0.25), "C_gain_after_0.5_tff_back": at(Cb, 0.5),
    "D_backward_leg_gain_after_0.3_tff_leg": D["gain_end"],
    "C6_revised": {"roundoff_echo_ratio_T1_over_T0.3_dt1200": eps[("nbody", 1.0, 1200)] / eps[("nbody", 0.3, 1200)],
                   "roundoff_echo_ratio_T1_over_T0.3_dt600": eps[("nbody", 1.0, 600)] / eps[("nbody", 0.3, 600)],
                   "backward_leg_gain_ratio_T1_over_T0.3": float(Cb[-1, 1]) / D["gain_end"]},
}

# ---------------- roundoff horizon (POST-HOC)
hz = {r["T_tff"]: r["eps_rec_R"] for r in hlo["echo"] + hhi["echo"]}
hz[0.3] = eps[("nbody", 0.3, 600)]
Ts = sorted(hz)
validation_bitwise = hz[1.0] == eps[("nbody", 1.0, 600)]
cross_1e8 = next((t for t in Ts if hz[t] > 1e-8), None)
cross_1e3 = next((t for t in Ts if hz[t] >= 1e-3), None)
T_star = mx("horizon_tff_ln2pow53_over_rate", f"log(2**53)/log({g_med['nbody']!r})")
horizon = {"eps_rec_R_by_T_tff": {str(t): hz[t] for t in Ts}, "validation_T1_bitwise_equal_original": validation_bitwise,
           "first_T_above_1e-8_R": cross_1e8, "first_T_at_or_above_1e-3_R": cross_1e3,
           "order_of_magnitude_T_star_tff": T_star,
           "r_median_forward_to_8tff": hhi["r_median_forward"]}

# ---------------- SPH side checks
SR = side["runs"]
def sp(**kw):
    return pick(SR, **kw)
cache = {}
for eos in ("isothermal", "ideal"):
    a_ = sp(eos=eos, alpha=0.0, invalidate_cache_at_reversal=True)["eps_rec_R"]
    b_ = sp(eos=eos, alpha=0.0, invalidate_cache_at_reversal=False)["eps_rec_R"]
    cache[eos] = {"invalidated": a_, "not_invalidated": b_, "bitwise_equal": a_ == b_, "rel_change": abs(b_ - a_) / a_}
cache["isothermal"]["matches_original_run"] = cache["isothermal"]["invalidated"] == eps[("sph0", 1.0, 600)]
cache["ideal"]["matches_original_run"] = cache["ideal"]["invalidated"] == eps[("ideal0", 1.0, 600)]
size = {}
for eos in ("isothermal", "ideal"):
    for a in (0.0, 1.0):
        r = SR[[j for j, q in enumerate(SR) if q["eos"] == eos and q["alpha"] == a][0]]
        s0, sT = r["size_track"][0], r["size_track"][-1]
        size[f"{eos}_a{int(a)}"] = {"t0": s0, "T1": sT,
                                     "r_median_max_over_leg_R": max(q["r_median_R"] for q in r["size_track"]),
                                     "r_median_min_over_leg_R": min(q["r_median_R"] for q in r["size_track"])}
size["nbody_r_median_T1_R"] = pick(NB, T_tff=1.0, n_per_tff=600, delta_R=0.0, N=64)["r_median_at_T_over_R"]
ideal_entropy = {f"n{n}": {"dS_state_forward_leg_J_per_K": i0[n]["dS_state_T_J_per_K"],
                           "dS_state_end_of_echo_J_per_K": i0[n]["dS_state_end_J_per_K"]} for n in (600, 1200)}
ideal_entropy["alpha1_n1200"] = {"dS_state_end": i1[1200]["dS_state_end_J_per_K"],
                                 "booked_sum_q_visc_over_T": i1[1200]["S_visc_eng_end_J_per_K"]}

# ---------------- Fourier
four = {"original": {k: v for k, v in fr.items() if k != "rows"}, "v2_conditioning": fr2}

# ---------------- predictions scorecard
wrong_predictions = [
    {"prediction": "dt_halving_hamiltonian_ratio 1..2", "measured": f"{ratio['nbody_T1.0']:.3f} (N-body), {ratio['sph0_T1.0']:.3f} (SPH a=0) at T=1"},
    {"prediction": "sph_a1_eps_rec_T1 1e-2 .. 1e-1 R", "measured": f"{eps[('sph1', 1.0, 600)]:.4f} / {eps[('sph1', 1.0, 1200)]:.4f} R isothermal; {eps[('ideal1', 1.0, 600)]:.4f} / {eps[('ideal1', 1.0, 1200)]:.4f} R ideal"},
    {"prediction": "verdict_corrected SEPARATE", "measured": verdict_corrected + " (D3 fails)"},
    {"prediction": "ideal_sentinel_energy <= 1e-4", "measured": f"max over run {i1[1200]['total_drift_max_over_absU0']:.3e} at dt/1200 (ideal a=1); {i1[600]['total_drift_max_over_absU0']:.3e} at dt/600"},
    {"prediction": "frozen C1: the original criterion 'cannot return SEPARATE whatever the physics'", "measured":
     "withdrawn: under the strict reading the Hamiltonian 'shrink' clause PASSES (ratios 0.80, 0.93); the original verdict is MIXED for other clauses"},
]

add2_score = {
    "P_A_delta_at_t0_gain_1_within_1e-3": {"measured": A["gain"], "pass": abs(A["gain"] - 1) <= 1e-3},
    "P_B_post_bounce_rate_1_to_6_per_tff": {"measured_1.5_to_4.0": post_rate, "pass_numeric_range": 1 <= post_rate <= 6,
                                           "chaos_reading_supported_rate_ge_0.5": post_rate >= 0.5},
    "P_C_more_than_half_of_ln_gain_in_first_0.25_tff_back": {"measured_fraction": frac_first_quarter,
                                                             "pass": frac_first_quarter > 0.5},
    "P_D_gain_after_0.3_leg_about_2.3": {"measured": D["gain_end"]},
    "P_E_cross_1e-8_between_3_and_6_tff": {"first_T_above": cross_1e8, "pass": cross_1e8 is not None and 3 < cross_1e8 <= 6},
    "P_E_eps_T8_ge_1e-3": {"measured": hz.get(8.0), "pass": hz.get(8.0, 0) >= 1e-3},
    "P_E_validation_bitwise": {"pass": validation_bitwise},
    "P_F_cache": {"isothermal_bitwise": cache["isothermal"]["bitwise_equal"], "ideal_rel_change": cache["ideal"]["rel_change"]},
    "P_G_fourier_factor": {"measured": fr2["per_substep_factor_measured"]["neumann_highest_k31"]["factor_mean"],
                           "rel_err_closed_form": fr2["factor_rel_err_vs_closed_form"],
                           "log10_amp_800": fr2["log10_inverse_amplification_800_from_measured_factor"],
                           "pass": fr2["factor_rel_err_vs_closed_form"] < 1e-9},
}
results = {
    "id": "U2", "version": "v2 (repair round)", "prereg": pre, "prereg_v2_addendum": add2,
    "mentat_leaves": leaves, "mentat_v2_simplify_expr": MENTAT, "choices": main["choices"],
    "eps_rec_R": {f"{k[0]}_T{k[1]}_n{k[2]}": v for k, v in eps.items()}, "dt_halving_ratio": ratio,
    "delta_control": {"deltas_R": deltas, "injected": "at the reversal (backward leg only)", "eps_rec_R": dctl,
                      "eps_rec_delta0_R": d0, "loglog_slope": slopes, "backward_leg_gain": gain,
                      "sph1_rel_change_delta1e-9": sph1_delta_indep},
    "growth_rate_post_hoc": growth, "roundoff_horizon_post_hoc": horizon,
    "smoke_N32_comparison": {"settings": "N=32, T=0.3 t_ff, dt=t_ff/600 (180+180 steps); the design smoke used 120 steps: a comparison at different settings, not a reproduction",
                             "nbody": pick(NB, N=32)["eps_rec_R"], "sph0": pick(SD, label="smoke_N32", alpha=0.0)["eps_rec_R"],
                             "sph1": pick(SD, label="smoke_N32", alpha=1.0)["eps_rec_R"], "design_smoke": [1.7e-16, 2.3e-15, 1.5e-2]},
    "ledger": {"Sdot_led_min_alpha1": sdot_min, "Sdot_led_negative_steps_all": sdot_neg,
               "engine_q_visc_step_min_J": qv_min, "engine_q_visc_negative_steps": qv_neg,
               "isothermal_T1": {n: {k: r1[n][k] for k in ("Q_led_end_J", "Q_visc_eng_end_J", "Q_led_over_absU0",
                                 "closure_literal_end", "closure_corrected_end", "closure_engine_vs_led_end",
                                 "total_drift_max_over_absU0", "total_drift_end_over_absU0", "S_led_cum_J_per_K_end",
                                 "r_median_at_T_over_R")} for n in (600, 1200)},
               "isothermal_alpha0_T1": {n: {k: r0[n][k] for k in ("bath_at_T", "Q_press_eng_end_J", "dF_iso_end_over_absU0")} for n in (600, 1200)},
               "post_hoc": post_hoc},
    "ideal_gas_entropy": ideal_entropy, "sph_cache_invalidation": cache, "cloud_size": size, "fourier": four,
    "criteria_original": {"readings": readings, "verdict": verdict_original, "S8_sentinel_literal": s8_literal},
    "criteria_corrected": {"clauses": c, "verdict": verdict_corrected, "failed": failed_c,
                           "separate_if_D3_dropped": all(c[k] for k in ("H1", "H2", "D1", "D2")),
                           "S8_sentinel_frozen_C3": s8_frozen, "could_it_fail": could_fail},
    "wrong_predictions": wrong_predictions, "addendum_2_scorecard": add2_score,
}

# ---------------- figure
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"]
plt.rcParams.update({"axes.unicode_minus": True, "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlesize": 9.5})
fig, axs = plt.subplots(2, 3, figsize=(15.5, 9.6), facecolor=SURF)
for ax in axs.flat:
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
M = lambda s_: s_.replace("e-", "e\u2212")

ax = axs[0, 0]
cats = [("nbody", "N-body\n(gravity)"), ("sph0", "SPH α = 0\nisothermal"), ("ideal0", "SPH α = 0\nideal gas"),
        ("sph1", "SPH α = 1\nisothermal"), ("ideal1", "SPH α = 1\nideal gas")]
x = np.arange(len(cats))
for j, (n, lab) in enumerate(((600, "dt = t_ff/600"), (1200, "dt = t_ff/1200"))):
    ax.scatter(x + (j - 0.5) * 0.22, [eps[(k, 1.0, n)] for k, _ in cats], s=46, color=COL[j], label=lab, zorder=3)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([c_[1] for c_ in cats], fontsize=8)
ax.axhline(1e-8, color=INK2, ls="--", lw=1); ax.axhline(1e-3, color=INK2, ls=":", lw=1)
ax.text(4.45, 1.6e-8, "1e\u22128 R (H1)", color=INK2, fontsize=8, ha="right")
ax.text(4.45, 1.6e-3, "1e\u22123 R (D1)", color=INK2, fontsize=8, ha="right")
ax.annotate("halves with dt:\nengine T-lag defect", xy=(2.1, 2.7e-4), xytext=(2.25, 1e-6), fontsize=7.5,
            color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_ylim(1e-17, 3); ax.set_ylabel("echo error  rms|x(2T) − x(0)| / R")
ax.set_title("(a) Forward 1 t_ff, reverse every velocity, back 1 t_ff", loc="left")
ax.legend(frameon=False, loc="upper left", fontsize=8)

ax = axs[0, 1]
lbl = {"nbody": "N-body", "sph0": "SPH α = 0 isothermal", "sph1": "SPH α = 1 isothermal"}
for j, k in enumerate(("nbody", "sph0", "sph1")):
    ax.plot([1e-15] + list(deltas), [d0[k]] + dctl[k], "o-", color=COL[j], label=lbl[k], ms=5, lw=1.3)
dd = np.array([1e-14, 1e-8])
ax.plot(dd, dd, ls="--", color=INK2, lw=1, label="error = δ")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([1e-15, 1e-13, 1e-11, 1e-9]); ax.set_xticklabels(["0", "1e\u221213", "1e\u221211", "1e\u22129"])
ax.set_xlabel("displacement δ / R injected at the reversal"); ax.set_ylabel("echo error / R")
ax.set_title(f"(b) Error ∝ δ (gain {g_med['nbody']:.0f} and {g_med['sph0']:.0f} over the 1 t_ff backward leg);\n"
             "viscous error does not depend on δ", loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower right")

ax = axs[0, 2]
ax.plot(Ts, [hz[t] for t in Ts], "o-", color=COL[0], ms=5, lw=1.3, label="N-body, roundoff only (dt = t_ff/600)")
ax.plot([0.3, 1.0], [eps[("sph0", 0.3, 600)], eps[("sph0", 1.0, 600)]], "s-", color=COL[1], ms=5, lw=1.2,
        label="SPH α = 0 isothermal")
ax.plot([0.3, 1.0], [eps[("sph1", 0.3, 600)], eps[("sph1", 1.0, 600)]], "^-", color=COL[2], ms=6, lw=1.2,
        label="SPH α = 1 isothermal")
ax.axhline(1e-8, color=INK2, ls="--", lw=1); ax.axhline(1e-3, color=INK2, ls=":", lw=1)
ax.set_yscale("log"); ax.set_ylim(1e-17, 3)
ax.set_xlabel("echo length T / t_ff (each leg)"); ax.set_ylabel("echo error / R")
ax.set_title("(c) Post-hoc: the Hamiltonian echo stays at roundoff only\ninside a horizon of a few t_ff", loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower right")

ax = axs[1, 0]
s_ = r1[1200]["series"]; tf = nb["choices"]["tff"]
ax.plot(np.array(s_["t_s"]) / tf, s_["Sdot_led"], color=COL[2], lw=1.2, label="SPH α = 1 isothermal (Q̇ / 300 K)")
s_ = r0[1200]["series"]
ax.plot(np.array(s_["t_s"]) / tf, s_["Sdot_led"], color=COL[1], lw=1.2, label="SPH α = 0 (identically 0)")
ax.axvline(1.0, color=INK2, ls="--", lw=1); ax.text(1.03, 3.0, "velocities reversed", color=INK2, fontsize=8)
ax.set_xlabel("t / t_ff  (forward 0–1, reversed 1–2)"); ax.set_ylabel("viscous entropy production  Ṡ  (J/K/s)")
ax.set_title("(d) Dissipation does not run backward when v → −v\n(Ṡ ≥ 0 is built into Π_ij)", loc="left"); ax.set_ylim(-6, 150)
ax.legend(frameon=False, fontsize=8, loc="upper left")

ax = axs[1, 1]
for j, (r, lab) in enumerate(((i1[1200], "α = 1: ΔS from (T_i, ρ_i)"), (i0[1200], "α = 0: ΔS from (T_i, ρ_i)"))):
    s_ = r["series"]
    ax.plot(np.array(s_["t_s"]) / tf, np.array(s_["S_state"]) - s_["S_state"][0], color=COL[[2, 1][j]], lw=1.3, label=lab)
s_ = i1[1200]["series"]
ax.plot(np.array(s_["t_s"]) / tf, s_["dS_visc_eng"], color=COL[3], lw=1.0, ls="--", label="α = 1: Σ q_visc / T booked")
ax.axvline(1.0, color=INK2, ls="--", lw=1)
ax.set_xlabel("t / t_ff  (forward 0–1, reversed 1–2)"); ax.set_ylabel("gas entropy change  ΔS  (J/K)")
ax.set_title("(e) Ideal gas: the net ΔS over the echo comes from viscosity;\nthe small α = 0 forward-leg rise is undone on the way back", loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper left")

ax = axs[1, 2]
rows = fr["rows"]; tt = np.array([q["t_s"] for q in rows]); tt = tt / tt[-1]
S_ = np.array([q["S_J_per_K"] for q in rows]); Sd_ = np.array([q["Sdot_W_per_K"] for q in rows])
ax.plot(tt, (S_ - S_[0]) / (S_[-1] - S_[0]), "o-", color=COL[1], ms=4, lw=1.2, label="entropy produced (fraction of final)")
ax.plot(tt, Sd_ / Sd_[0], "s-", color=COL[0], ms=4, lw=1.2, label="Ṡ = Σ k ΔT² / (T_a T_b) dx  (fraction of initial)")
ax.set_ylim(-0.05, 1.08)
ax.set_xlabel("t / t_run  (diffuse_fvm, 32³, insulated)"); ax.set_ylabel("fraction")
ax.set_title("(f) Fourier conduction: Ṡ > 0 in every slice\n(by construction, k > 0)", loc="left")
ax.legend(frameon=False, fontsize=8, loc="center right")

def sci(v):
    m, e = f"{v:.1e}".split("e")
    return f"{m}e{int(e)}"
cl = r1[1200]["closure_corrected_end"]
head = ("In this engine, reversing every velocity at T = 1 t_ff undoes gravity and isothermal inviscid SPH to roundoff "
        f"({sci(eps[('nbody', 1.0, 1200)])} R, {sci(eps[('sph0', 1.0, 1200)])} R) but not viscous SPH "
        f"({eps[('sph1', 1.0, 1200)]:.2f} R):\nthe only arrow of time is the dissipation the engine was built with. "
        f"Verdict MIXED under both frozen criteria (ledger closure {sci(cl)} > 1e-3).\n"
        + (f"Post hoc: the gravity echo is reversible only inside a roundoff horizon; by {cross_1e3:.0f} t_ff its error alone "
           f"({hz[cross_1e3]:.2f} R, roundoff-seeded) is larger than the viscous echo at 1 t_ff." if cross_1e3 is not None else
           f"Post hoc: the gravity echo error grows to {sci(hz[Ts[-1]])} R by {Ts[-1]:.0f} t_ff."))
fig.suptitle(M(head), fontsize=11, color=INK, x=0.01, ha="left", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig_path = os.path.join(HERE, "u2_loschmidt_echo_v2.png")
fig.savefig(fig_path, dpi=150, facecolor=SURF)
results["figure"] = fig_path
with open(os.path.join(HERE, "u2_results_v2.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1, default=lambda z: str(z))
for k in ("dt_halving_ratio", "growth_rate_post_hoc", "roundoff_horizon_post_hoc", "sph_cache_invalidation",
          "cloud_size", "ideal_gas_entropy", "criteria_original", "criteria_corrected", "wrong_predictions", "addendum_2_scorecard"):
    print(k, json.dumps(results[k], indent=1, default=str)[:4000])
print(json.dumps(four["v2_conditioning"], indent=1, default=str)[:2500])
print(json.dumps(MENTAT, indent=1))
