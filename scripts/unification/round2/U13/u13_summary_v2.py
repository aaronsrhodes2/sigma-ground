"""U13 v2 summary: merges u13_results_v2.json, u13_supp_small_v2.json, u13_supp_big_v2.json, u13_derive_v2.json
into u13_summary_v2.json (verdicts by rule and protocol, h-slopes under both readings, resolution decomposition,
v2 prediction checks, physical rows, engine defects).  No physics is computed here beyond ratios/fits of
measured numbers; analytic rows come from u13_derive_v2.json (Mentat)."""
import json
import math
import os

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))


def load(n):
    return json.load(open(os.path.join(OUT, n), encoding="utf-8"))


d = load("u13_results_v2.json")
sm = load("u13_supp_small_v2.json")
bg = load("u13_supp_big_v2.json")
dv = load("u13_derive_v2.json")
A, B, S, K2 = d["analysis"]["runA"], d["analysis"]["runB"], d["analysis"]["seeds"], d["analysis"]["literal_step50"]
src = d["mentat_sources"]


def verdict(x):
    return "FORBID" if x > 0.1 else ("ALLOW" if x < 0.01 else "GRAY")


def band(x, c=1.0, w=0.3):
    return "PASS" if x is not None and abs(x - c) <= w else "FAIL"


S2 = {str(x["N"]): x for x in sm["S2"] + bg["S2"]}
NS = [64, 128, 256, 512]


def samp(n, st):
    return [s for s in S2[str(n)]["samples"] if s["step"] == st][0]


# ---- resolution decomposition
res_by_t = {}
for st, lab in ((50, "0.083"), (300, "0.5"), (600, "1.0")):
    Ns = np.array(NS, float)
    rat = np.array([samp(n, st)["ratio_alpha1"] for n in NS])
    C = [samp(n, st)["C_kin"] for n in NS]
    div = np.array([abs(samp(n, st)["div_fit_s"]) for n in NS])
    loc = [math.log(rat[i + 1] / rat[i]) / math.log(Ns[i + 1] / Ns[i]) for i in range(3)]
    p_all = float(np.polyfit(np.log(Ns), np.log(rat), 1)[0])
    p_64_256 = float(np.polyfit(np.log(Ns[:3]), np.log(rat[:3]), 1)[0])
    p_div = float(np.polyfit(np.log(Ns), np.log(div), 1)[0])
    Cm = float(np.mean(C[:3]))
    res_by_t[lab] = dict(N=NS, ratio=rat.tolist(), verdict=[verdict(x) for x in rat], C_kin=C,
                         C_kin_mean_N64_256=Cm, C_kin_N512_rel_to_mean=C[3] / Cm - 1.0,
                         P2_C_flat_pass=all(abs(c / Cm - 1.0) <= 0.15 for c in C),
                         abs_div_fit_s=div.tolist(), div_exponent=p_div, local_exponents=loc,
                         fit_exponent_all_N=p_all, fit_exponent_N64_256=p_64_256,
                         N_for_0p01_SPECULATIVE={"fitted_exponent_from_N512": float(512 * (rat[3] / 0.01) ** (-1 / p_all)),
                                                 "minus_one_third_from_N512": float(512 * (rat[3] / 0.01) ** 3)})

# ---- h-slopes under both readings
hs = {"N64_RunA_step50(1.6e-4 t_ff)": A["h_slopes_step50"], "N64_RunA_step100(3.2e-4 t_ff)": A["h_slopes_step100"],
      "N64_RunB_step50(0.083 t_ff)": B["h_slopes_step50"], "N64_RunB_step300(0.5 t_ff)": B["h_slopes_step300"],
      "N64_RunB_step600(1 t_ff)": B["h_slopes_step600"]}
for n in (128, 256, 512):
    for st, lab in ((50, "0.083"), (300, "0.5"), (600, "1")):
        hs[f"N{n}_step{st}({lab} t_ff)"] = samp(n, st)["slopes"]
hs_tab = {k: {"reading_S": v["reading_S_spacing_x_1_1.5_2"], "verdict_S": band(v["reading_S_spacing_x_1_1.5_2"]),
              "reading_H": v["reading_H_h0_x_1_1.5_2"], "verdict_H": band(v["reading_H_h0_x_1_1.5_2"])}
          for k, v in hs.items()}
nn64 = {r["h_over_spacing"]: r["mean_neighbours"] for r in B["fine_hscan_step50"]["rows"]}

# ---- whole-collapse criterion per N (samples 50/300/600 for N >= 128; full grid for N = 64)
wc_by_N = {"64": dict(max_ratio=B["max_ratio_alpha1_0_to_1tff"], verdict=B["scope_rule_whole_collapse_preregistered_ii"],
                      sampled="every 0.05 t_ff")}
for n in (128, 256, 512):
    mx = max(s["ratio_alpha1"] for s in S2[str(n)]["samples"])
    wc_by_N[str(n)] = dict(max_ratio=mx, verdict=verdict(mx), sampled="0.083, 0.5, 1 t_ff only")

# ---- flow structure at N = 64, alpha = 1
stc = B["structure_alpha1"]
s0 = d["runB"][2]["rows"][0]["structure"]
s1tff = [r for r in d["runB"][2]["rows"] if r["step"] == 600][0]["structure"]
struct = dict(rms_radius_over_R_t0=s0["rms_radius_over_R"], rms_radius_over_R_1tff=s1tff["rms_radius_over_R"],
              rms_radius_fractional_drop=1 - s1tff["rms_radius_over_R"] / s0["rms_radius_over_R"],
              v_rms_over_cs_1tff=s1tff["v_rms_over_cs"], v_hom_at_R_over_cs_1tff=s1tff["v_hom_at_R_over_cs"],
              v_hom_at_rmax_over_cs_1tff=s1tff["v_hom_at_rmax_over_cs"],
              v_residual_rms_over_cs_1tff=s1tff["v_residual_rms_over_cs"],
              v_rms_over_cs_by_t={k: v["v_rms_over_cs"] for k, v in stc.items()},
              v_residual_rms_over_cs_by_t={k: v["v_residual_rms_over_cs"] for k, v in stc.items()},
              v_hom_at_rmax_over_cs_by_t={k: v["v_hom_at_rmax_over_cs"] for k, v in stc.items()},
              rms_radius_over_R_by_t={k: v["rms_radius_over_R"] for k, v in stc.items()},
              thermal_virial_ratio_2K_over_W=dv["thermal_virial_ratio_2K_over_W"])

rowsB1 = [r for r in d["runB"][2]["rows"] if r["t_over_tff"] <= 1 + 1e-9 and r.get("C_kin")]
s1 = {str(x["N"]): x for x in sm["S1"]}
m_p = d["choices"]["M_kg"] / d["choices"]["N"]

summary = {
    "verdict_by_rule": {
        "design_literal_step50_rule_by_protocol": K2["literal_step50_rule_by_protocol"],
        "note": "the design's 'step 50' depends on dt; only physical-time statements are protocol-free",
        "ratio_alpha1_by_t_over_tff_N64": {k: v for k, v in B["ratio_alpha1_by_t_over_tff"].items()},
        "preregistered_ii_whole_collapse_0_to_1tff_N64": [B["max_ratio_alpha1_0_to_1tff"],
                                                         B["scope_rule_whole_collapse_preregistered_ii"]],
        "ratio_still_rising_at_1tff": B["ratio_still_rising_at_1tff"],
        "whole_collapse_by_N": wc_by_N,
        "seeds_alpha1": {k: S[k] for k in ("7", "1", "2", "3")}, "seed_range_step50": S["range_step50"],
        "seed_range_1tff": S["range_step600"], "seed_verdicts_step50": S["verdicts_step50"],
        "seed_verdicts_1tff": S["verdicts_1tff"],
        "dt_control_max_rel_diff_tff_over_1200": B["dt_control_max_rel_diff"],
        "literal_dt_tff_over_100_vs_200_rel_diff_at_0p5tff": K2["rel_diff_0p5tff"],
    },
    "runA_design_protocol": {
        "ratio_alpha1_step50": A["ratio_alpha1_step50"], "t_step50_over_tff": A["t_step50_over_tff"],
        "t_step100_over_tff": A["t_step100_over_tff"], "verdict": A["scope_rule_design_step50"],
        "alpha_slope_step50": A["step50"]["alpha_slope"], "alpha_slope_step100": A["step100"]["alpha_slope"],
        "alpha_slope_tag": "BY CONSTRUCTION", "alpha_independence_of_state": d["analysis"]["runA_alpha_independence_step50"],
        "step0_ratio": A["step0"]["ratios"], "design_prediction_0p1alpha_measured_over_predicted": A["prediction_0p1alpha_measured_over_predicted"],
    },
    "runB_alpha_slope": {"t_over_tff": B["t_over_tff"], "slope": B["alpha_slope_vs_time"],
                         "first_below_0p85": B["alpha_slope_first_below_0p85_t_over_tff"],
                         "rms_radius_by_alpha_at_1tff": B["rms_radius_by_alpha_at_1tff"]},
    "h_slopes_both_readings": hs_tab,
    "h_scan_neighbours_N64_step50": nn64,
    "fine_hscan_N64_step50": B["fine_hscan_step50"]["slopes"],
    "S1_homologous_operator_check": {k: dict(slopes=v["slopes"], slope_all=v["slope_all"], C_implied=v["C_implied"])
                                     for k, v in s1.items()},
    "C_identity_DERIVED": {"f_eta": dv["f_eta"], "C_engine_mu_form": dv["C_engine_mu_form"],
                           "C_eta_to_0": dv["C_engine_mu_form_eta_to_0"], "C_vsig_form_M4": dv["C_vsig_form"],
                           "mean_q_W": dv["mean_q_W"]},
    "resolution_by_t_over_tff": res_by_t,
    "extension_post_hoc_N64": B["extension_post_hoc"],
    "flow_structure_N64_alpha1": struct,
    "physical_rows": {
        "eta_iron_eyring_Pa_s_upper_bound": src["eta_iron_eyring_at_Tmelt"]["value"],
        "zeta_eff_alpha1_Pa_s_range": [min(r["zeta_eff_Pa_s"] for r in rowsB1), max(r["zeta_eff_Pa_s"] for r in rowsB1)],
        "C_kin_alpha1_range_N64": [min(r["C_kin"] for r in rowsB1), max(r["C_kin"] for r in rowsB1)],
        "ratio_phys_eyring_range_alpha1": [min(r["ratio_phys_eyring"] for r in rowsB1),
                                           max(r["ratio_phys_eyring"] for r in rowsB1)],
        "Knudsen": dv["Knudsen"], "Knudsen_with_sqrt2": dv["Knudsen_with_sqrt2"],
        "three_P_over_rho_c2": dv["three_P_over_rho_c2"],
        "freefall_ratio_per_t_over_tff": dv["freefall_ratio_per_t_over_tff"],
        "freefall_ratio_step50": dv["freefall_ratio_step50_dt_tff_over_600"],
    },
    "energy_drift_over_W0": {"runB_by_alpha": B["energy_drift_over_W0"],
                             "seeds": {k: S[k]["energy_drift_over_W0_to_1tff"] for k in ("1", "2", "3")}},
    "timestep": {k: dv[k] for k in ("dt_cfl_material_s", "dt_cfl_material_over_tff", "dt_cs_only_s", "dt_cs_only_over_tff",
                                    "dt_accel_0p3_s", "dt_accel_0p3_over_tff", "dt_accel_0p25_s", "dt_accel_0p25_over_tff",
                                    "dt_used_tff_over_600_over_dt_cfl", "dt_cs_only_over_dt_cfl",
                                    "design_100_steps_with_cs_only_cfl_span_tff", "c_material_iron_from_engine_m_s")},
}

# ---- v2 prediction checks (u13_prereg_v2.json)
r512 = {st: samp(512, st)["ratio_alpha1"] for st in (50, 300, 600)}
summary["v2_prediction_checks"] = {
    "P1_N512": dict(ratio_0p5tff=r512[300], predicted_0p5="0.085-0.094 GRAY", verdict_0p5=verdict(r512[300]),
                    ratio_1tff=r512[600], predicted_1tff="0.157-0.21 FORBID", verdict_1tff=verdict(r512[600]),
                    scope_statement=("FORBID extends to N = 512" if r512[600] > 0.1 else "FORBID confined to N <= 256")),
    "P2_C_flat": {lab: dict(C=res_by_t[lab]["C_kin"], pass_=res_by_t[lab]["P2_C_flat_pass"]) for lab in res_by_t},
    "P3_alt_h_reading": {k: v for k, v in hs_tab.items() if k.startswith(("N128", "N256", "N512"))},
    "P4_extension": dict(ratio_1p5tff=B["extension_post_hoc"]["ratio_1p5tff"], predicted="> 0.43",
                         held=B["extension_post_hoc"]["ratio_1p5tff"] > 0.43),
}
summary["engine_defects"] = [
    {"file": "sigma_ground/dynamics/continuum.py", "line": 936,
     "defect": "cfl_timestep takes c_max from p.sound_speed (material: iron) and ignores scene.eos / scene.sound_speed_m_s; "
               "it also has no acceleration (force) criterion",
     "evidence": summary["timestep"]},
    {"file": "sigma_ground/dynamics/continuum.py", "line": 559,
     "defect": "docstring of _sph_viscous_force writes Pi = -alpha h c mu / rho; the code (line 605) is Pi = -alpha c mu / rho "
               "(mu already carries h). The code is the standard Monaghan 1992 form; the docstring has a spurious h.",
     "evidence": "code read; S1 ratio / (alpha k h / 2 c_s) = 0.93-0.96 matches the code form (f_eta = 0.990), not an extra factor h"},
]
summary["engine_facts_not_defects"] = [
    "continuum.py:719 h = scene.h: one global, fixed smoothing length (no adaptive h). Late in the collapse the core is "
    "covered by many more neighbours than at release (mean neighbours at h0, 1 t_ff: see flow structure rows), which is why "
    "the share falls only weakly with N at 1 t_ff."]
summary["mentat_sources"] = {"runner (u13_av_share.SRC via u13_results_v2.json)": src,
                             "analytic rows (u13_derive_v2.json)": dv["mentat_sources"]}
summary["neighbours_at_h0_by_N_and_step"] = {str(n): {str(s["step"]): row_at_h0["mean_neighbours"]
                                                      for s in S2[str(n)]["samples"]
                                                      for row_at_h0 in [r for r in s["rows"] if r["h_over_spacing"] == 1.5]}
                                             for n in NS}
summary["core_concentration_k_inner_over_k_outer"] = {str(n): {str(s["step"]): s["structure"]["k_inner_s"] / s["structure"]["k_outer_s"]
                                                               for s in S2[str(n)]["samples"]} for n in NS}
summary["div_sph_over_div_fit_1tff"] = {str(n): [r for r in samp(n, 600)["rows"] if r["h_over_spacing"] == 1.5][0]["div_sph_s"]
                                        / samp(n, 600)["div_fit_s"] for n in NS}
with open(os.path.join(OUT, "u13_summary_v2.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=1, default=str)
print(json.dumps(summary, indent=1, default=str)[:20000])
