"""U13 summary: merges u13_results.json + u13_supp_results.json into u13_summary.json (verdicts, derived rows)."""
import json
import math
import os

import numpy as np

OUT = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U13"
d = json.load(open(os.path.join(OUT, "u13_results.json"), encoding="utf-8"))
s = json.load(open(os.path.join(OUT, "u13_supp_results.json"), encoding="utf-8"))
A, B = d["analysis"]["runA"], d["analysis"]["runB"]
ch = d["choices"]
src = d["mentat_sources"]

# collisionality of the swarm [DERIVED from Mentat leaves + geometry]
m_p = ch["M_kg"] / ch["N"]
r_p = (3 * m_p / (4 * math.pi * src["rho_iron"]["value"])) ** (1 / 3)
V = 4 / 3 * math.pi * ch["R_m"] ** 3
mfp = V / (ch["N"] * math.pi * (2 * r_p) ** 2)

rowsB1 = [r for r in d["runB"] if r["alpha"] == 1.0][0]["rows"]
C_B = [r["C_kin"] for r in rowsB1 if r.get("C_kin")]
s1 = s["S1"]
S2 = s["S2"]
Ns = [x["N"] for x in S2]
r50 = [x["samples"][0]["ratio_alpha1"] for x in S2]
N300 = [x["N"] for x in S2 if len(x["samples"]) > 1]
r300 = [x["samples"][1]["ratio_alpha1"] for x in S2 if len(x["samples"]) > 1]
e50 = float(np.polyfit(np.log(Ns), np.log(r50), 1)[0])
e300 = float(np.polyfit(np.log(N300), np.log(r300), 1)[0])

summary = {
    "pre_registered_as_written_RunA": {
        "ratio_alpha1_step50": A["step50"]["ratios"]["1.0"], "t_step50_over_tff": A["t_step50_over_tff"],
        "scope_rule": A["scope_rule"], "alpha_slope_step50": A["step50"]["alpha_slope"],
        "alpha_slope_pass": A["alpha_slope_pass"], "alpha_slope_tag": "BY CONSTRUCTION (state alpha-independent)",
        "h_slope_step50": A["h_slope_step50"], "h_slope_pass": A["h_slope_pass"],
        "design_prediction_0p1_alpha": "FAIL: measured/predicted = %.2e" % A["prediction_0p1alpha_ratio_measured_over_predicted"],
        "step0_ratio": 0.0, "step0_tag": "BY CONSTRUCTION (cold start, v = 0, every pair vr = 0 -> skipped)",
    },
    "corrected_RunB": {
        "dt_s": d["t_ff_s"] / 600, "ratio_alpha1_step50": B["ratio_alpha1_step50"], "scope_rule_step50": B["scope_rule_step50"],
        "max_ratio_alpha1_0_to_1tff": B["max_ratio_alpha1_0_to_1tff"], "scope_rule_whole_collapse": B["scope_rule_whole_collapse"],
        "t_cross_0p01_over_tff_upper_bound": B["t_cross_0p01_over_tff"], "t_cross_0p1_over_tff": B["t_cross_0p1_over_tff"],
        "alpha_slope_range": [min(x for x in B["alpha_slope_vs_time"] if x), max(x for x in B["alpha_slope_vs_time"] if x)],
        "alpha_slope_first_below_0p85_t_over_tff": next(t for t, x in zip(B["t_over_tff"], B["alpha_slope_vs_time"]) if x and x < 0.85),
        "h_slope_step50": B["h_slope_step50"], "h_slope_step300": B["h_slope_step300"],
        "dt_halving_max_rel_change_of_ratio": B["dt_control_max_rel_diff"],
        "C_kin_range_alpha1": [min(C_B), max(C_B)],
    },
    "supplementary_posthoc": {
        "S1_homologous_h_slope": {str(x["N"]): x["h_slope"] for x in s1},
        "S1_ratio_over_alpha_k_h_over_2cs": {str(x["N"]): x["ratio_over_alpha_k_h_over_2cs"] for x in s1},
        "S1_implied_C_kin": {str(x["N"]): float(np.mean(x["ratio_over_alpha_k_h_over_2cs"])) / 6.0 for x in s1},
        "LIT_C_kin": "1/6 = 0.1667 (zeta ~ alpha v_sig h / 6, v_sig ~ c_s; Price 2012 MNRAS 420 L33, arXiv:1111.1255, citing Lodato & Price 2010 MNRAS 405 1212)",
        "S2_ratio_step50_by_N": dict(zip(map(str, Ns), r50)), "S2_exponent_step50": e50,
        "S2_ratio_step300_by_N": dict(zip(map(str, N300), r300)), "S2_exponent_step300": e300,
        "S2_h_slope_step50_by_N": {str(x["N"]): x["samples"][0]["h_slope"] for x in S2},
        "S2_h_slope_step300_by_N": {str(x["N"]): x["samples"][1]["h_slope"] for x in S2 if len(x["samples"]) > 1},
        "S2_structure_step50_N64": S2[0]["samples"][0]["structure"],
        "extrapolated_N_for_ratio_below_0p01_at_0p5tff_SPECULATIVE": float(N300[-1] * (r300[-1] / 0.01) ** (-1 / e300)),
        "extrapolated_N_for_ratio_below_0p01_at_0p083tff_SPECULATIVE": float(Ns[-1] * (r50[-1] / 0.01) ** (-1 / e50)),
    },
    "physical_rows": {
        "eta_iron_eyring_Pa_s_upper_bound": src["eta_iron_eyring_at_Tmelt"]["value"],
        "zeta_eff_alpha1_Pa_s_range": [min(r["zeta_eff_Pa_s"] for r in rowsB1 if r.get("zeta_eff_Pa_s")),
                                       max(r["zeta_eff_Pa_s"] for r in rowsB1 if r.get("zeta_eff_Pa_s"))],
        "ratio_phys_eyring_range_alpha1": [min(r["ratio_phys_eyring"] for r in rowsB1 if r.get("ratio_phys_eyring")),
                                           max(r["ratio_phys_eyring"] for r in rowsB1 if r.get("ratio_phys_eyring"))],
        "swarm_particle_radius_m": r_p, "swarm_mean_free_path_m": mfp, "Knudsen_mfp_over_R": mfp / ch["R_m"],
        "three_P_over_rho_c2": ch["three_P_over_rho_c2"],
    },
    "engine_defect": {
        "file": "sigma_ground/dynamics/continuum.py", "line": 936,
        "defect": "cfl_timestep takes c_max from p.sound_speed (material: iron 5942 m/s) and ignores scene.eos / scene.sound_speed_m_s, "
                  "while continuum_step uses the isothermal c_s (or the ideal-gas sqrt(gamma R T/M)) as the signal speed",
        "evidence": {"dt_cfl_s": d["dt_cfl_initial_s"], "c_material_m_s": d["material_sound_speed_iron_m_s"],
                     "c_s_m_s": ch["c_s_m_s"], "t_ff_s": d["t_ff_s"],
                     "dt_overconservative_factor": d["material_sound_speed_iron_m_s"] / ch["c_s_m_s"]},
    },
}
with open(os.path.join(OUT, "u13_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary, indent=1))
