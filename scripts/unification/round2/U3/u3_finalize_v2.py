"""U3 repair (v2) finalize: fold the corrected wording (headline, design corrections, defects, references, repairs)
into u3_results_v2.json. Numbers are read from the v2 analysis, not typed in, wherever a sentence quotes one."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "u3_results_v2.json")
R = json.load(open(P, encoding="utf-8"))
S = R["summary"]; L = R["mentat_leaves"]; ps = R["per_seed"]; seeds = [str(s) for s in R["choices"]["seeds"]]
lv = lambda k: float(L[k]["value"])
f2 = lambda v: f"{v:.2f}"; f3 = lambda v: f"{v:.3f}"
pre, ext, lng, early = S["pre"], S["ext"], S["long_posthoc"], S["early_3_5_posthoc"]
tr = lambda s: ps[s]["time_resolved"]

R["headline_v2"] = (
    "Adding energy to a gate-passing N = 128 Plummer cluster lowered its mean kinetic energy in all 12 kicked branches: "
    f"d<K>/dE = {f2(pre['slope_raw']['mean'])} +- {f2(pre['slope_raw']['sem'])} (seed SEM; seed std {f2(pre['slope_raw']['std'])}; "
    f"per seed {', '.join(f2(v) for v in pre['slope_raw']['per_seed'])}) in the pre-registered window, against +1 for an ideal gas "
    "(a formula, not a simulation). The pre-registered verdict is UNRESOLVED because the seed std exceeds the 0.05 limit; "
    f"the extended window (3 to 4.74 design t_relax, cut short of 6) gives {f2(ext['slope_raw']['mean'])} +- {f2(ext['slope_raw']['std'])}, "
    "also UNRESOLVED. The kicked clusters' moment of inertia is still growing relative to the control at 12.5 design units in every "
    "seed and the branches decorrelate after about 2-3 units, so no turnaround, breathing period or damping was measured. By the "
    "energy plus Lagrange-Jacobi identity the miss from -1 sits, by construction, in the non-stationarity term "
    f"({f3(pre['decomposition']['slope_L']['mean'])}) plus a softening term (+{f3(pre['decomposition']['slope_UmW']['mean'])}). "
    "This validates the virial theorem's sign on the engine; it is not evidence for the unification thesis.")

R["design_corrections_v2"] = [
    # 1 crossing-time convention (rewritten)
    "Time unit. The design's t_cross = sqrt(b^3/GM) = %.0f s is shorter than every standard crossing time of a Plummer sphere "
    "(Mentat simplify_expr): %.2fx shorter than R_V/v_rms, %.2fx shorter than r_half/v_rms (the half-diameter R/v crossing time that "
    "goes with t_relax = 0.1 N/lnN t_cross; the textbook derivation gives N/(8 lnN) = %.2f at N = 128 against 0.1 N/lnN = %.2f), "
    "and %.2fx shorter than the Heggie-Mathieu full-diameter t_cr = 2 R_V/v_rms. So the design's 3 t_relax run is %.2f (R_V) to "
    "%.2f (r_half) relaxation times in the matching convention, and %.2f only in the Heggie-Mathieu convention (v1 quoted only 0.48). "
    "Convention-free: the pre-registered window is %.2f design units = %.2f, and the whole post-kick run %.2f, of the circular-orbit "
    "period at r_half (%.2f units, Mentat orbital_period), so phase mixing after the kick cannot be complete." % (
        R["choices"]["t_cross_design_s"], lv("R_V_over_v_rms_units"), lv("r_half_over_v_rms_units"), lv("N_over_8lnN_N128"),
        lv("trelax_over_tcross_N128"), lv("t_cr_HM_over_unit"), lv("run_3trelax_in_trelax_RV"), lv("run_3trelax_in_trelax_rh"),
        lv("run_3trelax_in_trelax_HM"), ps["1"]["pre"]["window_units"], lv("pre_window_over_T_orb"), lv("post_kick_run_over_T_orb"),
        lv("T_orb_r_half_units")),
    # 2 seed-std gate (rewritten on a noise basis)
    "Seed-std gate. The gate is on the per-seed std, so adding seeds cannot meet it; only lower per-seed scatter can. The observed "
    "scatter (%.3f pre-registered) has two candidate sources this run cannot apportion: (i) fast finite-N fluctuations of branches "
    "that have decorrelated (relax-phase K, integrated autocorrelation time 0.5-1.2 units) predict %s per seed, about the within-seed "
    "4-point scatter (rms SE %.3f); (ii) slow, seed-specific evolution of K over several units (control-branch K over 1-12.5 units, "
    "tau_int 1.4-2.1 units) predicts %s, about the observed scatter. Between-seed vs within-seed variance: F = %.1f (p = %.3f, F(2,6)) "
    "in the pre-registered window only; F = %.2f (ext) and %.2f (long), so no robust seed-level excess after allowing for the four "
    "windows examined. Cutting the scatter about %.1fx to 0.05 needs, by noise scaling, about 20x longer windows (about 50 design units, "
    "4 T_orb(r_half)) or N of order 2,500; delta cannot be raised much (delta = 0.2 already injects 0.44 K0). And since the kicked "
    "clusters were still expanding at 12.5 units, a longer window would also have to wait for the expansion to finish. Alternatives: "
    "state the criterion on the seed SEM, or on the coherent pre-decorrelation phase. The v1 rationale ('several breathing periods, "
    ">~ 40 units') is withdrawn: no breathing period was measured." % (
        pre["slope_raw"]["std"], "/".join(f2(v) for v in pre["noise_model_slope_sigma"]["relax"]), pre["between_within"]["rms_within_seed_se"],
        "/".join(f2(v) for v in pre["noise_model_slope_sigma"]["control"]), pre["between_within"]["F_ratio"], pre["between_within"]["p_F_2_6"],
        ext["between_within"]["F_ratio"], lng["between_within"]["F_ratio"], pre["slope_raw"]["std"] / 0.05),
    # 3 LJ-corrected (kept) + identity is by construction (new)
    "The design's 'LJ-corrected' K_LJ = <K> - (1/4)[Idot]/T equals -<W_s>/2 and removes only half of the non-stationarity term L = "
    "(1/2)[Idot]/T; removing all of it (<K> - L) returns -1 + d<U_s - W_s>/dE by construction (%.3f +- %.3f). The decomposition "
    "<K> = -E + L + <U_s - W_s> is an identity [BY CONSTRUCTION] for any finite window of a conservative pairwise system: it books "
    "every departure from the virial value as L, whether it comes from a coherent mode or from noise, so it cannot identify a "
    "mechanism. Its non-trivial content is narrow: the softening term is only +%.3f (pre) / +%.3f (ext), the energy slope is %.6f "
    "(injection exact), and the identity closes to <= 4e-5. The raw slope is the test." % (
        pre["decomposition"]["slope_K_minus_L"]["mean"], pre["decomposition"]["slope_K_minus_L"]["std"],
        pre["decomposition"]["slope_UmW"]["mean"], ext["decomposition"]["slope_UmW"]["mean"], pre["decomposition"]["slope_E"]["mean"]),
    # 4 verdict against both criteria (kept, with corrected errors)
    "Against the original criterion: pre-registered window UNRESOLVED (mean %.3f in band, seed std %.3f > 0.05, end-point drift %.1e "
    "passes); SAME excluded (seed mean below zero at t = %.1f, 2 dof, one-sided p = %.4f). Against the corrected criterion (extended "
    "window, same thresholds): UNRESOLVED (%.3f +- %.3f, below -1.10). Post hoc, the longest window (2-4.74 t_relax) gives %.2f +- %.2f "
    "(error from the within-seed fit SEs; the 3-seed std %.3f is a small-sample accident, 95%% CI on sigma [%.3f, %.3f])." % (
        pre["slope_raw"]["mean"], pre["slope_raw"]["std"], pre["drift_gate_value"], pre["t_seed_mean_below_zero"], pre["p_one_sided_t2"],
        ext["slope_raw"]["mean"], ext["slope_raw"]["std"], lng["slope_raw"]["mean"], lng["sem_from_within_seed_se"],
        lng["slope_raw"]["std"], *lng["seed_std_95CI_chi2"]),
    # 5 drift (kept)
    "The design leaves 'drift' undefined. The pre-registered end-point drift passes (%.1e); the max-over-time excursion (%.1e) "
    "would fail a strict reading." % (pre["drift_gate_value"], max(ps[s]["drift_excursion_max_all"] for s in seeds)),
    # 6 gate does not mean virialised at the kick (kept, with numbers)
    "The gate is a 5-unit time average and passes (%s), but it does not make the cluster virialised at the kick: the instantaneous "
    "2K/|W_s| at injection was %s, it ranged %s inside the gate windows, and the delta = 0 controls swing over %s. 'Relaxed' is "
    "replaced by 'gate-passing'." % (
        ", ".join(f3(ps[s]["gate"]["virial_2K_over_W"]) for s in seeds), ", ".join(f3(ps[s]["virial_inst_at_kick"]) for s in seeds),
        ", ".join("%.2f-%.2f" % tuple(ps[s]["virial_inst_range_gate_window"]) for s in seeds),
        ", ".join("%.2f-%.2f" % tuple(ps[s]["virial_inst_range_control_branch"]) for s in seeds)),
    # 7 softened expectation (new)
    "The softened-virial value K = E w/(w-2) (Mentat solve_equation), %.4f here, holds at fixed w and leaves out E dw/dE. The "
    "consistent expectation including dw/dE is -1 + d<U_s - W_s>/dE = %.3f (pre) / %.3f (ext)." % (
        pre["softened_virial_fixed_w"]["mean"], pre["decomposition"]["consistent_softened_expectation"]["mean"],
        ext["decomposition"]["consistent_softened_expectation"]["mean"]),
    # 8 breathing claim withdrawn (new)
    "Withdrawn from v1: 'a coherent and slowly damped breathing oscillation, not sampling noise' and 'the 13-unit breathing period'. "
    "13.24 units is the Kepler period at r_half (Mentat), a proxy, never a measured period. Measured instead: the moment-of-inertia "
    "response (dI/dE)|E|/I0 rises at every step after 1 unit in all seeds, reaching %s at 12.5 units (homologous re-virialised value 2); "
    "dIdot/dE stays positive throughout, peaking at %s units, so in the extended window the expansion is decelerating, which by "
    "Lagrange-Jacobi puts 2K + W_s below zero and <K> below the virial value (10 of 12 branch responses below -1 there). The half-mass "
    "radius response levels off after 5-9 units near the homologous value 1 (seeds 2, 3: %s and %s at 12 units) or above it (seed 1: %s). "
    "The Plummer I integrand goes as 1/x at large r (Mentat series_expansion), so I is halo-weighted: the halo is still expanding." % (
        ", ".join(f2(tr(s)["dIdE_absE_over_I0_at_end"]) for s in seeds), ", ".join("%.1f" % tr(s)["dIdot_dE_max_at_units"] for s in seeds),
        f2(tr("2")["dlnrh_ddE_1u_at"]["12"]), f2(tr("3")["dlnrh_ddE_1u_at"]["12"]), f2(tr("1")["dlnrh_ddE_1u_at"]["12"])),
]

R["engine_defects_v2"] = [
    dict(file="D:/Aaron/development/SkippyTel/mentat/app.py", line=185,
         defect="Mentat gateway power-tower guard false positive: the second alternative of the _HUGE regex (defined from line 184, "
                "the power-tower pattern on line 185) refuses any expression of the form (...x^2)^(p) as 'looks like code'.",
         evidence="integrate_expr {'expression': '3*x^2*(1+x^2)^(-5/2)', 'variable': 'x'} returns 'That input looks like code, not maths' "
                  "(v1, reproduced by two skeptics); the pattern (\\^|\\*\\*)\\s*\\(?\\s*[\\d.]+\\s*\\)?\\s*(\\^|\\*\\*) matches '^2)^'. "
                  "Workaround used in v2: write the power as sqrt(1+x*x)^5 (series_expansion and integrate_expr accept it). Not a pass/fail input."),
    dict(file="D:/Aaron/development/sigma-ground/sigma_ground/field/interface/nbody.py", line=1143,
         defect="Already-filed PHYSICS_GAP, not new: NBodySystem.total_energy() sums the unsoftened 1/r pair potential while the "
                "integrator conserves K + U_soft (cite by function name; it was at line 1123 in v1 and moved as other sessions edited the file).",
         evidence="This run computes its own U_s and W_s; softened energy end-point drift <= 2.8e-5 per segment."),
]
R["lit_refs_v2"] = [
    dict(ref="Lynden-Bell D. 1999, 'Negative specific heat in astronomy, physics and chemistry', Physica A 263, 293-304, "
             "doi:10.1016/S0378-4371(98)00518-4, arXiv:cond-mat/9812172",
         status="verified in this round by the physics skeptic (arXiv abstract page and Crossref API); not re-searched in the repair"),
    dict(ref="Heggie D.C. & Mathieu R.D., 'Standardised units and time scales', in The Use of Supercomputers in Stellar Dynamics, "
             "Lecture Notes in Physics, pp. 233-235, doi:10.1007/BFb0116419",
         status="title, pages and DOI verified via Crossref by the physics skeptic; volume 267 and year 1986 NOT re-verified"),
    dict(ref="Aarseth S.J., Henon M. & Wielen R. 1974, A&A 37, 183 (bibcode 1974A&A....37..183A)",
         status="NOT verified (ADS refused the skeptic's fetch; the web-search budget was exhausted in this repair); cited only as the "
                "origin of the design sketch's sampler, not for any number"),
    dict(ref="Binney & Tremaine convention t_relax = 0.1 N/lnN x t_cross with t_cross = R/v",
         status="checked by the physics skeptic against Wikipedia 'Stellar dynamics' (t_cross = R/v); NOT re-verified in this repair "
                "(web-search budget exhausted). The design-unit ratios themselves are Mentat [DERIVED] and do not depend on it"),
]
R["repairs_v2"] = json.load(open(os.path.join(HERE, "u3_repairs_v2.json"), encoding="utf-8")) if os.path.exists(
    os.path.join(HERE, "u3_repairs_v2.json")) else None
json.dump(R, open(P, "w", encoding="utf-8"), indent=1)
print(R["headline_v2"]); print()
for i, d in enumerate(R["design_corrections_v2"], 1):
    print(i, d); print()
