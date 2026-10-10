"""U5 v2 aggregate: pre-registered criteria (unchanged), skeptic repairs, every text block code-generated.

usage (repo root as cwd): python -X utf8 u5_aggregate_v2.py <outdir>
reads  leaves_v2.json, runs_v2/*.json, probe_v2.json, PREREG.txt (and runs/ from v1 for a determinism check)
writes u5_results_v2.json, u5_mercury_sigma_hook_v2.png
"""
import sys, os, json, math
import numpy as np
import sympy
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
from mentat_client import mentat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1]
L = json.load(open(os.path.join(D, "leaves_v2.json")))
PR = json.load(open(os.path.join(D, "probe_v2.json")))
PREREG = open(os.path.join(D, "PREREG.txt"), encoding="utf8").read()
MS = []          # Mentat calls made by this aggregator: (key, expression, value, source)


def num(k):
    v = L[k]["value"]
    return float(sympy.sympify(v).evalf(17)) if isinstance(v, str) else float(v)


def mx(key, expr):
    v, s = mentat("simplify_expr", {"expression": expr})
    val = float(sympy.sympify(v).evalf(17))
    MS.append({"key": key, "expression": expr, "value": val, "source": s})
    return val


dwGR = num("dw_GR_rad_per_orbit")
dwS_pred = num("dw_sigma_rad_per_orbit_pred")
ARC = num("rad_to_arcsec")
NCY = num("orbits_per_century")
XI = num("XI")
XI3, XI6 = num("xi_over_3"), num("xi_over_6")
RUNS = [("A", 600), ("B", 600), ("C", 600), ("D", 600), ("E", 600), ("A", 300), ("C", 300)]


def load(sub, mode, dt):
    return json.load(open(os.path.join(D, sub, f"run_{mode}_dt{dt}_n50.json")))


R = {(m, dt): load("runs_v2", m, dt) for m, dt in RUNS}

# ---- determinism check against the v1 runs (same code, fresh processes) ----
det = {}
for m, dt in RUNS:
    try:
        o = load("runs", m, dt)
        det[f"{m}_{dt}"] = {
            "max_abs_diff_peri_lrl_rad": float(np.max(np.abs(np.array(o["peri_lrl"]) - np.array(R[(m, dt)]["peri_lrl"])))),
            "lrl_slope_identical": o["lrl"] == R[(m, dt)]["lrl"],
            "interp_slope_identical": o["interp"] == R[(m, dt)]["interp"]}
    except FileNotFoundError:
        det[f"{m}_{dt}"] = "v1 file missing"


def diff(m, dt, est):
    s, e = R[(m, dt)][est][0], R[(m, dt)][est][1]
    s0, e0 = R[("A", dt)][est][0], R[("A", dt)][est][1]
    return s - s0, math.hypot(e, e0)


out = {"id": "U5", "version": "v2 (repair after three skeptic reviews)",
       "prereg_file": os.path.join(D, "PREREG.txt"), "prereg_text": PREREG,
       "provenance_note": "Every block in this JSON is written by u5_aggregate_v2.py from leaves_v2.json, runs_v2/, "
                          "probe_v2.json and PREREG.txt. In v1, design_corrections, coupling_convention_record, "
                          "engine_defects, prereg_text and lit_refs were added to u5_results.json by hand after "
                          "u5_aggregate.py ran; v2 removes that gap.",
       "leaves": L, "estimators": {}}
for est in ("lrl", "sketch", "interp"):
    E = {}
    vals = {k: diff(*a, est) for k, a in {"B_minus_A": ("B", 600), "C_minus_A": ("C", 600), "D_minus_A": ("D", 600),
                                          "E_minus_A": ("E", 600), "C_minus_A_dt300": ("C", 300)}.items()}
    E["A_newtonian_rad_per_orbit_dt600"] = R[("A", 600)][est][0]
    E["A_newtonian_rad_per_orbit_dt300"] = R[("A", 300)][est][0]
    for k, (d, e) in vals.items():
        E[k] = {"rad_per_orbit": d, "fit_se_rad_per_orbit": e, "arcsec_per_cy": d * ARC * NCY,
                "over_GR_analytic": d / dwGR, "over_GR_fit_se_absolute": e / dwGR,
                "fit_se_relative": e / abs(d)}
    dB, eB = vals["B_minus_A"]; dC, eC = vals["C_minus_A"]
    E["design_ratio_C_over_B"] = dC / dB
    E["design_ratio_se"] = abs(dC / dB) * math.hypot(eC / dC, eB / dB)
    E["dt_control_rel_change"] = (vals["C_minus_A_dt300"][0] - dC) / dC
    out["estimators"][est] = E

P = out["estimators"]["lrl"]
I = out["estimators"]["interp"]
ratio_design = P["design_ratio_C_over_B"]
rC = P["C_minus_A"]["over_GR_analytic"]; seC = P["C_minus_A"]["over_GR_fit_se_absolute"]
rB = P["B_minus_A"]["over_GR_analytic"]; rD = P["D_minus_A"]["over_GR_analytic"]; rE = P["E_minus_A"]["over_GR_analytic"]
sig_arc = P["C_minus_A"]["arcsec_per_cy"]; D_arc = P["D_minus_A"]["arcsec_per_cy"]
dev_xi3_rel = rC / XI3 - 1
dev_xi3_se = (rC - XI3) / seC

out["criteria"] = {
    "design_engine_pass": bool(0.0497 <= ratio_design <= 0.0557), "design_ratio": ratio_design,
    "design_ratio_se": P["design_ratio_se"],
    "corrected_engine_pass_C": bool(0.0497 <= rC <= 0.0557), "corrected_ratio_C": rC,
    "corrected_ratio_C_fit_se_absolute": seC, "corrected_ratio_C_fit_se_relative": P["C_minus_A"]["fit_se_relative"],
    "C_deviation_from_xi_over_3_relative": dev_xi3_rel, "C_deviation_from_xi_over_3_in_fit_se": dev_xi3_se,
    "B_over_GR_analytic": rB, "B_magnitude_within_1pct": bool(0.99 <= abs(rB) <= 1.01),
    "B_sign": "retrograde (negative)" if rB < 0 else "prograde (positive)",
    "P2_D_ratio": rD, "P2_D_pass": bool(abs(rD - XI6) <= 0.0015),
    "P3_E_ratio": rE, "P3_E_pass": bool(abs(rE + 5.0 / 3.0) <= 0.03),
    "P4_dt_control_rel_change": P["dt_control_rel_change"], "P4_dt_control_pass": bool(abs(P["dt_control_rel_change"]) < 0.01),
    "physics_threshold_arcsec_cy": 0.05, "physics_threshold_label": "the design's own conservative budget",
    "sigma_advance_arcsec_cy": sig_arc, "physics_excluded": bool(sig_arc > 0.05),
}

# ---- LRL vs parabola-interpolated estimator ----
out["lrl_vs_interp_rel_diff"] = {k: I[k]["rad_per_orbit"] / P[k]["rad_per_orbit"] - 1
                                 for k in ("B_minus_A", "C_minus_A", "D_minus_A", "E_minus_A", "C_minus_A_dt300")}

# ---- second method: Gauss quadrature on engine forces vs the 50-orbit runs ----
G_ = PR["gauss_quadrature"]
pairs = {"B_minus_A": "gr_1pn", "C_minus_A": "sigma_field_eq", "D_minus_A": "sigma_from_potential",
         "E_minus_A": "eih_cross", "C_minus_A_dt300": "sigma_field_eq"}
expect = {"B_minus_A": -1.0, "C_minus_A": XI3, "D_minus_A": XI6, "E_minus_A": -5.0 / 3.0, "C_minus_A_dt300": XI3}
out["two_methods"] = {}
for k, g in pairs.items():
    run = P[k]["over_GR_analytic"]; gq = G_[g]["over_GR_analytic"]
    out["two_methods"][k] = {"run_over_GR": run, "gauss_over_GR": gq, "expected": expect[k],
                             "run_over_gauss_minus_1_ppm": (run / gq - 1) * 1e6,
                             "run_fit_se_ppm": P[k]["fit_se_relative"] * 1e6,
                             "run_vs_expected_ppm": (run / expect[k] - 1) * 1e6,
                             "gauss_vs_expected_ppm": (gq / expect[k] - 1) * 1e6}
out["gauss_IERS_reference_over_GR"] = G_["IERS_reference"]["over_GR_analytic"]

# ---- Mentat-evaluated derived numbers ----
X = {}
X["GR_analytic_arcsec_cy"] = mx("GR_analytic_arcsec_cy", f"{dwGR}*{ARC}*{NCY}")
X["hook_over_design_budget"] = mx("hook_over_design_budget", f"{sig_arc}/0.05")
X["D_over_design_budget"] = mx("D_over_design_budget", f"{D_arc}/0.05")
X["hook_over_will2014_bound"] = mx("hook_over_will2014_bound", f"{rC}/(8e-5/3)")
X["D_over_will2014_bound"] = mx("D_over_will2014_bound", f"{rD}/(8e-5/3)")
X["hook_over_will2006_bound"] = mx("hook_over_will2006_bound", f"{rC}/(3e-3/3)")
X["hook_beta_minus_1_equiv"] = mx("hook_beta_minus_1_equiv", f"-3*{rC}")
X["hook_z_park2017"] = mx("hook_z_park2017", f"Abs(-3*{rC} - (-2.7e-5))/3.9e-5")
X["hook_z_will2014"] = mx("hook_z_will2014", f"Abs(-3*{rC} - (-4.1e-5))/7.8e-5")
X["D_z_park2017"] = mx("D_z_park2017", f"Abs(-3*{rD} - (-2.7e-5))/3.9e-5")
P_anom = R[("A", 600)]["anomalistic_period_s"]
X["orbits_per_century_simulated"] = mx("orbits_per_century_simulated", f"36525*86400/{P_anom}")
X["period_catalogue_over_simulated_minus_1"] = mx("period_catalogue_over_simulated_minus_1",
                                                  f"87.969*86400/{P_anom} - 1")
X["hook_arcsec_cy_simulated_period"] = mx("hook_arcsec_cy_simulated_period",
                                          f"{P['C_minus_A']['rad_per_orbit']}*{ARC}*36525*86400/{P_anom}")
X["xi_docstring"] = num("xi_docstring_check")
X["hook_arcsec_cy_if_xi_docstring"] = mx("hook_arcsec_cy_if_xi_docstring", f"{sig_arc}*{X['xi_docstring']}/{XI}")
X["hook_over_budget_if_xi_docstring"] = mx("hook_over_budget_if_xi_docstring", f"{X['hook_arcsec_cy_if_xi_docstring']}/0.05")
out["derived"] = X
out["mentat_calls_in_aggregate"] = MS
out["probes"] = PR
out["anomalistic_period_s"] = {f"{m}_{dt}": R[(m, dt)]["anomalistic_period_s"] for (m, dt) in R}
out["wall_s"] = {f"{m}_{dt}": R[(m, dt)]["wall_s"] for (m, dt) in R}
out["n_steps"] = {f"{m}_{dt}": R[(m, dt)]["n_steps"] for (m, dt) in R}
out["fit_residual_rms_rad"] = {f"{m}_{dt}": R[(m, dt)]["lrl"][2] for (m, dt) in R}
out["determinism_vs_v1_runs"] = det

FP = PR["force_probe"]; ST = PR["static_pn"]
TM = out["two_methods"]
LV = out["lrl_vs_interp_rel_diff"]
c = out["criteria"]

# ---------------- code-generated text ----------------
out["headline"] = (
    f"With σ = ξ r_s/r imposed by the U5 harness on the Sun at every step (a long-range field; the engine itself only "
    f"carries a constant per-body σ), nbody's e^σ GM scaling advances Mercury's perihelion by {rC:.7f} ± {seC:.1e} of "
    f"GR. That matches ξ/3 to {abs(dev_xi3_rel):.1e} relative ({abs(dev_xi3_se):.2f} SE), and an orbit-averaged Gauss "
    f"quadrature on the engine forces gives {G_['sigma_field_eq']['over_GR_analytic']:.8f}. The advance is "
    f"{sig_arc:.3f}″/cy: {X['hook_over_design_budget']:.0f}× the design's conservative 0.05″/cy budget and "
    f"~{X['hook_over_will2014_bound']:.0f}× Will 2014's β bound, so this coupling is excluded. That result was [DERIVED] "
    f"from the fixed ξ before the run; the engine run confirms it. The design's own engine criterion fails (ratio "
    f"{ratio_design:+.4f}) because the engine's gr_1pn term has the wrong overall sign: the 1PN run precesses at "
    f"{P['B_minus_A']['arcsec_per_cy']:.2f}″/cy ({rB:.7f} × GR). eih_cross runs at {rE:.4f} × GR (velocity term "
    f"flipped). The 2PN static term's sign is correct.")

out["could_it_fail"] = (
    "The Mercury exclusion itself could not fail in this run. ξ = 0.1582 is a fixed repo constant, and Binet's ξ/3 × "
    "42.98″/cy ≈ 2.3″/cy was written into the design before the run. The exclusion is therefore [DERIVED] (Binet "
    "series and ξ/3 by Mentat) against [LIT] bounds, and the 50-orbit run is [VERIFIED numerics] confirmation. The run "
    "could only have changed the verdict by showing that the engine fails to apply its own e^σ scaling, which would be "
    "an engine defect, not the hook surviving. Three things could genuinely fail. (i) The engine reproducing ξ/3: a "
    "wrong derivation (the designs' earlier 2ξ/3), a hook on the wrong body, or σ lagging inside the Forest–Ruth step "
    f"would each have shown. It landed {abs(dev_xi3_rel):.1e} relative ({abs(dev_xi3_se):.2f} fit SE) from ξ/3, and the "
    f"integrator-free Gauss quadrature on the same engine forces gives ξ/3 to "
    f"{abs(TM['C_minus_A']['gauss_vs_expected_ppm']) * 1e-6:.1e}. (ii) The two 1PN sign predictions, pre-registered "
    f"from code reading in PREREG.txt (−1 and −5/3 × GR). Both came out ({rB:.7f}, {rE:.7f}), and the off-perihelion "
    f"force probe fixes each term's sign separately. (iii) The dt-halving control ({c['P4_dt_control_rel_change']:.1e}). "
    "The design's engine criterion did fail, as predicted. ξ/3 itself is [BY CONSTRUCTION] for any e^σ-scaled force "
    "with σ ∝ 1/r.")

out["evidential_weight"] = (
    "Against one specific object: nbody's e^σ scaling of the source's GM, driven by a harness-imposed σ = ξ r_s/r_Mercury "
    "with no range cutoff (λ → ∞). The engine has no σ(r) of its own; it carries a constant per-body sigma_field "
    "(nbody.py:199, 233), and this prescription is undefined for N > 2 bodies. As a long-range coupling it is excluded "
    f"[DERIVED + LIT, confirmed by engine]: {X['hook_over_design_budget']:.1f}× the design's conservative 0.05″/cy budget, "
    f"~{X['hook_over_will2014_bound']:.0f}× Will 2014's β−1 < 8e-5, and ~{X['hook_z_park2017']:.0f} σ from Park+2017's "
    f"β. The ξ r_s/2r amplitude, applied through the same source-GM coupling, is excluded at "
    f"{X['D_over_design_budget']:.0f}× (~{X['D_over_will2014_bound']:.0f}× Will 2014). This is an exclusion by an "
    "independent observable, but with the same ansatz and the same ξ as the program's S9a. It is much weaker than S9a's "
    "10^5 (clocks) to 10^11 (free fall), and comparable to the §14 PPN battery (Cassini ×10^4, lunar ranging ×10^3.6). "
    "Under the Captain's ξ bet the program already limits σ's range to below ~35 km (§6), where this hook is zero at "
    "Mercury. It adds nothing for the unification thesis: any 1/r³ force of this amplitude fails the same way. The "
    "significant side result is for the repo. Both nbody 1PN paths run Mercury's perihelion backwards: gr_1pn at −1 × GR, "
    "with both terms flipped, and eih_cross at −5/3 × GR, with the velocity term flipped. So every include_gr=True, "
    "gr_1pn or eih_cross result from this engine carries the wrong relativistic correction, and no test in the suite can "
    "see it.")

out["design_corrections"] = [
    "The design's engine criterion divides by the engine's own GR run (B−A). That run is retrograde "
    f"({rB:.7f} × GR) because of the gr_1pn sign defect, so the design ratio is {ratio_design:.7f} ± "
    f"{P['design_ratio_se']:.1e} and the criterion FAILS for a reason unrelated to the σ hook. The corrected criterion, "
    "pre-registered in PREREG.txt, measures (C−A) against the analytic GR advance 6πGM/(c²a(1−e²)) evaluated by Mentat. "
    f"Result: {rC:.7f}, PASS. Against the original criterion: FAIL.",
    "The design sketch's perihelion estimator (atan2 of the position one step after the discrete radial minimum) has "
    f"{R[('C', 600)]['sketch'][2]:.1e} rad rms sampling jitter per orbit (fit residual, run C), four orders above the σ "
    f"signal of {P['C_minus_A']['rad_per_orbit']:.1e} rad/orbit. As written it "
    f"returns a ratio of {out['estimators']['sketch']['design_ratio_C_over_B']:.3f} and C/GR = "
    f"{out['estimators']['sketch']['C_minus_A']['over_GR_analytic']:.2f}, which is noise. Before running it was replaced "
    "with the osculating Laplace–Runge–Lenz direction at the radial minimum (fit residual rms "
    f"{out['fit_residual_rms_rad']['C_600']:.1e} rad). The parabola-interpolated angle agrees to "
    f"{abs(LV['C_minus_A']):.1e} relative for C−A ({abs(LV['B_minus_A']):.1e} for B−A, {abs(LV['D_minus_A']):.1e} for "
    "D−A).",
    "The repo has two σ amplitudes. The field equation gives ξ r_s/r, so ξ/3; inventory sigma_from_potential gives "
    "ξ GM/(rc²) = ξ r_s/2r, so ξ/6. Both were run through nbody's source-GM e^σ coupling, and both are excluded in that "
    "coupling. Run D is NOT a test of the inventory's own coupling (σ scaling 0.99 of the TEST body's mass). Under that "
    "coupling, a pure static gradient force −∇(f m c² σ) with constant inertial mass is a 1/r² force that only "
    "renormalises GM, with no first-order perihelion advance [DERIVED]. If σ also changes the inertial mass (a σ-dependent "
    "mass in the point-particle action), velocity-dependent terms of 1PN order appear, of the form f ξ GM/(c²r²)(r̂·v)v "
    "[DERIVED, structural, not evaluated], and Mercury would be sensitive. Neither variant is tested here; the program's "
    "S9a battery addresses that convention.",
    f"The design predicted '~2 orders' of exclusion. Against the design's own 0.05″/cy budget it is "
    f"{math.log10(X['hook_over_design_budget']):.2f} orders ({X['hook_over_design_budget']:.1f}×). That budget is "
    "conservative: it is about 0.1 % of 42.98″/cy, the radar-era accuracy that Will 2006 quotes. Against Will 2014's "
    f"β−1 < 8e-5 the hook is ~{X['hook_over_will2014_bound']:.0f}× over the allowed {num('will2014_allowed_arcsec_cy'):.2e}″/cy. "
    f"Against Park+2017 it is {X['hook_z_park2017']:.0f} σ, where the 1σ is {num('park2017_1sigma_arcsec_cy'):.2e}″/cy.",
    "Framing: the σ(r) = ξ r_s/r prescription belongs to the U5 harness (u5_run.py sets Sun.sigma_field every step from "
    "Mercury's distance), not to the engine. The excluded object is that harness-driven, long-range e^σ GM hook.",
    "The design treated the physics verdict as something the run could decide. It could not: with ξ fixed, the "
    "exclusion follows from Binet's ξ/3 before any run. The run tests the engine, not the hook's survival.",
]

out["coupling_convention_record"] = (
    "nbody.CelestialBody.gm_m3_s2 (nbody.py:225-233) multiplies the SOURCE body's whole GM by e^σ. With σ set from the "
    "test body's distance (by the harness), this is a universal, attractive 1/r³ addition, and it is not reciprocal: "
    "the Sun does not feel a scaled Mercury. inventory.core.sigma scales only the QCD part (0.99) of the TEST body's mass. "
    "sigma_from_potential uses ξ GM/(rc²) = ξ r_s/2r, while the field equation gives ξ r_s/r. Three conventions, two "
    "amplitudes. If σ were the Sun's own (constant) value, the hook would be a constant GM rescaling with no precession.")

out["engine_defects"] = [
    {"file": "D:/Aaron/development/sigma-ground/sigma_ground/field/interface/nbody.py", "line": 476,
     "defect": "The gr_1pn branch (lines 472-477) has the overall wrong sign. It adds +GM/(r²c²)[(4GM/r − v²) r̂ + "
               "4(r̂·v)v] with r̂ = r_ij/r, and r_ij = x_source − x_body is defined at line 406. That is the negative of "
               "IERS Conventions eq. 10.12, GM/(c²r³)[(4GM/r − v²)r + 4(r·v)v], where r runs from the central body to "
               "the orbiter. Both the radial and the velocity term are flipped.",
     "evidence": (f"Force probe fitted at true anomalies 0, π/2, 2 and 4 rad, against the IERS pieces: radial coefficient "
                  f"{FP['gr_1pn']['alpha_radial_coef_vs_IERS']:.10f}, velocity coefficient "
                  f"{FP['gr_1pn']['beta_velocity_coef_vs_IERS']:.10f} (fit residual "
                  f"{FP['gr_1pn']['fit_rel_residual']:.1e}). At perihelion x·v = 0, so v1's perihelion-only probe "
                  f"(cos = −1.000) tested only the radial term. At f = π/2, where x·v/(rv) = "
                  f"{FP['gr_1pn']['per_anomaly'][1]['xv_over_rv']:.3f}, |engine + IERS|/|IERS| = "
                  f"{FP['gr_1pn']['per_anomaly'][1]['rel_residual_vs_minus_IERS']:.1e}. Static probe: the engine coefficient "
                  f"is {ST['gr_1pn'][1]['coefficient']:.6f} × x·GM/r², against +4 from harmonic Schwarzschild (Mentat "
                  f"series −1 + 4x − 9x²). Gauss quadrature on the engine force gives "
                  f"{G_['gr_1pn']['over_GR_analytic']:.7f} × GR. The 50-orbit run gives B−A = "
                  f"{P['B_minus_A']['arcsec_per_cy']:.3f}″/cy = {rB:.7f} × GR, against +42.98″/cy [LIT, Will 2014]. "
                  "This was predicted from code reading in PREREG.txt before the run. Suite blindness: "
                  "test_physics_stability.py:215, test_mercury_precesses_with_gr_on, asserts only delta_au > 0 (line 237). "
                  "test_nbody.py:1587 compares |a_eih|/|a_1pn| to 1 within 1e-4 (line 1617), about 1e4 coarser than the "
                  "~1e-8 1PN term. Callers that enable it: rolling_shootout.py:245-247, scripts/run_long_horizon_sweep.py:267, "
                  "scripts/run_toggle_iteration.py:39, test_future_predictions.py:222, test_probe_propagation.py:177 and 183, "
                  "test_shootout.py:245, and test_physics_stability.py:71, 225 and 454. Not patched (Mentat flags, never "
                  "fixes).")},
    {"file": "D:/Aaron/development/sigma-ground/sigma_ground/field/interface/nbody.py", "line": 458,
     "defect": "In the eih_cross branch, the velocity cross term (r_ij·(4v_i − 3v_j))(v_i − v_j) at lines 457-459 uses "
               "r_ij = x_j − x_i (defined at line 406). The EIH/Moyer form uses x_i − x_j there. The radial A_ij terms "
               "(lines 446-455) carry the correct sign.",
     "evidence": (f"Force probe against the IERS pieces: radial coefficient "
                  f"{FP['eih_cross']['alpha_radial_coef_vs_IERS']:.7f}, velocity coefficient "
                  f"{FP['eih_cross']['beta_velocity_coef_vs_IERS']:.7f}. At f = π/2, cos(engine, IERS) = "
                  f"{FP['eih_cross']['per_anomaly'][1]['cos_engine_vs_IERS']:.3f}, and the residual against IERS with the "
                  f"cross term flipped is {FP['eih_cross']['per_anomaly'][1]['rel_residual_vs_IERS_cross_flipped']:.1e}. "
                  f"Static radial coefficient {ST['eih_cross'][1]['coefficient']:.6f} (correct, +4). Gauss quadrature "
                  f"gives {G_['eih_cross']['over_GR_analytic']:.7f} × GR. The 50-orbit run gives E−A = "
                  f"{P['E_minus_A']['arcsec_per_cy']:.3f}″/cy = {rE:.7f} × GR, matching the pre-registered bookkeeping "
                  "(πm/p)(−A − 2B + 2D) with D = −4 (Mentat: −5/3). A correct EIH gives +1. The docstring calls this "
                  "branch the 'canonical force model used by JPL DE440'.")},
    {"file": "D:/Aaron/development/sigma-ground/sigma_ground/inventory/core/sigma.py", "line": 40,
     "defect": "Documentation and provenance only: the XI docstring (and line 25) attributes ξ = 0.1582 to Planck 2018 "
               "Ω_b h² = 0.02237, Ω_c h² = 0.1200, but those numbers give ξ = 0.1571.",
     "evidence": (f"Mentat simplify_expr 0.02237/(0.02237+0.1200) = {X['xi_docstring']:.6f}. With that ξ, the hook "
                  f"would be {X['hook_arcsec_cy_if_xi_docstring']:.3f}″/cy instead of {sig_arc:.3f}"
                  f"″/cy ({X['hook_over_budget_if_xi_docstring']:.1f}× the budget), so no conclusion changes. "
                  "misc/UNIFICATION_PROGRAM.md:317 already carries both values ('0.1571 / 0.1582'). Not patched.")},
]

out["resolved_not_defective"] = (
    f"The gr_2pn leading term (nbody.py:504-509) has the correct sign. Static probe: engine coefficient "
    f"{ST['gr_2pn'][0]['coefficient']:.6f}, {ST['gr_2pn'][1]['coefficient']:.6f} and "
    f"{ST['gr_2pn'][2]['coefficient']:.6f} × x²·GM/r² at x = 1e-4, 1e-3 and 1e-2 (outward positive). Harmonic-gauge "
    "static Schwarzschild gives a_r/(GM/r²) = −1 + 4x − 9x² (Mentat differentiate_expr, simplify_expr and "
    "series_expansion from g00 = −(r−m)/(r+m), g_rr = (r+m)/(r−m)). The +4 reproduces IERS 2(β+γ) = 4, which checks the "
    "gauge, and the −9 is inward like the engine's term. Only the static piece was tested; the docstring says the "
    "velocity-dependent 2PN terms are absent.")

out["lit_refs"] = [
    "NASA NSSDC Mercury Fact Sheet, nssdc.gsfc.nasa.gov/planetary/factsheet/mercuryfact.html: e = 0.2056, a = 57.909e6 km, "
    "sidereal period 87.969 d (re-fetched in the v2 repair).",
    "Will 2014, Living Rev. Relativ. 17, 4, arXiv:1403.7377, §4.2: eq. 65, 42.98″/cy × [(2+2γ−β)/3 + 3e-4 J2/1e-7]; "
    "MESSENGER-era fits with Cassini γ give β−1 = (−4.1 ± 7.8)e-5; Table 4 lists β−1 < 8e-5 from the perihelion shift. "
    "Read from the arXiv PDF in the v2 repair.",
    "Will 2006, Living Rev. Relativ. 9, 3, arXiv:gr-qc/0510072, text after eq. 50: Mercury's excess perihelion shift known "
    "to about 0.1 % from radar observations (1966–1990), giving |2γ − β − 1| < 3e-3. This, not Will 2014, is the source "
    "of the radar statement v1 attributed to Will 2014. Read from the arXiv PDF in the v2 repair.",
    "Park, Folkner, Konopliv, Williams, Smith, Zuber 2017, AJ 153, 121, doi:10.3847/1538-3881/aa5be2: β−1 = "
    "(−2.7 ± 3.9)e-5, J2 = (2.25 ± 0.09)e-7, total precession 575.3100 ± 0.0015″/cy. Verified by the v1 runner (MIT "
    "DSpace) and two skeptics; the v2 repair could not re-fetch it (publisher bot-check, ADS 405, search budget "
    "exhausted).",
    "U. Hugentobler 2008, 'Orbit Perturbations due to Relativistic Corrections' (citing IERS Conventions 2003): "
    "Schwarzschild term GM/(c²r³){[2(β+γ)GM/r − γv²]r + 2(1+γ)(r·v)v} with r from the central body, a constant "
    "outward radial acceleration for a circular orbit, and the Gauss equation for ω̇ (eq. 6) used in u5_probe_v2.py. "
    "Text read from the PDF in lit_pdf/.",
]

out["caveats"] = [
    "Mercury's eccentricity 0.2056 is typed as [LIT] (NSSDC, re-fetched) because Mentat's solar_system_body has no "
    "eccentricity field. The ratio ξ/3 does not depend on e; only the ″/cy values do.",
    f"Mentat's M_sun of 1.98892e30 kg (labelled IAU 2015 nominal) is ~2.6e-4 above GM_sun/G, so the analytic GR comes "
    f"out {X['GR_analytic_arcsec_cy']:.3f}″/cy rather than 42.98. The ″/cy conversion uses the catalogue sidereal period "
    f"of 87.969 d. The simulated orbit's anomalistic period is {P_anom / 86400:.4f} d, which is "
    f"{X['period_catalogue_over_simulated_minus_1']:.1e} different. Converted with the simulated period, the hook is "
    f"{X['hook_arcsec_cy_simulated_period']:.4f}″/cy instead of {sig_arc:.4f}″/cy. No ratio is affected.",
    f"Precision: the C ratio's fit SE is {seC:.1e} absolute ({P['C_minus_A']['fit_se_relative']:.1e} relative), so the "
    f"ξ/3 deviation of {dev_xi3_rel:.1e} is {dev_xi3_se:.2f} SE. Run vs Gauss quadrature differs by "
    f"{TM['B_minus_A']['run_over_gauss_minus_1_ppm']:+.1f} ppm (B), {TM['C_minus_A']['run_over_gauss_minus_1_ppm']:+.1f} "
    f"(C), {TM['D_minus_A']['run_over_gauss_minus_1_ppm']:+.1f} (D), {TM['E_minus_A']['run_over_gauss_minus_1_ppm']:+.1f} "
    f"(E) and {TM['C_minus_A_dt300']['run_over_gauss_minus_1_ppm']:+.1f} (C at dt 300). That is consistent with the "
    f"{c['P4_dt_control_rel_change'] * 1e6:+.1f} ppm dt-halving shift: a few-ppm integration and estimator systematic, "
    "immaterial to every conclusion.",
    "σ is set on the Sun once per Forest–Ruth step, from the start-of-step distance. The dt-halving control shows this is "
    "irrelevant at the ppm level.",
    "With σ on, momentum is not conserved: Mercury feels the Sun's GM × e^σ, but the Sun feels Mercury's unscaled GM. "
    "This is negligible at a mass ratio of 1.7e-7, and it is a property of the hook.",
    "The Park+2017 and Will 2014 comparisons treat the hook as a pure PPN β shift of −ξ [DERIVED: an extra inward "
    "2ξ(GM)²/(c²r³)]. Park's β bound assumes Cassini's γ and fits J2 jointly, so the σ figures are indicative. The "
    "headline factor uses the design's conservative 0.05″/cy budget.",
    "This is a two-body Sun–Mercury run with no planetary perturbations, no J2 and no 2PN. The comparison is "
    "differential (C−A), so those terms do not enter the ratio.",
    "The harmonic radius r = R − m used for the 2PN sign check is the standard textbook identification. It was not "
    "web-verified in this repair (search budget exhausted); its 1PN coefficient reproduces the verified IERS 2(β+γ) = 4.",
    "No engine code was changed. The 1PN sign defects need a Captain decision, a fix, and a quantitative perihelion "
    "test (for example B−A = +42.98 ± 0.1″/cy with the LRL estimator in u5_run.py).",
]

out["skeptic_responses"] = [
    "ACCEPTED (skeptic 1): SE labelling. The fit SE is absolute in GR-ratio units, and the deviation from ξ/3 is now "
    "stated in relative terms and in SE.",
    "ACCEPTED (skeptic 1): the LRL-vs-interp agreement is now computed by code (v1 said 3e-7).",
    "ACCEPTED (skeptics 1, 2, 3): the Will 2014 citation is fixed, and the radar statement is moved to Will 2006 (both "
    "read from the arXiv PDFs).",
    "ACCEPTED (skeptics 1, 3): the gr_1pn evidence now includes an off-perihelion probe. Both terms are fitted "
    "separately, at four anomalies.",
    "ACCEPTED (skeptic 1): the eih_cross line reference now points to r_ij at line 406 and the cross term at lines "
    "457-459.",
    "ACCEPTED (skeptic 1): provenance. The v2 aggregator writes every text block, and v1's hand additions are stated.",
    "ACCEPTED (skeptics 1, 3): suite-blindness lines (test_nbody.py:1587/1617) and the affected callers are listed; the "
    "period mismatch is quantified.",
    "ACCEPTED (skeptic 2): headline wording, '0.05″/cy' is now 'the design's conservative budget'.",
    "ACCEPTED (skeptic 2): the 2PN caveat is resolved by a static engine probe plus Mentat's harmonic expansion; the "
    "sign is correct.",
    "ACCEPTED (skeptic 2): XI provenance. It is cited as a repo constant, and the docstring mismatch is flagged.",
    "ACCEPTED (skeptic 3): could_it_fail is rewritten. The exclusion is [DERIVED] and was known before the run.",
    "ACCEPTED (skeptic 3): framing. The harness-imposed, long-range σ(r) is named in the headline and evidential_weight.",
    "ACCEPTED (skeptic 3): 'fourth independent exclusion' becomes 'independent observable, same ansatz and ξ, much "
    "weaker than S9a'.",
    "PARTLY ACCEPTED (skeptic 3, run D): run D is qualified as a test of the source-GM coupling only. Not adopted "
    "unconditionally: 'Mercury does not test the inventory convention'. That holds for a pure static gradient force with "
    "constant inertial mass, but if σ also changes the inertial mass, 1PN-order velocity terms appear. Both variants "
    "are now stated, and neither is tested here.",
    "REJECTED (skeptic 3): 'the 2.3e-6 deviation is ~10× the fit SE'. The fit SE in GR-ratio units is "
    f"{seC:.1e} absolute = {P['C_minus_A']['fit_se_relative']:.1e} relative, so the deviation is "
    f"{dev_xi3_se:.2f} SE (skeptic 1's reading). The run-vs-Gauss differences, a few ppm, match the dt-control shift, "
    "and that is how they are now described.",
]
json.dump(out, open(os.path.join(D, "u5_results_v2.json"), "w"), indent=1, ensure_ascii=False)

# ---------------- figure ----------------
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"
plt.rcParams.update({"axes.unicode_minus": True, "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.5, 5.8), facecolor=SURF,
                                    gridspec_kw={"width_ratios": [1.0, 1.0, 0.85]})
M = lambda t: t.replace('-', '\u2212')
mn = lambda v, f: format(v, f).replace('-', '\u2212')


def cum(m, dt=600):
    y = np.unwrap(np.array(R[(m, dt)]["peri_lrl"])) - np.unwrap(np.array(R[("A", dt)]["peri_lrl"]))
    return np.arange(1, len(y) + 1), (y - y[0]) * ARC


k = np.arange(1, 51)
for ax in (ax1, ax2, ax3):
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

x, y = cum("C")
ax1.plot(x, y, "o", ms=4, color=C1, label=f"C: σ = ξ r_s/r (field equation), {sig_arc:.3f}\u2033/cy")
ax1.plot(k, (k - 1) * dwS_pred * ARC, "-", color=C1, lw=1, alpha=0.7, label="ξ/3 of GR (prediction)")
x, y = cum("D")
ax1.plot(x, y, "s", ms=4, color=C2, label=f"D: σ = ξ r_s/2r (sigma_from_potential), {D_arc:.3f}\u2033/cy")
ax1.plot(k, (k - 1) * dwS_pred / 2 * ARC, "-", color=C2, lw=1, alpha=0.7, label="ξ/6 of GR (prediction)")
ax1.plot(k, (k - 1) * 0.05 / NCY, "--", color=INK2, lw=1.2, label="design's conservative budget, 0.05\u2033/cy")
ax1.plot(k, (k - 1) * num("will2014_allowed_arcsec_cy") / NCY, ":", color=INK, lw=1.4,
         label=f"Will 2014 β bound, {num('will2014_allowed_arcsec_cy'):.1e}\u2033/cy".replace("e-0", "e\u22120"))
ax1.set_xlabel("orbit number")
ax1.set_ylabel("perihelion shift vs Newtonian run (arcsec)")
ax1.set_title("(a) harness-imposed σ hook: points measured, lines predicted", fontsize=10, color=INK, loc="left")
ax1.legend(frameon=False, fontsize=8.0, loc="upper left")

x, y = cum("B")
ax2.plot(x, y, "o", ms=4, color=C3, label=M(f"B: engine gr_1pn, {P['B_minus_A']['arcsec_per_cy']:.2f}\u2033/cy"))
x, y = cum("E")
ax2.plot(x, y, "^", ms=4, color=C4, label=M(f"E: engine eih_cross, {P['E_minus_A']['arcsec_per_cy']:.2f}\u2033/cy"))
ax2.plot(k, (k - 1) * dwGR * ARC, "-", color=INK, lw=1.5,
         label=f"GR 6πGM/c²a(1\u2212e²): +{X['GR_analytic_arcsec_cy']:.2f}\u2033/cy (what Mercury does)")
ax2.plot(k, -(k - 1) * dwGR * ARC, ":", color=C3, lw=1.4, label="\u22121 × GR (pre-registered for gr_1pn)")
ax2.plot(k, -(k - 1) * dwGR * ARC * 5 / 3, ":", color=C4, lw=1.4, label="\u22125/3 × GR (pre-registered for eih_cross)")
ax2.set_xlabel("orbit number")
ax2.set_ylabel("perihelion shift vs Newtonian run (arcsec)")
ax2.set_title("(b) the engine's own 1PN terms turn the perihelion backwards", fontsize=10, color=INK, loc="left")
ax2.legend(frameon=False, fontsize=8.0, loc="lower left")

names = ["B_minus_A", "C_minus_A", "C_minus_A_dt300", "D_minus_A", "E_minus_A"]
labels = ["B gr_1pn\n(−1)", "C σ=ξr_s/r\n(ξ/3)", "C dt 300 s\n(ξ/3)", "D σ=ξr_s/2r\n(ξ/6)", "E eih_cross\n(−5/3)"]
xx = np.arange(len(names))
ax3.axhline(0, color=INK2, lw=1)
ax3.errorbar(xx - 0.1, [TM[n]["run_vs_expected_ppm"] for n in names], yerr=[TM[n]["run_fit_se_ppm"] for n in names],
             fmt="o", ms=6, color=C1, ecolor=C1, capsize=3, label="50-orbit engine run, LRL estimator (±1 fit SE)")
ax3.plot(xx + 0.1, [TM[n]["gauss_vs_expected_ppm"] for n in names], "D", ms=6, color=C2,
         label="Gauss quadrature on the engine forces (no integrator)")
ax3.set_xticks(xx); ax3.set_xticklabels([M(s) for s in labels], fontsize=8.5)
ax3.set_ylabel("(measured ÷ code-read prediction) \u2212 1   [ppm]")
ax3.set_title("(c) two methods hit the pre-registered ratios to a few ppm", fontsize=10, color=INK, loc="left")
ax3.legend(frameon=False, fontsize=8.0, loc="lower left")
ax3.set_ylim(-9, 9)

fig.suptitle(
    f"A harness-imposed long-range σ = ξ r_s/r scaling the Sun's GM advances Mercury's perihelion by ξ/3 of GR "
    f"({rC:.5f}), {sig_arc:.2f}″/cy: {X['hook_over_design_budget']:.0f}× the design's conservative 0.05″/cy "
    f"budget and ~{X['hook_over_will2014_bound']:.0f}× Will 2014's β bound, so it is excluded.\n"
    f"The engine's own 1PN terms run backwards (gr_1pn {mn(rB, '.4f')} × GR, eih_cross {mn(rE, '.4f')} × GR), so the "
    f"design ratio against its GR run is {mn(ratio_design, '+.4f')} and that criterion fails.",
    fontsize=11, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.915))
fig.savefig(os.path.join(D, "u5_mercury_sigma_hook_v2.png"), dpi=150, facecolor=SURF)

print(json.dumps({"headline": out["headline"], "criteria": out["criteria"], "derived": X,
                  "two_methods": TM, "lrl_vs_interp": LV, "determinism": det,
                  "gauss_IERS_ref": out["gauss_IERS_reference_over_GR"]}, indent=1, ensure_ascii=False))
