"""U3 finalize — fold the post-hoc decomposition, design corrections, defects and references into u3_results.json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "u3_results.json")
R = json.load(open(P, encoding="utf-8"))
D = json.load(open(os.path.join(HERE, "u3_decomposition.json"), encoding="utf-8"))
L = R["mentat_leaves"]
R["decomposition_posthoc"] = dict(
    identity="<K> = -<E_s> + L + <U_s - W_s>, L = (1/2)[Idot]/T  (Lagrange-Jacobi, softened); so d<K>/dE = -1 + dL/dE + d<U_s-W_s>/dE",
    tag="[VERIFIED numerics] identity residual 0.0000 in every window; [BY CONSTRUCTION] the fully corrected slope <K> - L",
    **D)
R["design_corrections"] = [
    "Unit mismatch: the design's t_cross = sqrt(b^3/GM) = %.0f s is %.3f x shorter than the Heggie-Mathieu crossing time "
    "GM^(5/2)/(2|E|)^(3/2) of a Plummer sphere (Mentat simplify_expr (3pi/32)^(-3/2)); t_relax/t_cross = 0.1N/lnN = %.3f "
    "(Mentat) refers to the latter. So 'relax 10 t_cross' is 1.6 t_cr and '3 t_relax = 7.9 t_cross' is 1.27 t_cr = 0.48 "
    "standard t_relax, shorter than one orbital period at the half-mass radius (13.24 design units, Mentat orbital_period) "
    "and shorter in seconds than P5b's run (12 t_ff = 5.8e5 s vs 3.4e5 s)." % (
        R["choices"]["t_cross_design_s"], R["choices"]["t_cross_HM_over_design_unit"], R["choices"]["trelax_over_tcross"]),
    "Seed-std gate (<= 0.05) is not reachable with N = 128, delta <= 0.10 and a 2.6-unit window: the seed scatter is the "
    "Lagrange-Jacobi breathing term (std of dL/dE = 0.224 pre-registered window), a coherent, slowly damped oscillation of "
    "the cluster after the kick, not sampling noise of the virial value. A criterion on the time-averaged slope needs "
    "windows spanning several breathing periods (>~ 3 x 13 design units), or a criterion stated on the identity.",
    "The design's 'LJ-corrected' K_LJ = <K> - (1/4)[Idot]/T equals -<W_s>/2 and removes only HALF of the non-stationarity "
    "term; removing all of it (<K> - L) returns -1 + d<U_s-W_s>/dE by construction (-0.989 +- 0.003), so neither "
    "correction is a test; the raw slope is the test.",
    "Drift metric unspecified in the design: end-point softened-energy drift (pre-registered metric) is 2.8e-5 <= 1e-4 "
    "(pass); the max-over-time excursion is 2.1e-4 (would fail a strict reading).",
    "The gate (2<K>/|<W_s>| over the last 5 units of relax) passes for all seeds (1.020, 1.016, 1.005) but does not "
    "guarantee a virialised injection instant: seed 1 was kicked at instantaneous 2K/|W_s| = 1.094; the delta = 0 control "
    "removes the common relaxation only to first order.",
    "Gas side is still formula only (heat_capacity_equipartition(3, 128) = %.4e J/K); the Tait-box repeat waits for S8." %
    L["C_gas_equipartition_J_per_K"]["value"],
]
R["engine_defects"] = [
    dict(file="D:/Aaron/development/SkippyTel/mentat/app.py", line=185,
         defect="Mentat gateway power-tower guard (_HUGE regex) false-positive: any expression of the form (...x^2)^(p) "
                "is refused as 'looks like code'",
         evidence="integrate_expr {'expression': '3*x^2*(1+x^2)^(-5/2)', 'variable': 'x'} -> error 'That input looks like "
                  "code, not maths'; the regex (\\^|\\*\\*)\\s*\\(?\\s*[\\d.]+\\s*\\)?\\s*(\\^|\\*\\*) matches '^2)^'. "
                  "integrate_expr 'x^2' works. Blocks the Plummer mass integral; not used as a pass/fail input."),
    dict(file="D:/Aaron/development/sigma-ground/sigma_ground/field/interface/nbody.py", line=1123,
         defect="(already filed PHYSICS_GAP, not new) NBodySystem.total_energy() sums the unsoftened 1/r potential",
         evidence="this run used its own softened U_s and W_s; softened energy conserved to 2.8e-5 end-point"),
]
R["lit_refs_verified"] = [
    "Lynden-Bell D. 1999, Physica A 263, 293-304, doi:10.1016/S0378-4371(98)00518-4, arXiv:cond-mat/9812172 (negative specific heat; injection protocol idea)",
    "Heggie D.C. & Mathieu R.D. 1986, LNP 267, 233, doi:10.1007/BFb0116419 (standard N-body units; crossing time 2*sqrt(2) in Henon units per Heggie's lecture notes)",
    "Aarseth S.J., Henon M. & Wielen R. 1974, A&A 37, 183 (1974A&A....37..183A) (Plummer sampling recipe used by the design sketch)",
]
json.dump(R, open(P, "w", encoding="utf-8"), indent=1)
print("ok", sorted(R.keys()))
