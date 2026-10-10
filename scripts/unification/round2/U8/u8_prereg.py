"""U8 pre-registration — written and run BEFORE any engine evaluation.

Records (a) the design's pass/fail verbatim, (b) the predictions derived from
reading the engine code, (c) the design corrections found while reading, and
(d) the numbers behind them, each from Mentat (sympy tools) with its source.
Nothing here touches the SPH/N-body engines.
"""
import sys, json, math, time
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
from mentat_client import mentat

OUT = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U8/u8_prereg.json"
K_H = 1.2            # h = 1.2 * d  (engine default smoothing_length k), d = (m/rho)^(1/3)
EPS_OVER_H = 0.5     # engine default gravity_softening_m = 0.5*h
src = {}

# local log-slope of the Plummer-softened pair acceleration  a ∝ r (r²+s²)^(-3/2)
dv, s1 = mentat("differentiate_expr", {"expression": "log(r) - 3/2*log(r^2 + s^2)", "variable": "r"})
loc, s2 = mentat("simplify_expr", {"expression": f"r*({dv})".replace("**", "^")})
src["plummer_dlnr"] = {"value": dv, "source": s1}
src["plummer_local_slope"] = {"value": loc, "source": s2}

def R_over_d(N):
    v, s = mentat("simplify_expr", {"expression": f"N((3*{N}/(4*pi))^(1/3))"})
    return float(v), s

def secant(r1, r2, s):
    r1, r2, s2 = round(r1, 6), round(r2, 6), round(s * s, 6)
    expr = (f"(log({r2}/({r2}**2+{s2})**1.5) - log({r1}/({r1}**2+{s2})**1.5))"
            f"/log({r2}/{r1})")
    v, so = mentat("simplify_expr", {"expression": expr})
    return float(v), so

pred = {}
for N, lo_h in ((256, 2.0), (2048, 2.0), (2048, 4.0)):
    Rd, sR = R_over_d(N)
    lo = lo_h * K_H; s = EPS_OVER_H * K_H
    sec, ss = secant(lo, Rd, s)
    pred[f"N{N}_window_{lo_h:g}h_to_R"] = {"R_over_d": Rd, "R_over_h": Rd / K_H,
                                           "secant_slope_eps_0.5h": sec,
                                           "sources": [sR, ss]}

# r beyond which |local slope + 2| < 0.05 for s = 0.5 h :  3 s²/(r²+s²) < 0.05
rmin, s3 = mentat("solve_equation", {"equation": "3*s^2/(r^2+s^2) = 1/20", "variable": "r"})
src["r_min_for_0.05_band"] = {"value": rmin, "source": s3}

# (1 - 1/N) finite-N factor of E|U| for iid uniform points: LS slope it adds over N = 32..512
Ns = [32, 64, 128, 256, 512]
x = [math.log(n) for n in Ns]; y = [math.log(1 - 1 / n) for n in Ns]
xm = sum(x) / 5; ym = sum(y) / 5
dslope = sum((a - xm) * (b - ym) for a, b in zip(x, y)) / sum((a - xm) ** 2 for a in x)

prereg = {
  "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "design_pass_fail_verbatim": ("SEPARATE: exponents 1.667 ± 0.03 vs 1.00 ± 0.01 AND gravity response "
      "-2.00 ± 0.05 with pressure response identically 0 beyond 2h. SAME: exponents within 0.1. "
      "Tags: U-scaling [VERIFIED numerics / LIT O2]; cut-off [BY CONSTRUCTION]."),
  "choices": {"N_set": Ns, "seed": 7, "omega": 0.0, "particle_mass_kg": 1.0,
              "cloud_density": "rho_iron from Mentat density('iron'), so the SPH auto-h equals 1.2 d",
              "h": "1.2 d, d = (m/rho)^(1/3) (fixed density => fixed h for every N)",
              "softening": "both eps = 0 (pure Newton) and eps = 0.5 h (engine default), reported separately",
              "c_s": "sqrt(K/rho) for iron, K from Mentat elastic_analysis hydrostatic stress (Tait's dP/drho)",
              "central_parcel": "parcel nearest the barycentre; mass x 1.1; r measured from it",
              "fit_window": "[2h, R] per design (R = seeded cloud radius)",
              "supplementary_declared_now": ["seeds 1..10 at every N (noise of the U slope)",
                                             "N = 2048 range run (parcels beyond 4h exist)",
                                             "short-range control: kernel pair energy sum_i m_i rho_i^SPH",
                                             "Tait EOS pressure response and sum m P/rho (exploratory)",
                                             "virial-balance (Jeans-locus) crossing N* of the two fitted laws"]},
  "predictions": {
    "slope_U_engine_pe_j_seed7": f"5/3 + {dslope:.4f} (finite-N factor 1-1/N) = {5/3+dslope:.4f}, plus single-seed sampling noise of order 0.01-0.02; predicted PASS of 1.667 ± 0.03 (not certain)",
    "slope_U_mentat": "5/3 exactly [BY CONSTRUCTION: (3/5)GM²/R at fixed rho]",
    "slope_E_P": "1.000 exactly [BY CONSTRUCTION]; sum m P_i/rho_i (isothermal, engine) = M c_s² to roundoff",
    "U_engine_over_U_mentat": "(N-1)/N on average (iid uniform points; continuum includes self-pairs)",
    "gravity_response_eps0_N256": "-2.000 to < 1e-9 [BY CONSTRUCTION: dm enters only the pair term G dm/r²]",
    "gravity_response_eps0.5h_N256": f"{pred['N256_window_2h_to_R']['secant_slope_eps_0.5h']:.3f} (secant) -> FAILS -2.00 ± 0.05: softening bias, a design defect not physics",
    "gravity_response_eps0.5h_N2048_window_4h_R": f"{pred['N2048_window_4h_to_R']['secant_slope_eps_0.5h']:.3f} -> inside the band",
    "pressure_response_beyond_2h": "NOT zero: the isothermal/Tait force on i uses P_j/rho_j² of i's neighbours j, and rho_j changes for every j within 2h of the perturbed parcel, so the response reaches 4h (two kernel hops). Predicted FAIL of the design clause.",
    "pressure_response_beyond_4h": "identically 0.0 (bitwise) — the corrected criterion; at N = 256 R/h = 3.28 < 4 so no parcel lies beyond 4h (clause vacuous there); tested at N = 2048",
    "short_range_control_slope": "between 1.0 and 1.25, local slopes falling toward 1 (bulk N + negative surface N^(2/3) term)",
    "tait_sum_mP_over_rho": "exploratory: Poisson sampling inflates interior rho_SPH by m W(0) = rho/(pi k³) = +18 %, so bulk Tait P ≈ +0.18 K; sign/slope not a criterion",
  },
  "design_corrections_known_before_running": [
    "Pressure support is 4h, not 2h: SPH density sum (2h) followed by the symmetric pressure force over neighbours (2h).",
    "The gravity -2.00 ± 0.05 band holds only for eps = 0 or for a window starting beyond ~3.84 h when eps = 0.5 h (local slope (s²-2r²)/(r²+s²), Mentat).",
    "At fixed density with k = 1.2, R/h = 3.28 at N = 256: the window [2h, R] spans only a factor 1.64 in r and contains no parcel beyond 4h.",
    "E_P = M c_s² (isothermal) is extensive by construction; the U slope is the only engine-measured exponent, and it can fail only if the pair sum is not 1/r or the seeding is not at fixed density.",
  ],
  "corrected_pass_fail": ("SEPARATE if slope_U = 1.667 ± 0.03 (seed 7; also the 10-seed mean) AND slope_E_P = 1.00 ± 0.01 "
      "AND gravity response = -2.00 ± 0.05 (eps = 0 over [2h,R] at N = 256; eps = 0.5 h over [4h,R] at N = 2048) "
      "AND pressure response identically 0 beyond 4h (N = 2048) while non-zero somewhere in (2h, 4h). SAME if slopes within 0.1."),
  "mentat_sources": src, "secant_predictions": pred, "finite_N_slope_shift": dslope,
}
json.dump(prereg, open(OUT, "w", encoding="utf8"), indent=1, ensure_ascii=False)
print(json.dumps(prereg["predictions"], indent=1, ensure_ascii=False))
print(json.dumps(pred, indent=1)); print(src)
