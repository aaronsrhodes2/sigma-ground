"""N1 survey v2 -- nuclear and degenerate matter in Mentat, repaired after three skeptic reviews.
READ-ONLY on the repo and on SkippyTel; writes only into this folder (n1_survey_v2.json).

Every physical constant comes from Mentat (PLATINUM 21) with its source recorded; repo engines imported directly
are Mentat too. Literature values are tagged [LIT] with the citation in n1_references_v2.json (verified by direct
fetch of the arXiv / Crossref record). Section 12 is scratch verification numerics (scipy TOV on the [LIT]
Haensel-Potekhin 2004 SLy fit) because Mentat has no TOV: it is labelled as such and is NOT a Mentat engine result.
v2 changes against v1 are listed in out["v2_changes"].
Run from the repo root:  python -X utf8 <this file>
"""
import json, math, os, re, sys
from pathlib import Path
import numpy as np

REPO = Path(r"D:/Aaron/development/sigma-ground")
SKIPPY = Path(r"D:/Aaron/development/SkippyTel")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "tools")); sys.path.insert(0, str(REPO))
from mentat_client import mentat, mentat_raw  # noqa: E402
import sympy as sp  # only to evaluate expression strings that Mentat returned

out = {"item": "N1", "version": "v2", "kind": "survey for the nuclear/degenerate-matter implementation plan",
       "mentat_sources": {}}
src = out["mentat_sources"]


def M(tool, args, key):
    v, s = mentat(tool, args)
    src[key] = {"tool": tool, "args": args, "value": v, "source": s}
    return v


def grep_line(path, pattern):
    """1-based line number of the first line matching pattern (cited lines are computed, not typed)."""
    for i, l in enumerate(Path(path).read_text(encoding="utf8").splitlines(), 1):
        if re.search(pattern, l):
            return i
    return None


# ---------------------------------------------------------------- 1. leaf constants (Mentat)
C = {}
for name in ["G", "c", "hbar", "k_B", "m_e", "m_p", "m_n", "M_sun", "electron_volt", "AMU_KG",
             "N0_FM3", "E_SAT_MEV", "K_SAT_MEV", "J_SYM_MEV", "R0_FM", "A_C_MEV", "LAMBDA_QCD_MEV",
             "muon mass energy equivalent in MeV", "neutron mass energy equivalent in MeV",
             "proton mass energy equivalent in MeV", "atomic mass constant energy equivalent in MeV"]:
    C[name] = M("lookup_constant", {"name": name}, f"const:{name}")
out["constants"] = C
from sigma_ground.field import constants as FC
out["constants_note"] = {
    "m_e_lookup_scipy": C["m_e"], "M_ELECTRON_KG_field": FC.M_ELECTRON_KG,
    "rel_diff": (C["m_e"] - FC.M_ELECTRON_KG) / C["m_e"],
    "note": "N0_FM3, E_SAT_MEV, K_SAT_MEV, J_SYM_MEV carry no citation in field/constants.py:399-411; L_sym absent."}
G, c, MSUN = C["G"], C["c"], C["M_sun"]

# ---------------------------------------------------------------- 2. Chandrasekhar functions certified symbolically by Mentat
sym = {}
sym["P_integral"] = M("integrate_expr", {"expression": "t**4/sqrt(1+t**2)", "variable": "t", "lower": "0", "upper": "x"}, "sym:P_integral")
sym["eps_integral"] = M("integrate_expr", {"expression": "t**2*sqrt(1+t**2)", "variable": "t", "lower": "0", "upper": "x"}, "sym:eps_integral")
sym["trace_integrand_simplified"] = M("simplify_expr", {"expression": "t**2*sqrt(1+t**2) - t**4/sqrt(1+t**2)"}, "sym:trace_integrand")
sym["trace_integral"] = M("integrate_expr", {"expression": "t**2/sqrt(1+t**2)", "variable": "t", "lower": "0", "upper": "x"}, "sym:trace_integral")
sym["P_NR_series"] = M("series_expansion", {"expression": "x**3*sqrt(x**2 + 1)/4 - 3*x*sqrt(x**2 + 1)/8 + 3*asinh(x)/8", "variable": "x", "point": "0", "order": 8}, "sym:P_NR_series")
sym["P_UR_limit_over_x4"] = M("compute_limit", {"expression": "(x**3*sqrt(x**2 + 1)/4 - 3*x*sqrt(x**2 + 1)/8 + 3*asinh(x)/8)/x**4", "variable": "x", "point": "oo", "direction": "-"}, "sym:P_UR_limit")
sym["threeP_over_eps_UR_limit"] = M("compute_limit", {"expression": "(x**3*sqrt(x**2 + 1)/4 - 3*x*sqrt(x**2 + 1)/8 + 3*asinh(x)/8)/(x**3*sqrt(x**2+1)/4 + x*sqrt(x**2+1)/8 - asinh(x)/8)", "variable": "x", "point": "oo", "direction": "-"}, "sym:3P_over_eps_UR")
sym["d_trace_dx"] = M("differentiate_expr", {"expression": "x*sqrt(x**2+1)/2 - asinh(x)/2", "variable": "x"}, "sym:d_trace")
out["chandrasekhar_functions_symbolic"] = sym
_x = sp.Symbol("x")
P_int = sp.lambdify(_x, sp.sympify(sym["P_integral"]), "math")      # Int_0^x t^4/sqrt(1+t^2) dt (Mentat's closed form)
E_int = sp.lambdify(_x, sp.sympify(sym["eps_integral"]), "math")    # Int_0^x t^2 sqrt(1+t^2) dt

# ---------------------------------------------------------------- 2b. Mentat solve_ode: what is refused, what works (v2: re-attributed)
le = {}
for eq in ["y'' + y", "x*y'' + 2*y' + x*y", "x*Derivative(y(x),x,2) + 2*Derivative(y(x),x) + x*y(x)"]:
    r = mentat_raw("solve_ode", {"equation": eq})
    le[eq] = r.get("result", {}).get("value") if r.get("ok") else ("REFUSED: " + str(r.get("error"))[:160])
    src[f"solve_ode:{eq}"] = {"tool": "solve_ode", "args": {"equation": eq}, "value": le[eq],
                             "source": r.get("result", {}).get("source") if r.get("ok") else "SkippyTel/mentat front door (_UNSAFE)"}
APP = SKIPPY / "mentat/app.py"; MATHX = REPO / "sigma_ground/mcp/tools/mathx.py"
out["solve_ode_probe"] = {
    "results": le,
    "front_door_regex_line": grep_line(APP, r"^_UNSAFE = re\.compile"),
    "front_door_refusal_line": grep_line(APP, r"looks like code, not maths"),
    "docstring_example_line": grep_line(MATHX, r"Example: \"y''"),
    "solve_ode_def_line": grep_line(MATHX, r"^def solve_ode"),
    "reading": "The apostrophe (prime) syntax that solve_ode documents is refused by the SkippyTel front-door guard, "
               "including the docstring's own example. With Derivative() syntax Mentat solves Lane-Emden n = 1 symbolically "
               "(spherical Bessel j0 = sin x/x up to constants). Mentat still has no NUMERIC initial-value tool."}

# ---------------------------------------------------------------- 3. Chandrasekhar-mass test target [DERIVED] + numeric omega_3 check
OMEGA3_LIT = 2.01824
pref = (math.sqrt(3 * math.pi) / 2) * OMEGA3_LIT * (C["hbar"] * c / G) ** 1.5
mch = {}
for label, mB in [("m_u (AMU_KG)", C["AMU_KG"]), ("m_p + m_e (hydrogen atom, unbound)", C["m_p"] + C["m_e"]), ("m_n", C["m_n"])]:
    mch[label] = {"M_Ch_over_Msun_mu_e_2": pref / (2 * mB) ** 2 / MSUN}
_v = [d["M_Ch_over_Msun_mu_e_2"] for d in mch.values()]
from scipy.integrate import solve_ivp  # verification numerics only


def lane_emden(n):
    def f(xi, y):
        th, dth = y
        return [dth, -max(th, 0.0) ** n - 2 * dth / xi]
    ev = lambda xi, y: y[0]; ev.terminal = True; ev.direction = -1
    x0 = 1e-6
    s = solve_ivp(f, (x0, 20), [1 - x0 ** 2 / 6, -x0 / 3], method="DOP853", rtol=1e-12, atol=1e-14, events=ev)
    xi1 = s.t_events[0][0]; dth1 = s.y_events[0][0][1]
    return xi1, xi1 ** 2 * abs(dth1)


xi3, om3 = lane_emden(3); xi1, _ = lane_emden(1)
out["chandrasekhar_mass_target"] = {"omega_3_LIT": OMEGA3_LIT, "omega_3_numeric_scratch": om3, "xi1_n3_numeric": xi3,
    "xi1_n1_minus_pi": xi1 - math.pi, "by_baryon_mass_convention": mch,
    "convention_spread_pct": 100 * (max(_v) - min(_v)) / min(_v)}

# ---------------------------------------------------------------- 4. SEMF implementations vs AME2020 (10 named nuclides, as v1)
iso = json.load(open(REPO / "sigma_ground/inventory/data/isotopes.json", encoding="utf8"))
tab = {(x["Z"], x["A"]): x for x in iso}
from sigma_ground.field.interface import element as EL
ELPY = REPO / "sigma_ground/field/interface/element.py"
L_ASYM = grep_line(ELPY, r"B -= _SEMF_AA \* \(A - 2\*Z\)\*\*2 / \(4\.0 \* A\)")
L_COEFF = grep_line(ELPY, r"coeff = _SEMF_AC / \(2\.0 \* _SEMF_AA\)")
L_BE_DEF = grep_line(ELPY, r"^def _binding_energy_MeV")
L_SMN_DEF = grep_line(ELPY, r"^def stable_mass_number")
nuclides = [(2, 4), (6, 12), (8, 16), (20, 40), (26, 56), (28, 62), (50, 120), (79, 197), (82, 208), (92, 238)]
rows = []
for Z, A in nuclides:
    rec = tab[(Z, A)]
    be_ame = rec["binding_energy_per_nucleon_kev"] * A / 1000.0
    semf_mcp = M("nuclear_binding_energy", {"protons": Z, "neutrons": A - Z}, f"nbe_semf:{Z}-{A}")["binding_energy_MeV"]
    exact_mcp = M("nuclear_binding_energy", {"protons": Z, "neutrons": A - Z, "measured_mass_u": rec["atomic_mass_u"]},
                  f"nbe_exact:{Z}-{A}")["binding_energy_MeV"]
    semf_el = EL._binding_energy_MeV(Z, A)
    semf_el_consistent = semf_el + EL._SEMF_AA * (A - 2 * Z) ** 2 / (4.0 * A) - EL._SEMF_AA * (A - 2 * Z) ** 2 / A
    rows.append(dict(Z=Z, A=A, BE_AME2020_MeV=be_ame, BE_mcp_exact_MeV=exact_mcp, BE_mcp_SEMF_MeV=semf_mcp,
                     BE_element_py_SEMF_MeV=semf_el, BE_element_py_asym_consistent_MeV=semf_el_consistent,
                     resid_mcp_SEMF_pct=100 * (semf_mcp - be_ame) / be_ame,
                     resid_element_py_pct=100 * (semf_el - be_ame) / be_ame,
                     resid_element_py_consistent_pct=100 * (semf_el_consistent - be_ame) / be_ame))


def rms(v):
    v = list(v); return math.sqrt(sum(x * x for x in v) / len(v))


out["semf_comparison"] = {"rows": rows, "element_py_lines": {"_binding_energy_MeV_def": L_BE_DEF, "asymmetry_term": L_ASYM,
                                                            "stable_mass_number_def": L_SMN_DEF, "stable_mass_number_coeff": L_COEFF},
    "rms_pct_A_ge_12_ten_set": {k: rms(r[k] for r in rows if r["A"] >= 12)
                                for k in ["resid_mcp_SEMF_pct", "resid_element_py_pct", "resid_element_py_consistent_pct"]}}

# ---------------------------------------------------------------- 4b. the same check on the whole isotope table (v2)
from scipy.stats import spearmanr, pearsonr
full = []
for x in iso:
    Z, A = x["Z"], x["A"]
    if A < 12 or not x.get("binding_energy_per_nucleon_kev"):
        continue
    be = x["binding_energy_per_nucleon_kev"] * A / 1000.0
    asis = EL._binding_energy_MeV(Z, A)
    cons = asis + EL._SEMF_AA * (A - 2 * Z) ** 2 / (4.0 * A) - EL._SEMF_AA * (A - 2 * Z) ** 2 / A
    full.append(dict(Z=Z, A=A, NmZ=A - 2 * Z, BE=be, r_asis=100 * (asis - be) / be, r_cons=100 * (cons - be) / be))
f12 = full; f16 = [x for x in full if x["A"] >= 16]
ks = np.linspace(0.5, 6.0, 1101); best = None
for k in ks:
    rr = []
    for x in f12:
        Z, A = x["Z"], x["A"]
        b = EL._binding_energy_MeV(Z, A) + EL._SEMF_AA * (A - 2 * Z) ** 2 / (4.0 * A) - EL._SEMF_AA * (A - 2 * Z) ** 2 / (k * A)
        rr.append(100 * (b - x["BE"]) / x["BE"])
    v = rms(rr)
    if best is None or v < best[1]:
        best = (float(k), v)
aC = EL._SEMF_AC; X, yv = [], []
for x in f16:
    Z, A = x["Z"], x["A"]; N = A - Z
    pair = (1 if (Z % 2 == 0 and N % 2 == 0) else (-1 if (Z % 2 == 1 and N % 2 == 1) else 0)) / math.sqrt(A)
    X.append([A, -A ** (2 / 3), -(N - Z) ** 2 / A, pair]); yv.append(x["BE"] + aC * Z * (Z - 1) / A ** (1 / 3))
X, yv = np.array(X), np.array(yv)
coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
pred = X @ coef - np.array([aC * x["Z"] * (x["Z"] - 1) / x["A"] ** (1 / 3) for x in f16])
bes = np.array([x["BE"] for x in f16])
out["semf_full_table"] = {
    "n_A_ge_12": len(f12), "n_A_ge_16": len(f16),
    "rms_pct_A_ge_12": {"as_written": rms(x["r_asis"] for x in f12), "consistent": rms(x["r_cons"] for x in f12)},
    "rms_pct_A_ge_16": {"as_written": rms(x["r_asis"] for x in f16), "consistent": rms(x["r_cons"] for x in f16)},
    "spearman_resid_as_written_vs_NmZ_A12": float(spearmanr([x["r_asis"] for x in f12], [x["NmZ"] for x in f12])[0]),
    "pearson_resid_as_written_vs_NmZ2_over_A_A16": float(pearsonr([x["r_asis"] for x in f16], [x["NmZ"] ** 2 / x["A"] for x in f16])[0]),
    "best_divisor_k_A12": {"k": best[0], "rms_pct": best[1], "code_uses": 4.0},
    "refit_4param_A16": {"a_V": float(coef[0]), "a_S": float(coef[1]), "a_A": float(coef[2]), "a_P": float(coef[3]), "a_C_fixed": aC,
                         "rms_rel_pct": rms((100 * (pred - bes) / bes).tolist()),
                         "rms_abs_MeV": float(np.sqrt(np.mean((pred - bes) ** 2)))},
    "points": [{k: x[k] for k in ["Z", "A", "NmZ", "r_asis", "r_cons"]} for x in f12],
    "tag": "[MEASURED] repo engine element.py vs inventory/data/isotopes.json (AME2020); refit is scratch numpy lstsq"}

_pb = [r for r in rows if (r["Z"], r["A"]) == (82, 208)][0]; _u = [r for r in rows if (r["Z"], r["A"]) == (92, 238)][0]

# ---------------------------------------------------------------- 5. observational anchors -> compactness C = GM/(R c^2) (Mentat r_s)
anchors = {  # (M, R_km, R_minus, R_plus, M_minus, M_plus, status); intervals are the papers' 68 % credible intervals [LIT]
    "J0030+0451 Riley+2019": (1.34, 12.71, 1.19, 1.14, 0.16, 0.15, "revisited by Vinciguerra+2024 (2019 solutions disfavoured)"),
    "J0030+0451 Miller+2019": (1.44, 13.02, 1.06, 1.24, 0.14, 0.15, "revisited by Vinciguerra+2024 (2019 solutions disfavoured)"),
    "J0030+0451 Vinciguerra+2024 mode A (approx)": (1.4, 11.5, None, None, None, None, "abstract gives ~values only, model-dependent"),
    "J0030+0451 Vinciguerra+2024 mode B (approx)": (1.7, 14.5, None, None, None, None, "abstract gives ~values only, model-dependent"),
    "J0740+6620 Riley+2021": (2.072, 12.39, 0.98, 1.30, 0.066, 0.067, "superseded by Salmi+2024 (3.6 yr data)"),
    "J0740+6620 Salmi+2024": (2.073, 12.49, 0.88, 1.28, 0.069, 0.069, "current"),
    "J0437-4715 Choudhury+2024": (1.418, 11.36, 0.63, 0.95, 0.037, 0.037, "current"),
}
comp = {}
for k, (m, R, rm, rp, mm, mp, st) in anchors.items():
    rs = M("schwarzschild_radius", {"mass_kg": m * MSUN}, f"rs:{k}")
    d = {"M_Msun_LIT": m, "R_km_LIT": R, "R_68": None if rm is None else [R - rm, R + rp],
         "M_68": None if mm is None else [m - mm, m + mp], "status": st, "r_s_m": rs, "C_point": rs / (2 * R * 1e3)}
    if rm is not None:
        d["C_over_R_68"] = [rs / (2 * (R + rp) * 1e3), rs / (2 * (R - rm) * 1e3)]
    comp[k] = d
out["nicer_compactness"] = {"rows": comp, "definition": "C = GM/(R c^2) = r_s/(2R), as Podkowka-Mendes-Poisson define it",
    "trace_flip_compactness_LIT": {"C": 0.262, "plus": 0.011, "minus": 0.017,
        "what_it_is": "median and 90 % spread over 20,000 four-parameter piecewise polytropes (Read+2009 form) with the "
                      "low-density part fixed to the DH2001 (SLy) fit; a model-ensemble spread, NOT a measurement",
        "ref": "Podkowka, Mendes & Poisson 2018, PRD 98, 064057"}}

# ---------------------------------------------------------------- 6. pressure share: P6 rows, GW170817, Chandrasekhar EOS (v2)
p6 = {"Sun core": (1.5e5, 2.5e16), "white dwarf centre": (1e9, 1e22), "neutron star centre (~2 n0)": (5e17, 1e34)}
out["p6_rows_3P_over_rho_c2"] = {k: {"rho_kg_m3_LIT": r, "P_Pa_LIT": P, "3P_over_rho_c2": 3 * P / (r * c ** 2),
                                     "tag": "[DERIVED from LIT rows in scripts/unification/pilot_p6_sigma_bridge.py:48-50]"}
                                 for k, (r, P) in p6.items()}
gw = {}
for lab, val in [("central", 3.5e34), ("low_90", 3.5e34 - 1.7e34), ("high_90", 3.5e34 + 2.7e34)]:
    gw[lab] = M("convert_units", {"value": val, "from_units": "dyn/cm**2", "to_units": "Pa"}, f"conv:gw170817_{lab}")
out["gw170817_P_2nsat_Pa"] = gw
rho_nuc = M("convert_units", {"value": 2.8e14, "from_units": "g/cm**3", "to_units": "kg/m**3"}, "conv:abbott_rho_nuc")
dens = {"2 rho_nuc (Abbott 2018 convention, rho_nuc = 2.8e14 g/cm3) [LIT]": 2 * rho_nuc,
        "2 n_sat (repo N0_FM3 x m_n)": 2 * C["N0_FM3"] * 1e45 * C["m_n"],
        "2 n_sat (Margueron 2018 n_sat = 0.155 fm^-3 x m_n)": 2 * 0.155e45 * C["m_n"]}
out["gw170817_pressure_share"] = {k: {"rho_kg_m3": r, "3P_over_rho_c2": {lab: 3 * P / (r * c ** 2) for lab, P in gw.items()}}
                                  for k, r in dens.items()}
r_ns, P_ns = p6["neutron star centre (~2 n0)"]
out["p6_ns_row_check"] = {"n_over_n_sat_repo": r_ns / (C["m_n"] * C["N0_FM3"] * 1e45),
                          "P_row_over_GW170817_90pct_upper_at_2nsat": P_ns / gw["high_90"],
                          "reading": "P6's neutron-star row sits below 2 n_sat yet exceeds GW170817's 90 % upper bound AT 2 n_sat; "
                                     "since P rises with density it is excluded by the plan's own anchor. 0.67 is [DERIVED from LIT] "
                                     "of an excluded row and is withdrawn as a headline number."}
PU = C["m_e"] ** 4 * c ** 5 / (math.pi ** 2 * C["hbar"] ** 3)   # Pa, unit of the Chandrasekhar functions


def P_chandra(rho, mu_e=2.0):
    n_e = rho / (mu_e * C["AMU_KG"])
    x = C["hbar"] * (3 * math.pi ** 2 * n_e) ** (1 / 3) / (C["m_e"] * c)
    return PU * P_int(x) / 3.0, x


Pw, xw = P_chandra(1e9)
wd_range = {f"{r:.0e}": 3 * P_chandra(r)[0] / (r * c ** 2) for r in [1e8, 1e9, 1e10, 1e11, 1e12, 1e13]}
out["white_dwarf_pressure_share"] = {"P_chandra_at_1e9_Pa": Pw, "x_at_1e9": xw, "P6_row_over_chandra": 1e22 / Pw,
                                     "3P_over_rho_c2_chandra_by_rho": wd_range,
                                     "tag": "[DERIVED] Mentat integrate_expr closed form + Mentat hbar, c, m_e, AMU_KG; mu_e = 2"}

# ---------------------------------------------------------------- 7. saturation anchors (v2: Margueron 2018 Tables 1-2, experiment-based)
marg = {"n_sat_fm3": (0.155, 0.005), "E_sat_MeV": (-15.8, 0.3), "K_sat_MeV": (230.0, 20.0), "E_sym_MeV": (32.0, 2.0),
        "L_sym_MeV": (60.0, 15.0), "K_sym_MeV": (-100.0, 100.0), "Q_sat_MeV": (300.0, 400.0)}
repo = {"n_sat_fm3": C["N0_FM3"], "E_sat_MeV": C["E_SAT_MEV"], "K_sat_MeV": C["K_SAT_MEV"], "E_sym_MeV": C["J_SYM_MEV"]}
out["saturation_anchors"] = {
    "Margueron2018_Tables1_2_present_estimation_LIT": {k: {"value": v, "sigma": s} for k, (v, s) in marg.items()},
    "Margueron2018_Table4_50_model_average_LIT (model spread, not data)": {"K_sat_MeV": (251, 29), "L_sym_MeV": (76.6, 29.2)},
    "Lattimer2023_LIT": {"J_MeV": (32.0, 1.1), "L_MeV": (51.9, 7.9)},
    "repo_vs_Margueron_pulls_sigma": {k: (repo[k] - marg[k][0]) / marg[k][1] for k in repo}}

# ---------------------------------------------------------------- 8. short-range sigma indicator (unchanged)
from sigma_ground.inventory.core.sigma import XI
rho_sat = C["N0_FM3"] * 1e45 * C["m_n"]
sig = {f"lambda_{l}km": 8 * math.pi * G * XI * (l * 1e3) ** 2 * rho_sat / c ** 2 for l in (0.3, 1.0, 3.0)}
out["short_range_sigma_indicator"] = {"xi_repo": XI, "rho_sat_kg_m3": rho_sat, "sigma_at_rho_sat": sig,
    "note": "[DERIVED, SPECULATIVE reading] section 6 local formula, valid only for lambda << R"}

# ---------------------------------------------------------------- 9. P7 factor: flat vs Tolman normalisation (v2: analytic, Mentat integrals)
i_sin2 = M("integrate_expr", {"expression": "sin(x)**2", "variable": "x", "lower": "0", "upper": "pi"}, "sym:int_sin2")
i_xsin = M("integrate_expr", {"expression": "x*sin(x)", "variable": "x", "lower": "0", "upper": "pi"}, "sym:int_xsin")
r_flat = float(sp.N(sp.sympify(i_sin2) / sp.sympify(i_xsin)))
out["p7_normalisation"] = {"int_0^pi sin^2": i_sin2, "int_0^pi x sin x": i_xsin,
    "intP_over_int_rho_c2_in_units_of_Pc_over_rhoc_c2": r_flat, "flat_ratio": "1 - w_c/2", "tolman_ratio_first_order": "1 - w_c",
    "reading": "[DERIVED] For an n = 1 polytrope (rho ~ theta, P ~ theta^2, theta = sin xi/xi) 3 int P dV / int rho c^2 dV = "
               "w_c * (pi/2)/pi = w_c/2, so P7's flat ratio is 1 - w_c/2 (normalised to the rest mass). Normalised to the Tolman "
               "mass (eps + 3P) the deficit is 6 int P / int eps = w_c at first order. The two differ by normalisation, not by "
               "a correction to P7, so the M8 small-w_c slope test is bug-only."}
try:
    tsp = json.load(open(REPO / "tov_sigma_pair_results.json", encoding="utf8"))
    out["prior_scratch_tov_sigma_pair"] = {"status": "untracked parallel-session file, read not rerun",
        "benchmark_keys": [k for k in tsp if k.startswith("benchmark")],
        "rows": [{k: r.get(k) for k in ["w_c", "Q_sigma_over_Q_Tolman_exact", "pred_1PN_1_minus_wc",
                                         "P7_as_written_1_minus_wc_over_2"]} for r in tsp["rows"]]}
except Exception as e:  # pragma: no cover
    out["prior_scratch_tov_sigma_pair"] = {"error": repr(e)}

# ---------------------------------------------------------------- 10. static survey of the repo (v2: integrators and EOS layers added)
TERMS = ["Fermi", "degenerate", "TOV", "Tolman", "polytrope", "Lane.Emden", "binding_energy", "nuclear", "neutron_star",
         "white_dwarf", "equation_of_state", "Chandrasekhar", "semi.empirical", "Weizs", "bag.model", "symmetry_energy",
         "saturation", r"solve_ivp|odeint", r"\brk4\b|runge.kutta|dopri|dormand", r"\beos\b"]
DIRS = ["sigma_ground/field", "sigma_ground/inventory", "sigma_ground/mcp", "sigma_ground/dynamics", "sigma_ground/kernel", "scripts"]
hits = {}
for t in TERMS:
    rx = re.compile(t, re.I); files = []
    for d in DIRS:
        for p in (REPO / d).rglob("*.py"):
            try:
                if rx.search(p.read_text(encoding="utf8", errors="ignore")):
                    files.append(str(p.relative_to(REPO)).replace("\\", "/"))
            except OSError:
                pass
    hits[t] = sorted(files)
out["grep_hits"] = {t: {"count": len(v), "files": v} for t, v in hits.items()}


def defs(path):
    txt = (REPO / path).read_text(encoding="utf8")
    return [f"{i+1}: {l.strip()}" for i, l in enumerate(txt.splitlines()) if re.match(r"^(def|class) \w+", l)]


np_top = []
for d in ["sigma_ground/field", "sigma_ground/dynamics", "sigma_ground/kernel"]:
    for p in (REPO / d).rglob("*.py"):
        if p.name.startswith("test_"):
            continue
        if re.search(r"^(import numpy|from numpy)", p.read_text(encoding="utf8", errors="ignore"), re.M):
            np_top.append(str(p.relative_to(REPO)).replace("\\", "/"))
out["existing_integrators_and_eos"] = {
    "sigma_ground/field/interface/adapters/integrator.py": {"defs": defs("sigma_ground/field/interface/adapters/integrator.py"),
        "what": "sg_solve_ivp / sg_odeint / sg_nbody: FIXED-step Forest-Ruth, Verlet and RK4, numpy-based, aimed at Hamiltonian "
                "systems; no adaptive error control, no dense output, no terminal events (a TOV surface needs all three)"},
    "sigma_ground/dynamics/stepper.py": {"defs": defs("sigma_ground/dynamics/stepper.py"),
        "what": "scene leapfrog sub-stepping with a CFL step for the SPH/continuum engine; not a general ODE solver"},
    "sigma_ground/dynamics/fluid/eos.py": {"defs": defs("sigma_ground/dynamics/fluid/eos.py"),
        "what": "SPH pressure functions of rho (Tait, ideal gas, liquid sound speed); no barotropic EOS object, no energy density, "
                "no degenerate or nuclear matter"},
    "sigma_ground/dynamics/continuum.py": {"what": "eos in {'tait', 'isothermal', 'ideal'} selects among the fluid/eos.py functions"},
    "numpy_top_level_imports_non_test_field_dynamics_kernel": {"count": len(np_top), "files": sorted(np_top)},
    "pyproject.toml": "dependencies = [] (line 12), yet the files above import numpy at module level, so 'the core is "
                      "zero-dependency' is true of the declared metadata only"}
out["existing_relevant_code"] = {
    "sigma_ground/field/binding.py": {"defs": defs("sigma_ground/field/binding.py")},
    "sigma_ground/mcp/tools/nuclear.py": {"defs": defs("sigma_ground/mcp/tools/nuclear.py")},
    "sigma_ground/field/nucleon.py": {"defs": defs("sigma_ground/field/nucleon.py")}}

# ---------------------------------------------------------------- 11. bag-model stability window (v2) [DERIVED]
# Massless u,d,s, alpha_s = 0, g = 6 per flavour: P = N_f mu^4/(4 pi^2 (hbar c)^3) - B. At P = 0: eps = 4B and
# E/A = k * B^(1/4). 3 flavours (equal mu): k3 = 12 pi^2 / (3^(1/4) (4 pi^2)^(3/4)).
# 2 flavours, charge-neutral without electrons (n_d = 2 n_u): k2 = (4 pi^2)^(1/4) (1 + 2^(4/3))^(3/4).
# (pi*pi, not pi**2: the front door's power-tower guard _HUGE refuses "(4*pi**2)**(3/4)" as a false positive.)
k3s = M("simplify_expr", {"expression": "12*pi*pi/(3**(1/4)*(4*pi*pi)**(3/4))"}, "sym:bag_k3")
k2s = M("simplify_expr", {"expression": "(4*pi*pi)**(1/4)*(1 + 2**(4/3))**(3/4)"}, "sym:bag_k2")
k3, k2 = float(sp.N(sp.sympify(k3s))), float(sp.N(sp.sympify(k2s)))
EA_fe = tab[(26, 56)]["atomic_mass_u"] * C["atomic mass constant energy equivalent in MeV"] / 56
out["bag_window"] = {"k3_expr": k3s, "k3": k3, "k2_expr": k2s, "k2": k2, "E_per_A_56Fe_atom_MeV": EA_fe,
                     "B14_lower_MeV (2-flavour must be unbound)": EA_fe / k2, "B14_upper_MeV (3-flavour bound)": EA_fe / k3,
                     "tag": "[DERIVED] Mentat simplify_expr + Mentat atomic mass unit + AME2020 56Fe mass (isotopes.json)"}

# ---------------------------------------------------------------- 12. what the plan's reference EOS (SLy) will score (scratch, v2)
# [LIT] Haensel & Potekhin 2004, A&A 428, 191, eq. (14) and Table 2 (SLy). zeta = log10 P[dyn/cm2], xi = log10 rho[g/cm3],
# rho = total mass-energy density / c^2. scipy TOV with Mentat G, c, M_sun: verification numerics, NOT a Mentat engine.
# Adapted from the skeptic's verify_skeptic/skeptic_check.py (same fit, same integrator settings) and extended.
a = [None, 6.22, 6.121, 0.005925, 0.16326, 6.48, 11.4971, 19.105, 0.8938, 6.54, 11.4950, -22.775, 1.5707,
     4.3, 14.08, 27.80, -1.653, 1.50, 14.67]
f0 = lambda x: 1.0 / (np.exp(x) + 1.0)


def zeta(xi):
    return ((a[1] + a[2] * xi + a[3] * xi ** 3) / (1 + a[4] * xi) * f0(a[5] * (xi - a[6]))
            + (a[7] + a[8] * xi) * f0(a[9] * (a[10] - xi)) + (a[11] + a[12] * xi) * f0(a[13] * (a[14] - xi))
            + (a[15] + a[16] * xi) * f0(a[17] * (a[18] - xi)))


xi_grid = np.linspace(5.0, 16.3, 20001); z_grid = zeta(xi_grid)
assert np.all(np.diff(z_grid) > 0)
lnP_grid = np.log(0.1) + z_grid * np.log(10); lneps_grid = np.log(1e3 * c ** 2) + xi_grid * np.log(10)
P_surf = 0.1 * 10 ** zeta(6.0)


def tov_generic(Pc, eps_of_P, Psurf):
    def rhs(r, y):
        P, m = y
        if P <= Psurf:
            return [0.0, 0.0]
        e = eps_of_P(P)
        dP = -G * (e + P) * (m + 4 * math.pi * r ** 3 * P / c ** 2) / (c ** 2 * r ** 2 * (1 - 2 * G * m / (r * c ** 2)))
        return [dP, 4 * math.pi * r ** 2 * e / c ** 2]
    ev = lambda r, y: y[0] - Psurf; ev.terminal, ev.direction = True, -1
    r0 = 1.0; ec = eps_of_P(Pc)
    s = solve_ivp(rhs, (r0, 1e6), [Pc, 4 / 3 * math.pi * r0 ** 3 * ec / c ** 2], events=ev, rtol=1e-8,
                  atol=[1e-6 * Psurf, 1e-6], max_step=50.0)
    R = s.t_events[0][0] if len(s.t_events[0]) else s.t[-1]
    Mk = s.y_events[0][0][1] if len(s.t_events[0]) else s.y[1, -1]
    return R / 1e3, Mk / MSUN, 3 * Pc / ec


sly_eps = lambda P: math.exp(np.interp(math.log(P), lnP_grid, lneps_grid))
GM_c2 = G * MSUN / c ** 2
seq = []
for lx in np.linspace(14.2, 15.6, 141):
    R, Mm, w = tov_generic(0.1 * 10 ** zeta(lx), sly_eps, P_surf)
    seq.append(dict(log10_rho_c=float(lx), R_km=R, M_Msun=Mm, w_c=w, C=GM_c2 * Mm / (R * 1e3)))
i_max = int(np.argmax([s_["M_Msun"] for s_ in seq])); stable = seq[:i_max + 1]
Ms = np.array([s_["M_Msun"] for s_ in stable]); Rs = np.array([s_["R_km"] for s_ in stable]); Ws = np.array([s_["w_c"] for s_ in stable])
R_at = lambda m: float(np.interp(m, Ms, Rs)) if Ms[0] <= m <= Ms[-1] else None
W_at = lambda m: float(np.interp(m, Ms, Ws)) if Ms[0] <= m <= Ms[-1] else None
flip = None
for s0, s1 in zip(stable[:-1], stable[1:]):
    if s0["w_c"] < 1 <= s1["w_c"]:
        t = (1 - s0["w_c"]) / (s1["w_c"] - s0["w_c"])
        flip = {"C": s0["C"] + t * (s1["C"] - s0["C"]), "M_Msun": s0["M_Msun"] + t * (s1["M_Msun"] - s0["M_Msun"])}


def score(mlo, mhi, rlo, rhi):
    """EOS-vs-data score over the measured 68 % mass interval: is any stable-branch R(M) inside the 68 % radius interval?"""
    ms = [m for m in np.linspace(mlo, mhi, 41) if R_at(m) is not None]
    if not ms:
        return {"status": "NOT REACHABLE (M_max below the mass interval)"}
    rr = [R_at(m) for m in ms]
    inside = [rlo <= r <= rhi for r in rr]
    return {"mass_range_reached": [min(ms), max(ms)], "R_range_km": [min(rr), max(rr)],
            "status": "CONSISTENT at 68 %" if any(inside) else ("TENSION: R below 68 % interval" if max(rr) < rlo else "TENSION: R above 68 % interval")}


sc = {k: score(v["M_68"][0], v["M_68"][1], v["R_68"][0], v["R_68"][1]) for k, v in comp.items() if v.get("M_68")}
P_2rho = 0.1 * 10 ** float(zeta(math.log10(2 * rho_nuc / 1e3)))
out["sly_expectation_scratch"] = {
    "tag": "[VERIFIED numerics on a LIT fit] scipy RK45 TOV on the Haensel-Potekhin 2004 SLy fit, Mentat G, c, M_sun; "
           "not a Mentat engine (Mentat has no TOV: that is the gap)",
    "M_max_Msun": seq[i_max]["M_Msun"], "R_at_Mmax_km": seq[i_max]["R_km"], "C_at_Mmax": seq[i_max]["C"], "w_c_at_Mmax": seq[i_max]["w_c"],
    "DH2001_LIT_M_max": 2.05,
    "R_km": {str(m): R_at(m) for m in [1.0, 1.34, 1.4, 1.418, 1.44, 2.0, 2.072]},
    "central_3P_over_eps": {str(m): W_at(m) for m in [1.0, 1.4, 1.8, 2.0]},
    "central_trace_flip": flip,
    "P_at_2rho_nuc_Pa": P_2rho, "P_at_2rho_nuc_note": "HP2004 rho is mass-energy density, Abbott's rho_nuc a mass density: ~3 % offset, noted",
    "scores_68pct": sc,
    "M_max_ge_1p94_J0740_Fonseca2021": bool(seq[i_max]["M_Msun"] >= 1.94),
    "GW170817_P_in_90pct_band": bool(gw["low_90"] <= P_2rho <= gw["high_90"]),
    "PMP_flip_C_in_band": bool((flip is not None) and (0.245 <= flip["C"] <= 0.273)),
    "sequence": seq}

# free neutron gas (Chandrasekhar functions for neutrons, rest mass included): the M5 OV test target (scratch)
PUn = C["m_n"] ** 4 * c ** 5 / (math.pi ** 2 * C["hbar"] ** 3)
xg = np.logspace(-3, 1.5, 6001)
Pn = np.array([PUn * P_int(x) / 3 for x in xg]); En = np.array([PUn * E_int(x) for x in xg])
lnPn, lnEn = np.log(Pn), np.log(En)
fn_eps = lambda P: math.exp(np.interp(math.log(P), lnPn, lnEn))
fseq = []
for x in np.logspace(-0.5, 1.2, 60):
    R, Mm, w = tov_generic(PUn * P_int(x) / 3, fn_eps, Pn[5])
    fseq.append((float(x), R, Mm, w))
im = int(np.argmax([f[2] for f in fseq]))
out["free_neutron_star_scratch"] = {"M_max_Msun": fseq[im][2], "R_at_Mmax_km": fseq[im][1], "x_c": fseq[im][0],
    "w_c_at_Mmax": fseq[im][3], "tag": "[VERIFIED numerics, scratch] OV free-neutron maximum; literature ~0.7 Msun (OV 1939)"}

# ---------------------------------------------------------------- 13. engine and document defects (v2)
out["engine_defects_found"] = [
    {"file": "sigma_ground/field/interface/element.py", "line": L_ASYM,
     "defect": f"SEMF asymmetry term written a_a (A-2Z)^2/(4A) with a_a = {EL._SEMF_AA} MeV, the coefficient of the (A-2Z)^2/A form "
               f"that the same file's stable_mass_number() (coeff = a_c/(2 a_a), line {L_COEFF}) assumes: the penalty is 4x too small",
     "evidence": f"10 named nuclides: Pb-208 {_pb['resid_element_py_pct']:+.2f} % -> {_pb['resid_element_py_consistent_pct']:+.2f} %, "
                 f"U-238 {_u['resid_element_py_pct']:+.2f} % -> {_u['resid_element_py_consistent_pct']:+.2f} %. Whole table, "
                 f"{len(f12)} nuclides A >= 12: rms {out['semf_full_table']['rms_pct_A_ge_12']['as_written']:.2f} % -> "
                 f"{out['semf_full_table']['rms_pct_A_ge_12']['consistent']:.2f} %; best divisor k = {best[0]:.2f} (code 4); "
                 f"Spearman(residual, N-Z) = {out['semf_full_table']['spearman_resid_as_written_vs_NmZ_A12']:.2f}"},
    {"file": "sigma_ground/mcp/tools/nuclear.py", "line": 62,
     "defect": "second, different SEMF coefficient set hardcoded inline, plus private u / m_H / m_n constants (PLATINUM 3 duplicate)",
     "evidence": "15.75/17.8/0.711/23.7/11.18 vs element.py 15.56/17.23/A_C/23.29/12.0; rms 1.14 % (A >= 12), He-4 -19.3 %"},
    {"file": "SkippyTel/mentat/app.py", "line": out["solve_ode_probe"]["front_door_regex_line"],
     "defect": "the front-door _UNSAFE regex refuses any apostrophe, so solve_ode's documented prime syntax (mathx.py:"
               f"{out['solve_ode_probe']['docstring_example_line']}, example \"y'' + y\") can never reach the tool; the guard is a "
               "deliberate eval() defence, so the mismatch is between the guard and the tool's documented interface",
     "evidence": f"Mentat refused \"y'' + y\" and the prime-form Lane-Emden with the error at app.py:{out['solve_ode_probe']['front_door_refusal_line']}; "
                 f"Derivative() syntax returns {le['x*Derivative(y(x),x,2) + 2*Derivative(y(x),x) + x*y(x)']}"},
    {"file": "sigma_ground/field/constants.py", "line": 399,
     "defect": "nuclear-matter anchors N0_FM3, E_SAT_MEV, K_SAT_MEV, J_SYM_MEV carry no citation and no uncertainty; L_sym missing",
     "evidence": "constants.py:395-411; values match Margueron 2018 Table 1 within 1 sigma (n_sat 0.16 vs 0.155 +/- 0.005)"},
    {"file": "sigma_ground/field/audit.py", "line": 266,
     "defect": "audit ledger lists 'TOV integration (full)' and 'M_TOV = 2.071 Msun' as live; the code is only in ../ssbm-theory",
     "evidence": "no TOV integrator anywhere under sigma_ground/"},
    {"file": "sigma_ground/field/planets.py", "line": 138,
     "defect": "NEUTRON_STAR body hand-entered (1.4 Msun, 11 km, four layers with round densities), uncited", "evidence": "planets.py:137-176"},
    {"file": "scripts/unification/pilot_p6_sigma_bridge.py", "line": 50,
     "defect": "P6's [LIT] dense-matter rows are inconsistent with checkable physics: the neutron-star row (5e17 kg/m^3, 1e34 Pa) "
               "exceeds GW170817's 90 % bound, and the white-dwarf row (1e9 kg/m^3, 1e22 Pa) is several times the Chandrasekhar pressure",
     "evidence": f"NS row P / GW170817 upper bound = {out['p6_ns_row_check']['P_row_over_GW170817_90pct_upper_at_2nsat']:.2f}; "
                 f"WD row P / Chandrasekhar P(1e9, mu_e = 2) = {out['white_dwarf_pressure_share']['P6_row_over_chandra']:.2f}"},
    {"file": "GOLDEN_RULES.md", "line": 31,
     "defect": "Rule 2 points to sigma_ground/field/bounds.py, removed in e66615b (SSBM excision); documentation defect",
     "evidence": "ls sigma_ground/field/bounds.py: no such file"}]

out["v2_changes"] = [
    "element.py lines computed by grep (asymmetry term, stable_mass_number coeff) instead of typed",
    "solve_ode probe adds the docstring example and the Derivative() form; refusal attributed to SkippyTel/mentat/app.py",
    "SEMF check extended to the whole isotope table, with a divisor scan and a 4-parameter refit",
    "NICER anchors updated: Vinciguerra+2024 (J0030), Salmi+2024 (J0740), Choudhury+2024 (J0437); compactness intervals",
    "pressure share: GW170817 at Abbott's 2 rho_nuc and at 2 n_sat; Chandrasekhar WD pressures; P6 rows checked and flagged",
    "Margueron 2018 Tables 1-2 (experiment-based) replace the Table 4 model average as the saturation anchors",
    "P7 normalisation made analytic with Mentat integrals",
    "existing integrators and EOS layers surveyed; numpy top-level imports counted",
    "bag-model window derived with Mentat",
    "expected outcome of the plan's validation tests for its own reference EOS (SLy, HP2004 fit, scratch TOV)",
    "free-neutron OV maximum computed (scratch) as the M5 test target"]

refs_path = OUT / "n1_references_v2.json"
out["references_verified"] = json.load(open(refs_path if refs_path.exists() else OUT / "n1_references.json", encoding="utf8"))
json.dump(out, open(OUT / "n1_survey_v2.json", "w", encoding="utf8"), indent=1, ensure_ascii=False, default=float)
show = {k: out[k] for k in ["chandrasekhar_mass_target", "solve_ode_probe", "p6_ns_row_check", "gw170817_pressure_share",
                            "white_dwarf_pressure_share", "saturation_anchors", "p7_normalisation", "bag_window",
                            "free_neutron_star_scratch"]}
print(json.dumps(show, indent=1, ensure_ascii=False, default=float))
print(json.dumps({k: v for k, v in out["semf_full_table"].items() if k != "points"}, indent=1, default=float))
print(json.dumps(out["semf_comparison"]["element_py_lines"]), json.dumps(out["semf_comparison"]["rms_pct_A_ge_12_ten_set"]))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("C_point", "C_over_R_68")} for k, v in comp.items()}, indent=1))
print(json.dumps({k: v for k, v in out["sly_expectation_scratch"].items() if k != "sequence"}, indent=1, default=float))
print(json.dumps(out["existing_integrators_and_eos"]["numpy_top_level_imports_non_test_field_dynamics_kernel"], indent=1))
