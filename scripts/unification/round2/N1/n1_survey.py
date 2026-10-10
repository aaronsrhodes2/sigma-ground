"""N1 survey — nuclear and degenerate matter in Mentat: what exists, what is missing, and the anchor numbers
the implementation plan's tests will target. READ-ONLY on the repo; writes only into this folder.

Every physical constant comes from Mentat (PLATINUM 21) with its source recorded; repo engines imported directly
are Mentat too. Literature values are tagged [LIT] with the verified citation in REFERENCES.
Run from the repo root:  python -X utf8 <this file>
"""
import json, math, os, re, sys
from pathlib import Path

REPO = Path(r"D:/Aaron/development/sigma-ground")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "tools")); sys.path.insert(0, str(REPO))
from mentat_client import mentat, mentat_raw  # noqa: E402

out = {"item": "N1", "kind": "survey for the nuclear/degenerate-matter implementation plan", "mentat_sources": {}}
src = out["mentat_sources"]

def M(tool, args, key):
    v, s = mentat(tool, args)
    src[key] = {"tool": tool, "args": args, "value": v, "source": s}
    return v

# ---------------------------------------------------------------- 1. leaf constants (Mentat)
C = {}
for name in ["G", "c", "hbar", "k_B", "m_e", "m_p", "m_n", "M_sun", "electron_volt", "AMU_KG",
             "N0_FM3", "E_SAT_MEV", "K_SAT_MEV", "J_SYM_MEV", "R0_FM", "A_C_MEV", "LAMBDA_QCD_MEV",
             "muon mass energy equivalent in MeV", "neutron mass energy equivalent in MeV", "proton mass energy equivalent in MeV"]:
    C[name] = M("lookup_constant", {"name": name}, f"const:{name}")
out["constants"] = C
# The repo stores m_e twice: field/constants.py M_ELECTRON_KG (CODATA 2018) vs lookup 'm_e' (scipy CODATA 2022).
from sigma_ground.field import constants as FC
out["constants_note"] = {
    "m_e_lookup_scipy": C["m_e"], "M_ELECTRON_KG_field": FC.M_ELECTRON_KG,
    "rel_diff": (C["m_e"] - FC.M_ELECTRON_KG) / C["m_e"],
    "saturation_constants_cited_in_code": False,
    "note": "N0_FM3, E_SAT_MEV, K_SAT_MEV, J_SYM_MEV carry no citation in field/constants.py:399-411; L_sym absent."}

# ---------------------------------------------------------------- 2. Chandrasekhar functions certified symbolically by Mentat
# T=0 ideal Fermi gas, x = p_F/(m c); in units of m^4 c^5/(pi^2 hbar^3):
#   P   = (1/3) * Int_0^x t^4/sqrt(1+t^2) dt ,  eps = Int_0^x t^2 sqrt(1+t^2) dt ,  n = x^3/(3 pi^2 lambda_C^3)
sym = {}
sym["P_integral"] = M("integrate_expr", {"expression": "t**4/sqrt(1+t**2)", "variable": "t", "lower": "0", "upper": "x"}, "sym:P_integral")
sym["eps_integral"] = M("integrate_expr", {"expression": "t**2*sqrt(1+t**2)", "variable": "t", "lower": "0", "upper": "x"}, "sym:eps_integral")
# trace (eps - 3P) in the same units = Int t^2 sqrt(1+t^2) - Int t^4/sqrt(1+t^2) = Int t^2/sqrt(1+t^2)
sym["trace_integrand_simplified"] = M("simplify_expr", {"expression": "t**2*sqrt(1+t**2) - t**4/sqrt(1+t**2)"}, "sym:trace_integrand")
sym["trace_integral"] = M("integrate_expr", {"expression": "t**2/sqrt(1+t**2)", "variable": "t", "lower": "0", "upper": "x"}, "sym:trace_integral")
# limits: non-relativistic P ~ x^5/15, ultra-relativistic P ~ x^4/12 (units above) ; 3P/eps -> 1 as x -> oo
sym["P_NR_series"] = M("series_expansion", {"expression": "x**3*sqrt(x**2 + 1)/4 - 3*x*sqrt(x**2 + 1)/8 + 3*asinh(x)/8", "variable": "x", "point": "0", "order": 8}, "sym:P_NR_series")
sym["P_UR_limit_over_x4"] = M("compute_limit", {"expression": "(x**3*sqrt(x**2 + 1)/4 - 3*x*sqrt(x**2 + 1)/8 + 3*asinh(x)/8)/x**4", "variable": "x", "point": "oo", "direction": "-"}, "sym:P_UR_limit")
sym["threeP_over_eps_UR_limit"] = M("compute_limit", {"expression": "(x**3*sqrt(x**2 + 1)/4 - 3*x*sqrt(x**2 + 1)/8 + 3*asinh(x)/8)/(x**3*sqrt(x**2+1)/4 + x*sqrt(x**2+1)/8 - asinh(x)/8)", "variable": "x", "point": "oo", "direction": "-"}, "sym:3P_over_eps_UR")
sym["d_trace_dx"] = M("differentiate_expr", {"expression": "x*sqrt(x**2+1)/2 - asinh(x)/2", "variable": "x"}, "sym:d_trace")
out["chandrasekhar_functions_symbolic"] = sym

# Lane-Emden n = 1 closed form via Mentat's symbolic dsolve (Mentat has no numeric IVP tool: a gap the plan fills)
le = {}
for eq in ["x*y'' + 2*y' + x*y", "y'' + 2*y'/x + y"]:
    r = mentat_raw("solve_ode", {"equation": eq})
    le[eq] = r.get("result", {}).get("value") if r.get("ok") else ("ERROR: " + str(r.get("error"))[:160])
    src[f"solve_ode:{eq}"] = {"tool": "solve_ode", "args": {"equation": eq}, "value": le[eq],
                             "source": r.get("result", {}).get("source") if r.get("ok") else None}
out["lane_emden_n1_mentat_dsolve"] = le

# ---------------------------------------------------------------- 3. Chandrasekhar-mass test target [DERIVED]
# M_Ch = (sqrt(3 pi)/2) * omega_3 * (hbar c / G)^(3/2) / (mu_e m_B)^2 ; omega_3 = 2.01824 [LIT: Chandrasekhar 1939/1967,
# Shapiro & Teukolsky 1983 eq. 3.3.12, as reproduced on the Tohline/vistrails structure page]. m_B convention matters.
OMEGA3_LIT = 2.01824
pref = (math.sqrt(3 * math.pi) / 2) * OMEGA3_LIT * (C["hbar"] * C["c"] / C["G"]) ** 1.5
mch = {}
for label, mB in [("m_u (AMU_KG)", C["AMU_KG"]), ("m_p + m_e (hydrogen atom, unbound)", C["m_p"] + C["m_e"]), ("m_n", C["m_n"])]:
    mch[label] = {"M_Ch_over_Msun_mu_e_2": pref / (2 * mB) ** 2 / C["M_sun"],
                  "coefficient_over_mu_e2": pref / mB ** 2 / C["M_sun"]}
_v = [d["M_Ch_over_Msun_mu_e_2"] for d in mch.values()]
out["chandrasekhar_mass_target"] = {"omega_3_LIT": OMEGA3_LIT, "by_baryon_mass_convention": mch,
    "convention_spread_pct": 100 * (max(_v) - min(_v)) / min(_v),
    "note": "[DERIVED] from Mentat hbar, c, G, M_sun, AMU_KG/m_p/m_e/m_n and omega_3 [LIT]. The spread between baryon-mass "
            "conventions is why the module must define mu_e with the atomic mass unit and say so."}

# ---------------------------------------------------------------- 4. existing SEMF implementations vs AME2020 (repo isotope table)
iso = json.load(open(REPO / "sigma_ground/inventory/data/isotopes.json", encoding="utf8"))
tab = {(x["Z"], x["A"]): x for x in iso}
from sigma_ground.field.interface import element as EL
nuclides = [(2, 4), (6, 12), (8, 16), (20, 40), (26, 56), (28, 62), (50, 120), (79, 197), (82, 208), (92, 238)]
rows = []
for Z, A in nuclides:
    rec = tab[(Z, A)]
    be_ame = rec["binding_energy_per_nucleon_kev"] * A / 1000.0           # MeV, AME2020 via isotopes.json
    semf_mcp = M("nuclear_binding_energy", {"protons": Z, "neutrons": A - Z}, f"nbe_semf:{Z}-{A}")["binding_energy_MeV"]
    exact_mcp = M("nuclear_binding_energy", {"protons": Z, "neutrons": A - Z, "measured_mass_u": rec["atomic_mass_u"]},
                  f"nbe_exact:{Z}-{A}")["binding_energy_MeV"]
    semf_el = EL._binding_energy_MeV(Z, A)                                   # repo engine, field/interface/element.py:485
    # the same element.py coefficients with the asymmetry term in the (A-2Z)^2/A convention that its own
    # stable_mass_number() (element.py:530-569, coeff = a_c/(2 a_a)) assumes:
    asym_consistent = EL._SEMF_AA * (A - 2 * Z) ** 2 / A
    semf_el_consistent = semf_el + EL._SEMF_AA * (A - 2 * Z) ** 2 / (4.0 * A) - asym_consistent
    rows.append(dict(Z=Z, A=A, BE_AME2020_MeV=be_ame, BE_mcp_exact_MeV=exact_mcp, BE_mcp_SEMF_MeV=semf_mcp,
                     BE_element_py_SEMF_MeV=semf_el, BE_element_py_asym_consistent_MeV=semf_el_consistent,
                     resid_mcp_SEMF_pct=100 * (semf_mcp - be_ame) / be_ame,
                     resid_element_py_pct=100 * (semf_el - be_ame) / be_ame,
                     resid_element_py_consistent_pct=100 * (semf_el_consistent - be_ame) / be_ame))
out["semf_comparison"] = {"rows": rows,
    "coefficients": {"mcp/tools/nuclear.py:62-66": {"a_v": 15.75, "a_s": 17.8, "a_c": 0.711, "a_a": 23.7, "a_p": 11.18,
                                                     "asym_form": "a_a (N-Z)^2/A"},
                     "field/interface/element.py:111-116": {"a_v": EL._SEMF_AV, "a_s": EL._SEMF_AS, "a_c": EL._SEMF_AC,
                                                            "a_a": EL._SEMF_AA, "a_p": EL._SEMF_AP,
                                                            "asym_form_in_binding": "a_a (A-2Z)^2/(4A)  (element.py:518)",
                                                            "asym_form_in_stable_mass_number": "a_a (A-2Z)^2/A (implied by coeff a_c/(2 a_a), element.py:563)"}},
    "rms_pct": {k: math.sqrt(sum(r[k] ** 2 for r in rows if r["A"] >= 12) / sum(1 for r in rows if r["A"] >= 12))
                for k in ["resid_mcp_SEMF_pct", "resid_element_py_pct", "resid_element_py_consistent_pct"]}}

_pb = [r for r in rows if (r["Z"], r["A"]) == (82, 208)][0]; _u = [r for r in rows if (r["Z"], r["A"]) == (92, 238)][0]
out["engine_defects_found"] = [{
    "file": "sigma_ground/field/interface/element.py", "line": 518,
    "defect": "SEMF asymmetry term written a_a (A-2Z)^2/(4A) with a_a = 23.29 MeV, the coefficient of the (A-2Z)^2/A form that "
              "the same file's stable_mass_number() (line 563, coeff = a_c/(2 a_a)) assumes; the asymmetry penalty is 4x too small",
    "evidence": f"vs AME2020 (isotopes.json): Pb-208 {_pb['resid_element_py_pct']:+.2f} %, U-238 {_u['resid_element_py_pct']:+.2f} %; "
                f"with the (A-2Z)^2/A form: {_pb['resid_element_py_consistent_pct']:+.2f} %, {_u['resid_element_py_consistent_pct']:+.2f} %; "
                f"rms (A>=12) {out['semf_comparison']['rms_pct']['resid_element_py_pct']:.2f} % -> "
                f"{out['semf_comparison']['rms_pct']['resid_element_py_consistent_pct']:.2f} %",
    "status": "flagged, not fixed (Mentat flags, never fixes); fix lands with module M3 (single SEMF) under a regression test"},
   {"file": "sigma_ground/mcp/tools/nuclear.py", "line": 62,
    "defect": "second, different SEMF coefficient set hardcoded inline, plus private u / m_H / m_n constants (PLATINUM 3 duplicate)",
    "evidence": "a_v 15.75 vs 15.56, a_s 17.8 vs 17.23, a_p 11.18 vs 12.0 between the two implementations; He-4 off by -19 %",
    "status": "flagged; M3 replaces both with one fitted, cited implementation"},
   {"file": "sigma_ground/field/constants.py", "line": 399,
    "defect": "nuclear-matter anchors N0_FM3, E_SAT_MEV, K_SAT_MEV, J_SYM_MEV carry no citation and no uncertainty; L_sym missing",
    "evidence": "read of constants.py:395-411; Mentat lookup_constant returns source 'sigma_ground.field.constants' only",
    "status": "flagged; M4 adds citations (Margueron 2018, Lattimer 2023, Garg & Colo 2018) and L_sym"},
   {"file": "sigma_ground/field/audit.py", "line": 266,
    "defect": "audit ledger lists 'TOV integration (full)' and 'M_TOV = 2.071 Msun' as live formulas; the code lives only in ../ssbm-theory",
    "evidence": "grep: no TOV integrator under sigma_ground/; ../ssbm-theory/sigma_ground/field/unsolved.py:495 has it",
    "status": "flagged; M5 supersedes, audit entry to be re-pointed"},
   {"file": "sigma_ground/field/planets.py", "line": 138,
    "defect": "NEUTRON_STAR body hand-entered (1.4 Msun, 11 km, four layers with round densities), uncited",
    "evidence": "read of planets.py:137-176",
    "status": "flagged; M6 derives the layers from a TOV profile"}]

# ---------------------------------------------------------------- 5. observational anchors -> compactness (Mentat r_s)
anchors = {
    "PSR J0030+0451 (Riley+2019)": (1.34, 12.71), "PSR J0030+0451 (Miller+2019)": (1.44, 13.02),
    "PSR J0740+6620 (Riley+2021)": (2.072, 12.39)}
comp = {}
for k, (m_sun, R_km) in anchors.items():
    rs = M("schwarzschild_radius", {"mass_kg": m_sun * C["M_sun"]}, f"rs:{k}")
    comp[k] = {"M_Msun_LIT": m_sun, "R_km_LIT": R_km, "r_s_m": rs, "C_GM_over_Rc2": rs / (2 * R_km * 1e3)}
out["nicer_compactness"] = {"rows": comp, "trace_flip_compactness_LIT": {"C": 0.262, "plus": 0.011, "minus": 0.017,
                            "ref": "Podkowka, Mendes & Poisson 2018, PRD 98, 064057 (90 %, approx. EOS-independent)"}}

# ---------------------------------------------------------------- 6. 3P/(rho c^2) for the repo's own P6 [LIT] rows (pilot_p6_sigma_bridge.py:48-50)
p6 = {"Sun core": (1.5e5, 2.5e16), "white dwarf centre": (1e9, 1e22), "neutron star centre (~2 n0)": (5e17, 1e34)}
out["p6_rows_3P_over_rho_c2"] = {k: {"rho_kg_m3_LIT": r, "P_Pa_LIT": P, "3P_over_rho_c2": 3 * P / (r * C["c"] ** 2)}
                                 for k, (r, P) in p6.items()}

# ---------------------------------------------------------------- 7. GW170817 P(2 n_sat) band in SI (Mentat unit conversion)
gw = {}
for lab, val in [("central", 3.5e34), ("low_90", 3.5e34 - 1.7e34), ("high_90", 3.5e34 + 2.7e34)]:
    gw[lab] = M("convert_units", {"value": val, "from_units": "dyn/cm**2", "to_units": "Pa"}, f"conv:gw170817_{lab}")
out["gw170817_P_2nsat_Pa"] = gw

# ---------------------------------------------------------------- 8. nuclear saturation mass density and the short-range sigma indicator
# Section 6 of UNIFICATION_PROGRAM: lambda -> 0 limit sigma = 8 pi G xi lambda^2 rho / c^2 (valid only for lambda << R).
from sigma_ground.inventory.core.sigma import XI
rho_sat = C["N0_FM3"] * 1e45 * C["m_n"]
sig = {f"lambda_{l}km": 8 * math.pi * C["G"] * XI * (l * 1e3) ** 2 * rho_sat / C["c"] ** 2 for l in (0.3, 1.0, 3.0)}
out["short_range_sigma_indicator"] = {"xi_repo": XI, "rho_sat_kg_m3": rho_sat, "sigma_at_rho_sat": sig,
    "note": "[DERIVED] under section 6's stated reading (normalisation C2 and sign unresolved); local formula valid only "
            "for lambda << R_NS ~ 12 km, so lambda >= ~3 km needs the full Yukawa integral over a TOV profile (module M8)."}

# ---------------------------------------------------------------- 9. the parallel-session scratch result (untracked, not rerun here)
try:
    tsp = json.load(open(REPO / "tov_sigma_pair_results.json", encoding="utf8"))
    out["prior_scratch_tov_sigma_pair"] = {
        "file": str(REPO / "tov_sigma_pair_results.json"), "status": "untracked, parallel session, not rerun here",
        "benchmarks": {k: tsp[k] for k in tsp if k.startswith("benchmark") or k == "max_mass"},
        "rows": [{k: r.get(k) for k in ["w_c", "M_Msun", "R_km", "Q_sigma_over_Q_Tolman_exact", "pred_1PN_1_minus_wc",
                                         "P7_as_written_1_minus_wc_over_2", "Tolman_over_Schwarzschild"]} for r in tsp["rows"]]}
except Exception as e:  # pragma: no cover
    out["prior_scratch_tov_sigma_pair"] = {"error": repr(e)}

# ---------------------------------------------------------------- 10. static survey of the repo (what exists)
TERMS = ["Fermi", "degenerate", "TOV", "Tolman", "polytrope", "Lane.Emden", "binding_energy", "nuclear", "neutron_star",
         "white_dwarf", "equation_of_state", "Chandrasekhar", "semi.empirical", "Weizs", "bag.model", "symmetry_energy",
         "saturation"]
DIRS = ["sigma_ground/field", "sigma_ground/inventory", "sigma_ground/mcp", "sigma_ground/dynamics", "scripts"]
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
out["existing_relevant_code"] = {
    "sigma_ground/field/binding.py": {"defs": defs("sigma_ground/field/binding.py"),
        "what": "Coulomb term from electrostatics (A_C_MEV) and sigma-scaling of a GIVEN binding energy; no SEMF fit"},
    "sigma_ground/field/interface/element.py": {"defs": [d for d in defs("sigma_ground/field/interface/element.py") if "binding" in d or "stable_mass" in d],
        "what": "SEMF set #1 (15.56/17.23/A_C/23.29/12.0, 'FITTED', uncited); asymmetry-term convention inconsistent with stable_mass_number"},
    "sigma_ground/mcp/tools/nuclear.py": {"defs": defs("sigma_ground/mcp/tools/nuclear.py"),
        "what": "SEMF set #2 hardcoded inline (15.75/17.8/0.711/23.7/11.18) + private u, m_H, m_n constants (PLATINUM 3)"},
    "sigma_ground/field/nucleon.py": {"defs": defs("sigma_ground/field/nucleon.py"), "what": "m_p(sigma), m_n(sigma) — the sigma hook"},
    "sigma_ground/field/constants.py:395-411": {"what": "N0_FM3=0.16, E_SAT_MEV=-16, K_SAT_MEV=230, J_SYM_MEV=32 — uncited, unused by any engine"},
    "sigma_ground/field/planets.py:137-176": {"what": "NEUTRON_STAR: hand-entered 1.4 Msun / 11 km with four layers, densities 1e10-1e18, no citation"},
    "sigma_ground/field/audit.py:247-279": {"what": "audit entries describe a Chandrasekhar EOS + full TOV + 'M_TOV = 2.071 Msun' that live only in ../ssbm-theory"},
    "sigma_ground/field/interface/statistical.py": {"defs": [d for d in defs("sigma_ground/field/interface/statistical.py") if "fermi" in d.lower()],
        "what": "Fermi-Dirac occupation only; no Fermi-gas EOS"},
    "sigma_ground/field/interface/quantum_matter.py": {"defs": [d for d in defs("sigma_ground/field/interface/quantum_matter.py") if "fermi" in d.lower()],
        "what": "metal free-electron Fermi energy / Thomas-Fermi screening, keyed by material"},
    "sigma_ground/mcp/tools/mathx.py::solve_ode": {"what": "symbolic sympy.dsolve only — no numeric IVP; Lane-Emden/TOV cannot be integrated through Mentat today"},
    "../ssbm-theory/sigma_ground/field/unsolved.py:148-635": {"what": "prior art outside the repo: Chandrasekhar EOS + Skyrme-like term (pure neutron matter, "
        "negative P clamped to 0, forward-Euler TOV dr = 10 m, sigma from a fixed R = 12 km); claims M_TOV = 2.071 Msun. Not portable as is."},
    "scratchpad/sigma-bridge-1pn/tov_sigma_pair.py": {"what": "parallel-session scratch: TOV + Tolman mass + sigma monopole on the TOV metric, "
        "benchmarks (Schwarzschild interior, Lane-Emden n=1 limit). Source of the M8 test design."},
    "pyproject.toml:12": {"what": "core library dependencies = [] (scipy only in the mcp extra) -> integrators must be pure Python"}}

# ---------------------------------------------------------------- 11. references (verified by web search this session)
out["references_verified"] = json.load(open(OUT / "n1_references.json", encoding="utf8"))
json.dump(out, open(OUT / "n1_survey.json", "w", encoding="utf8"), indent=1, ensure_ascii=False)
print(json.dumps({k: out[k] for k in ["chandrasekhar_mass_target", "nicer_compactness", "p6_rows_3P_over_rho_c2",
                                      "gw170817_P_2nsat_Pa", "short_range_sigma_indicator", "lane_emden_n1_mentat_dsolve"]},
                 indent=1, ensure_ascii=False))
print(json.dumps(out["semf_comparison"]["rms_pct"], indent=1))
for r in rows:
    print(r["Z"], r["A"], round(r["BE_AME2020_MeV"], 2), round(r["resid_mcp_SEMF_pct"], 2), round(r["resid_element_py_pct"], 2),
          round(r["resid_element_py_consistent_pct"], 2), round(r["BE_mcp_exact_MeV"] - r["BE_AME2020_MeV"], 4))
print(json.dumps(sym, indent=1))
