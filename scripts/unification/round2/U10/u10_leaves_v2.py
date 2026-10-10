"""U10 repair (v2) leaves: the extra Mentat numbers the skeptics' corrections need (PLATINUM 21). Writes u10_leaves_v2.json.
The original u10_leaves.json (G, c, I, I3, J, the n = 1 series) is kept and re-used; G and c are re-queried here and must match.
New:
  * split-kick fixed point of 'm explicit-Euler substeps of h, then a kick m*h*s': per mode T*/v_d = m mu / (1 - (1 - mu)^m),
    mu = h * lambda.  Mentat series in mu (symbolic m), the m = 1 case, and the m = 20 case.
  * internal-energy sensitivity of the 'exact' O(w^2) C1 offset: for n = 1, u = n P = P, so if rho is the REST density both
    sources gain P/c^2 = B/3 (B = the 3P/c^2 integral).  C1 offset, Q_T/Q_N, and the C2 difference (P/c^2 cancels).
Run from the repo root:  python -X utf8 <this file>"""
import json, os, sys
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
from mentat_client import mentat
OUT = os.path.dirname(os.path.abspath(__file__))
L = {}


def rec(key, tool, args):
    v, s = mentat(tool, args); L[key] = {"value": v, "source": s, "tool": tool, "args": args}; return v


old = json.load(open(os.path.join(OUT, "u10_leaves.json")))
G = rec("G", "lookup_constant", {"name": "G"}); C = rec("c", "lookup_constant", {"name": "c"})
assert G == old["G"]["value"] and C == old["c"]["value"], "Mentat constants changed since the original run"
I3 = old["numeric_integrals"]["I3"]

# ---- split-kick fixed point ----
rec("split_series_symbolic_m", "series_expansion", {"expression": "m*mu/(1-(1-mu)**m)", "variable": "mu", "point": "0", "order": 3})
rec("split_mu2_coefficient_factored", "factor_expression", {"expression": "(1/2 - m/2)**2 - (m/6 - 1/6)*(m - 2)"})
rec("split_m1_exact", "simplify_expr", {"expression": "mu/(1-(1-mu)**1)"})
rec("split_series_m20", "series_expansion", {"expression": "20*mu/(1-(1-mu)**20)", "variable": "mu", "point": "0", "order": 3})
# operator form used in the analysis: T* - v_d = ((m-1)/2) h s + ((m^2-1)/12) h^2 (-L s) + ...;  near the centre -L theta ~ theta/a^2,
# so T* - v_d ~ ((m-1)/2) h s [1 + (m+1) mu/6],  mu = h/a^2 (alpha = 1).
rec("split_bias_factor_check", "simplify_expr",
    {"expression": "((m-1)/2)*mu*(1 + (m+1)*mu/6) - (mu*(m/2 - 1/2) + mu**2*((1/2 - m/2)**2 - (m/6 - 1/6)*(m - 2)))"})

# ---- internal-energy sensitivity (symbols as in u10_leaves.py; q3 = I3) ----
A = "(pi + (4*w/3)*(pi + pi/2))"; B = "(w*(pi/2 + (4*w/3)*(pi/2 + q3)))"
rec("C1_offset_rho_total_energy", "series_expansion",
    {"expression": f"({A}-{B})/({A}+{B}) - (1-w/2)/(1+w/2)", "variable": "w", "point": "0", "order": 3})
rec("C1_offset_rho_rest_plus_internal_energy", "series_expansion",
    {"expression": f"({A}+{B}/3-{B})/({A}+{B}/3+{B}) - (1-w/2)/(1+w/2)", "variable": "w", "point": "0", "order": 3})
rec("QT_over_QN_rho_rest_plus_internal_energy", "series_expansion",
    {"expression": f"({A}+{B}/3+{B})/pi", "variable": "w", "point": "0", "order": 3})
rec("C2_difference_unchanged_by_internal_energy", "simplify_expr",
    {"expression": f"(({A}+{B}/3+{B}) - ({A}+{B}/3-{B})) - (({A}+{B}) - ({A}-{B}))"})
for w, tag in ((1e-3, "w1e-3"), (1e-2, "w1e-2")):
    wr = "Rational(1,1000)" if w == 1e-3 else "Rational(1,100)"
    for k in ("C1_offset_rho_total_energy", "C1_offset_rho_rest_plus_internal_energy"):
        expr = L[k]["value"].replace("q3", repr(I3))
        rec(f"{k}_{tag}", "simplify_expr", {"expression": f"N(({expr}).subs(w, {wr}), 12)"})
    # exact (unexpanded) values too
    rec(f"C1_offset_rho_total_energy_exact_{tag}", "simplify_expr",
        {"expression": f"N((({A}-{B})/({A}+{B}) - (1-w/2)/(1+w/2)).subs(q3, {I3!r}).subs(w, {wr}), 12)"})
    rec(f"C1_offset_rho_rest_plus_internal_energy_exact_{tag}", "simplify_expr",
        {"expression": f"N((({A}+{B}/3-{B})/({A}+{B}/3+{B}) - (1-w/2)/(1+w/2)).subs(q3, {I3!r}).subs(w, {wr}), 12)"})

json.dump(L, open(os.path.join(OUT, "u10_leaves_v2.json"), "w"), indent=1)
for k, v in L.items():
    print(k, "=", v["value"], " <-", v["source"])
