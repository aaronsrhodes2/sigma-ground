"""U10 leaves: every constant and closed-form number from Mentat (PLATINUM 21), plus the exact n = 1 predictions built from them.
Writes u10_leaves.json. Run from the repo root."""
import json, math, os, sys
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
sys.path.insert(0, r"D:/Aaron/development/sigma-ground")
from mentat_client import mentat
OUT = os.path.dirname(os.path.abspath(__file__))
L = {}
def rec(key, tool, args):
    v, s = mentat(tool, args); L[key] = {"value": v, "source": s, "tool": tool, "args": args}; return v

G = rec("G", "lookup_constant", {"name": "G"}); C = rec("c", "lookup_constant", {"name": "c"})
from sigma_ground.field.constants import G as G_eng, C as C_eng
from sigma_ground.inventory.core.sigma import XI
L["engine_constants_equal_mentat"] = {"G": G_eng == G, "c": C_eng == C}
L["XI"] = {"value": XI, "source": "sigma_ground.inventory.core.sigma.XI (engine)"}
ints = {"I1": "x*sin(x)", "I2": "sin(x)**2", "I": "sin(x)**2/x", "I3": "sin(x)**3/x", "J": "sin(x)**3/x**2"}
num = {}
for k, e in ints.items():
    sym = rec(k + "_symbolic", "integrate_expr", {"expression": e, "variable": "x", "lower": "0", "upper": "pi"})
    num[k] = float(rec(k, "simplify_expr", {"expression": f"N({sym}, 17)"}))
rec("series_pressure_only_ratio", "series_expansion", {"expression": "(1-w/2)/(1+w/2)", "variable": "w", "point": "0", "order": 3})
rec("six_P_difference", "simplify_expr", {"expression": "(rho*c**2 + 3*P) - (rho*c**2 - 3*P)"})
rec("n1_potential_solves_poisson", "simplify_expr", {"expression": "diff(r**2*diff(a**2*(1+a*sin(r/a)/r), r), r)/r**2 + a*sin(r/a)/r"})
# exact weighted monopole ratio and centre deficit, expanded by Mentat (symbols: q3 = I3, ii = I, jj = J)
A = "(pi + (4*w/3)*(pi + pi/2))"; B = "(w*(pi/2 + (4*w/3)*(pi/2 + q3)))"
rec("series_weighted_ratio", "series_expansion", {"expression": f"({A}-{B})/({A}+{B})", "variable": "w", "point": "0", "order": 3})
Ac = "(2 + (4*w/3)*(2 + ii))"; Bc = "(w*(ii + (4*w/3)*(ii + jj)))"
rec("series_deficit_over_w", "series_expansion", {"expression": f"2*{Bc}/(w*({Ac}+{Bc}))", "variable": "w", "point": "0", "order": 2})
rec("series_QT_over_QN", "series_expansion", {"expression": f"({A}+{B})/pi", "variable": "w", "point": "0", "order": 3})

def pred(w):
    I1, I2, I, I3, J = num["I1"], num["I2"], num["I"], num["I3"], num["J"]
    A_ = I1 + (4 * w / 3) * (I1 + I2); B_ = w * (I2 + (4 * w / 3) * (I2 + I3))
    Ac_ = 2 + (4 * w / 3) * (2 + I); Bc_ = w * (I + (4 * w / 3) * (I + J))
    return dict(w=w, ratio_pressure_only=(1 - w / 2) / (1 + w / 2), ratio_weighted_exact=(A_ - B_) / (A_ + B_),
                ratio_weighted_minus_pressure_only=(A_ - B_) / (A_ + B_) - (1 - w / 2) / (1 + w / 2),
                dQ_over_QN_weighted_exact=2 * B_ / I1, dQ_over_QN_weighted_rel_offset_from_w=(2 * B_ / I1 - w) / w,
                dQ_over_QN_unweighted=w * I2 / I1 * 2, QT_over_QN_weighted_exact=(A_ + B_) / I1, QT_over_QN_design_1p2_5w=1 + 2.5 * w,
                deficit_over_w_weighted_centre_exact=2 * Bc_ / (w * (Ac_ + Bc_)), deficit_over_w_unweighted_centre_exact=2 * I / (2 + w * I),
                compactness_rs_over_R=4 * w / 3)
L["numeric_integrals"] = num
L["predictions"] = [pred(1e-3), pred(1e-2)]
json.dump(L, open(os.path.join(OUT, "u10_leaves.json"), "w"), indent=1)
print(json.dumps({k: L[k] for k in ("numeric_integrals", "predictions")}, indent=1))
for k in ("series_weighted_ratio", "series_deficit_over_w", "series_QT_over_QN", "six_P_difference", "n1_potential_solves_poisson"):
    print(k, L[k]["value"], L[k]["source"])
