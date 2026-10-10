"""U6 caveat numbers (Mentat): mass/pressure crossover frequency, and the route-(b) Klein-Gordon cutoff vs the swept omega.
Appends a 'caveat_numbers' block to u6_results.json. Run from the repo root after u6_analyse.py."""
import sys, os, json
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
from mentat_client import mentat

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "u6_results.json")
d = json.load(open(P, encoding="utf-8"))
L = d["mentat_leaves"]
c = float(L["c"]["value"]); R_expr = "pi*5000"; w = 1e-3
B = {}
def rec(k, tool, args):
    v, s = mentat(tool, args); B[k] = {"value": v, "source": s, "tool": tool, "args": args}; print(k, v, "|", s)
# A_mass/A_press = (2/3)(I2/MR^2)(omega R/c)^2 / w_c = 1  ->  omega R/c = sqrt(3 w_c / (2 I2/MR^2))
rec("crossover_omegaR_over_c", "simplify_expr", {"expression": f"N(sqrt(3*{w}/(2*(pi**3 - 6*pi)/pi**3)), 15)"})
rec("crossover_over_cs_centre", "simplify_expr", {"expression": f"N(sqrt(3*{w}/(2*(pi**3 - 6*pi)/pi**3))/sqrt(2*{w}/3), 15)"})
rec("crossover_symbolic_in_cs_units", "simplify_expr", {"expression": "sqrt(3*w/(2*(1 - 6/pi**2)))/sqrt(2*w/3)"})
# route (b): massive sigma with range lambda <= 34.5 km (program's MICROSCOPE-derived bound, design_synthesis 2.5 item 6)
rec("omega_cut_s-1_lambda_34p5km", "simplify_expr", {"expression": f"N({c}/34500, 15)"})
rec("omega_max_swept_s-1", "simplify_expr", {"expression": f"N(0.08*{c}/({R_expr}), 15)"})
rec("omega_min_swept_s-1", "simplify_expr", {"expression": f"N(0.01*{c}/({R_expr}), 15)"})
rec("yukawa_suppression_at_30R_lambda_34p5km", "simplify_expr", {"expression": f"N(exp(-30*{R_expr}/34500), 6)"})
d["caveat_numbers"] = B
json.dump(d, open(P, "w", encoding="utf-8"), indent=1)
print("appended to", P)
