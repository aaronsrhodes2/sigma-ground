"""U6 — fetch every physical leaf and every predicted amplitude from Mentat (PLATINUM 21); write mentat_leaves.json.
Run from the sigma-ground repo root:  python -X utf8 <this file>
"""
import sys, json, math, os
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
from mentat_client import mentat
from sigma_ground.inventory.core.sigma import XI

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mentat_leaves.json")
L = {}
def rec(key, tool, args):
    v, s = mentat(tool, args)
    L[key] = {"value": v, "source": s, "tool": tool, "args": args}
    print(key, v, "|", s, flush=True)
    return v

G = rec("G", "lookup_constant", {"name": "G"})
c = rec("c", "lookup_constant", {"name": "c"})
L["XI"] = {"value": XI, "source": "sigma_ground.inventory.core.sigma.XI (repo engine; only surviving copy of xi, decision C1)"}

# design choices (disclosed): a, w_c, eps, grid
a = 5e3; w_c = 1e-3; eps = 0.05; Nr = 8000
L["choices"] = {"a_m": a, "R_m": "pi*a", "w_c": w_c, "eps": eps, "N_r": Nr, "domain": "[0, 40R]", "probe": "30R",
                "omegaR_over_c": [0.01, 0.02, 0.04, 0.08], "periods": 6, "window": "last 3 periods",
                "note": "prescribed pulsation (SPH has no breathing mode); w_c, a, eps are the design's choices, not physical leaves"}

# n = 1 Lane-Emden integrals (Mentat sympy)
Im = rec("int_x_sinx_0_pi", "integrate_expr", {"expression": "x*sin(x)", "variable": "x", "lower": "0", "upper": "pi"})
Ip = rec("int_sin2x_0_pi", "integrate_expr", {"expression": "sin(x)**2", "variable": "x", "lower": "0", "upper": "pi"})
I2 = rec("int_x3_sinx_0_pi", "integrate_expr", {"expression": "x**3*sin(x)", "variable": "x", "lower": "0", "upper": "pi"})
# closed forms used by the conservative (cell-integrated) source
rec("antideriv_r_sin_r_over_b", "integrate_expr", {"expression": "r*sin(r/b)", "variable": "r"})
rec("antideriv_sin2_r_over_a", "integrate_expr", {"expression": "sin(r/a)**2", "variable": "r"})

# polytrope leaves: rho_c from w_c; M = 4 pi rho_c a^3 * Im
rho_c = rec("rho_c_kg_m3", "simplify_expr", {"expression": f"N({w_c}*{c}**2/(6*pi*{G}*{a}**2), 15)"})
rho_c = float(rho_c)
M = float(rec("M_kg", "simplify_expr", {"expression": f"N(4*pi*{rho_c}*{a}**3*pi, 15)"}))
r_s = rec("r_s_m", "schwarzschild_radius", {"mass_kg": M})
rec("check_rs_over_R_symbolic", "simplify_expr", {"expression": "(2*G*4*pi**2*(w*c**2/(6*pi*G*a**2))*a**3/c**2)/(pi*a)"})
rec("check_intPdV_over_Mc2_symbolic", "simplify_expr",
    {"expression": "(4*pi*(2*pi*G*a**2)*(w*c**2/(6*pi*G*a**2))**2*a**3*(pi/2))/((4*pi**2*(w*c**2/(6*pi*G*a**2))*a**3)*c**2)"})
rec("I2_over_MR2_symbolic", "simplify_expr", {"expression": "(pi**3 - 6*pi)/pi/pi**2"})
rec("I2_over_MR2", "simplify_expr", {"expression": "N((pi**3 - 6*pi)/pi**3, 15)"})
rec("dr_m", "simplify_expr", {"expression": f"N(40*pi*{a}/{Nr}, 15)"})
rec("cs_centre_over_c_symbolic", "simplify_expr", {"expression": "sqrt(2*w/3)"})
rec("cs_centre_over_c", "simplify_expr", {"expression": f"N(sqrt(2*{w_c}/3), 15)"})
rec("cs_air_293K_m_s", "speed_of_sound_in_ideal_gas", {"temperature_k": 293.15})

# predicted amplitudes (design formulas), evaluated by Mentat sympy
rec("A_press_m", "simplify_expr", {"expression": f"N({XI}*{r_s}*{w_c}*{eps}/2, 15)"})
for q in (0.01, 0.02, 0.04, 0.08):
    # A_mass = xi r_s eps omega^2 I2/(3 M c^2) = xi r_s eps (omega R/c)^2 (I2/(M R^2)) / 3
    rec(f"A_mass_m_wR{q}", "simplify_expr", {"expression": f"N({XI}*{r_s}*{eps}*{q}**2*((pi**3 - 6*pi)/pi**3)/3, 15)"})
rec("v_static_m", "simplify_expr", {"expression": f"N({XI}*{r_s}*(1 - {w_c}/2), 15)"})
# all-orders monopole form factors (exact retarded solution, j0 kernel) -- supplementary
rec("FF_pressure_symbolic", "integrate_expr", {"expression": "sin(x)**2*sin(k*x)/(k*x)", "variable": "x", "lower": "0", "upper": "pi"})
for q in (0.01, 0.02, 0.04, 0.08):
    k = q / math.pi   # k*a with R = pi a
    rec(f"FF_pressure_wR{q}", "simplify_expr",
        {"expression": f"N(((-Si(pi*({k} - 2)) - Si(pi*({k} + 2)))/(4*{k}) + Si(pi*{k})/(2*{k}))/(pi/2), 15)"})
    # mass mode: amplitude/(leading) = 3 * int x^2 sin x * j1(k x)/k dx / int x^4 sinc... = [k int x^2 sin(x) j1(kx) dx] / [(k^2/3) int x^3 sin x dx]
    rec(f"FF_mass_wR{q}", "simplify_expr",
        {"expression": f"N(Integral(x**2*sin(x)*(sin({k}*x)/({k}*x)**2 - cos({k}*x)/({k}*x)), (x, 0, pi))*{k}/(({k}**2/3)*(pi**3 - 6*pi)))"})

json.dump(L, open(OUT, "w", encoding="utf-8"), indent=1)
print("wrote", OUT)
