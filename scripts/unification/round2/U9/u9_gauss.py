"""U9 part C prediction — secular perihelion advance of the ENGINE's 1PN laws, rest and boosted, from first-order
perturbation theory (Gauss's equation for omega), integrals by Mentat (integrate_expr, sympy, exact rationals).

[DERIVED] Gauss:  d(omega)/df = (r^2/(G M e)) [ -R cos f + T (1 + r/p) sin f ],  with the perturbing acceleration
written as (G M/(c^2 r^2)) * w^2 * P~ (w = sqrt(GM/p), velocities in units of w, GM/r = w^2 x, x = 1 + e cos f):
    d(omega)/orbit = u * (1/(pi e)) * Int_0^2pi [ -R~ cos f + T~ (1 + 1/x) sin f ] df,   u = pi G M/(c^2 p)
Test-particle laws in a frame where the Sun moves with V = k w (cos psi, sin psi) and Mercury with v + V:
    gr_1pn (absolute v_i, engine coefficients (A,B,C) measured by u9_instant.py):
        P~ = A x n - B |v+V|^2 n + C (n.(v+V)) (v+V)
    EIH, cross-term sign s (engine s = -1, Will/Wikipedia s = +1):
        P~ = (4x - |v+V|^2 - 2|V|^2 + 4 (v+V).V + (3/2)(n.V)^2) n + s (n.(4(v+V) - 3V)) v
Rest frame check: (-A + 2B + 2C) -> 6 for the textbook (4,1,4).
[DERIVED] Gauss for e:  de/df = (r^2/(G M)) [ R sin f + T (cos f + cos E) ],  cos E = (e + cos f)/x.
Usage: python -X utf8 u9_gauss.py OUT.json k_rational_numerator k_rational_denominator
"""
import json, sys
import sympy as sp
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
from mentat_client import mentat

OUT = sys.argv[1]
K_NUM, K_DEN = int(sys.argv[2]), int(sys.argv[3])
f = sp.symbols("f", real=True)
e = sp.Rational(257, 1250)            # 0.2056, repo rolling_analysis._ORBITAL['Mercury']
k = sp.Rational(K_NUM, K_DEN)         # V/w
x = 1 + e * sp.cos(f)
n = sp.Matrix([sp.cos(f), sp.sin(f)])
t = sp.Matrix([-sp.sin(f), sp.cos(f)])
v = sp.Matrix([-sp.sin(f), e + sp.cos(f)])
FRAMES = {"rest": (0, 0), "x": (1, 0), "y": (0, 1), "z": (0, 0)}   # z: no in-plane component (see docstring)


def law(name, Vv):
    vi = v + Vv
    if name.startswith("gr"):
        A, B, C = LAWS[name]
        return A * x * n - B * vi.dot(vi) * n + C * n.dot(vi) * vi
    s = LAWS[name]
    return (4 * x - vi.dot(vi) - 2 * Vv.dot(Vv) + 4 * vi.dot(Vv) + sp.Rational(3, 2) * n.dot(Vv) ** 2) * n \
        + s * n.dot(4 * vi - 3 * Vv) * v


LAWS = {"gr_engine": (-4, -1, -4), "gr_textbook": (4, 1, 4), "eih_engine": -1, "eih_textbook": 1}
res = {"e": str(e), "k": str(k), "k_float": float(k), "pred_over_u": {}, "sources": {}}
for lname in LAWS:
    for fr, (cps, sps) in FRAMES.items():
        Vv = sp.Matrix([k * cps, k * sps])
        P = law(lname, Vv)
        R, T = P.dot(n), P.dot(t)
        integrand = sp.expand(-R * sp.cos(f) + T * sp.sin(f)) + sp.expand(T * sp.sin(f)) / x
        val, src = mentat("integrate_expr", {"expression": str(integrand), "variable": "f", "lower": "0", "upper": "2*pi"})
        num = float(sp.N(sp.sympify(val) / (sp.pi * e), 15))
        res["pred_over_u"][f"{lname}_{fr}"] = num
        res["sources"][f"{lname}_{fr}"] = {"tool": "integrate_expr", "source": src, "value": val}
        # [DERIVED] Gauss for e:  de/orbit = (w^2/c^2) Int [R~ sin f + T~ (cos f + (e + cos f)/x)] df  = u * (1/pi) Int[...]
        ie = sp.expand(R * sp.sin(f) + T * sp.cos(f)) + sp.expand(T * (e + sp.cos(f))) / x
        val_e, src_e = mentat("integrate_expr", {"expression": str(ie), "variable": "f", "lower": "0", "upper": "2*pi"})
        num_e = float(sp.N(sp.sympify(val_e) / sp.pi, 15))
        res.setdefault("de_pred_over_u", {})[f"{lname}_{fr}"] = num_e
        res["sources"][f"{lname}_{fr}_de"] = {"tool": "integrate_expr", "source": src_e, "value": val_e}
        print(lname, fr, num, num_e, flush=True)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1)
