"""[VERIFIED numerics] cross-check of the Mentat (sympy) definite integrals in u9_gauss.json by periodic trapezoid
quadrature of the identical integrands (spectrally accurate for smooth periodic functions). Also integrates the
same laws numerically for the z-boost with the out-of-plane V kept (3-vector algebra), which u9_gauss.py reduced to
the in-plane problem.
Usage: python -X utf8 u9_gauss_check.py <folder>"""
import json, os, sys
import numpy as np
D = sys.argv[1]
g = json.load(open(os.path.join(D, "u9_gauss.json"), encoding="utf-8"))
e = 257 / 1250; k = g["k_float"]
f = np.linspace(0, 2 * np.pi, 20001)[:-1]
x = 1 + e * np.cos(f)
z0 = np.zeros_like(f)
n = np.stack([np.cos(f), np.sin(f), z0]); t = np.stack([-np.sin(f), np.cos(f), z0])
v = np.stack([-np.sin(f), e + np.cos(f), z0])
dot = lambda a, b: np.sum(a * b, axis=0)
LAWS = {"gr_engine": (-4, -1, -4), "gr_textbook": (4, 1, 4), "eih_engine": -1, "eih_textbook": 1}
FR = {"rest": (0, 0, 0), "x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
out = {}
for ln, par in LAWS.items():
    for fr, d in FR.items():
        V = k * np.array(d, dtype=float)[:, None] * np.ones_like(f)
        vi = v + V
        if ln.startswith("gr"):
            A, B, C = par
            P = A * x * n - B * dot(vi, vi) * n + C * dot(n, vi) * vi
        else:
            P = (4 * x - dot(vi, vi) - 2 * dot(V, V) + 4 * dot(vi, V) + 1.5 * dot(n, V) ** 2) * n + par * dot(n, 4 * vi - 3 * V) * v
        R, T = dot(P, n), dot(P, t)
        dw = np.mean(-R * np.cos(f) + T * (1 + 1 / x) * np.sin(f)) * 2 * np.pi / (np.pi * e)
        de = np.mean(R * np.sin(f) + T * (np.cos(f) + (e + np.cos(f)) / x)) * 2 * np.pi / np.pi
        key = f"{ln}_{fr}"
        out[key] = {"domega_over_u_quad": dw, "de_over_u_quad": de,
                    "domega_over_u_mentat": g["pred_over_u"][key], "de_over_u_mentat": g["de_pred_over_u"][key]}
        print(f"{key:20s} dω/u quad {dw:+.6f} mentat {g['pred_over_u'][key]:+.6f}   de/u quad {de:+.6f} mentat {g['de_pred_over_u'][key]:+.6f}")
json.dump(out, open(os.path.join(D, "u9_gauss_check.json"), "w", encoding="utf-8"), indent=1)
