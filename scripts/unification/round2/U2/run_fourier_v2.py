"""U2 repair round - Fourier-leg conditioning, measured on the engine.

The original run_fourier.py computed the inverse-map amplification as 800 x log10(1/(1 - lam)) with lam = 0.4, the
checkerboard value. On an insulated (Neumann, cell-centred) 32-cell grid the highest eigenmode is
prod_axes cos(31 pi (i + 1/2) / 32), with eigenvalue 12 sin^2(31 pi / 64) instead of 12. Here the per-substep decay
factor of that mode is MEASURED by one diffuse_fvm substep on the same material (repo cascade water_ice, 200 K) and
grid (dx = 1 cm); the 800-substep amplification follows by powering (DERIVED). Mentat simplify_expr gives the
closed forms. The two 'reversed run' requests are re-characterised from the engine source: total_time <= 0 hits the
early return at heat.py:122; k < 0 makes stable_dt return inf (heat.py:32-33), which hits the return at heat.py:159.
Both are input guards (silent no-ops), i.e. [BY CONSTRUCTION], not a measurement of physics."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import u2_common as C
sys.path.insert(0, C.REPO + "/tools")
from mentat_client import mentat
from sigma_ground.dynamics.fields.heat import diffuse_fvm, stable_dt
from sigma_ground.dynamics.continuum import material_properties

n, dx, Tb = 32, 0.01, 200.0
props = material_properties("water_ice", Tb)
kap, rho, cp = props["thermal_conductivity_W_mK"], props["density_kg_m3"], props["specific_heat_J_kgK"]
k = np.full((n, n, n), kap)
rcp = np.full((n, n, n), rho * cp)
alpha = kap / (rho * cp)
dt1 = stable_dt(alpha, dx)
i = np.arange(n) + 0.5
out = {"material_cascade_200K": {"k_W_mK": kap, "rho_kg_m3": rho, "cp_J_kgK": cp, "alpha_m2_s": alpha,
                                 "melting_point_K": props["melting_point_K"],
                                 "source": "sigma_ground.dynamics.continuum.material_properties('water_ice', 200 K)"},
       "dt_substep_s": dt1, "lam_12_alpha_dt_over_dx2": 12 * alpha * dt1 / dx ** 2}
modes = {}
for name, m1 in (("neumann_highest_k31", np.cos(31 * np.pi * i / n)), ("checkerboard", (-1.0) ** np.arange(n))):
    shape = m1[:, None, None] * m1[None, :, None] * m1[None, None, :]
    T0 = Tb + 1.0 * shape
    T1, info = diffuse_fvm(T0, k, rcp, dx, total_time=dt1)
    f = (T1 - Tb) / (T0 - Tb)
    interior = f[1:-1, 1:-1, 1:-1]
    modes[name] = {"substeps": info["substeps"], "factor_mean": float(f.mean()), "factor_min": float(f.min()),
                   "factor_max": float(f.max()), "factor_interior_mean": float(interior.mean()),
                   "is_eigenmode_spread": float(f.max() - f.min())}
out["per_substep_factor_measured"] = modes
fN = modes["neumann_highest_k31"]["factor_mean"]
out["log10_inverse_amplification_800_from_measured_factor"] = float(-800 * math.log10(fN))
L = {}
for key, expr in (("factor_neumann_closed_form", "1 - 0.4*sin(31*pi/64)**2"),
                  ("log10_amp_800_neumann", "-800*log(1 - 0.4*sin(31*pi/64)**2, 10)"),
                  ("log10_amp_800_checkerboard", "-800*log(1 - 0.4, 10)")):
    v, s = mentat("simplify_expr", {"expression": expr})
    import sympy
    L[key] = {"value": str(v), "float": float(sympy.sympify(v).evalf(17)), "source": s, "tool": "simplify_expr",
              "args": {"expression": expr}}
out["mentat"] = L
out["factor_rel_err_vs_closed_form"] = abs(fN - L["factor_neumann_closed_form"]["float"]) / L["factor_neumann_closed_form"]["float"]
# reversed-run requests: what the engine returns, and which guard returns it
T = Tb + 1.0 * np.cos(31 * np.pi * i / n)[:, None, None] * np.ones((n, n, n))
Tn, inf_n = diffuse_fvm(T, k, rcp, dx, total_time=-10 * dt1)
Tk, inf_k = diffuse_fvm(T, -k, rcp, dx, total_time=10 * dt1)
out["reversed_requests"] = {
    "negative_time": {"substeps": inf_n["substeps"], "unchanged": bool(np.array_equal(Tn, T)),
                      "guard": "heat.py:122 'if total_time <= 0 or not solid.any(): return T, info'"},
    "negative_k": {"substeps": inf_k["substeps"], "unchanged": bool(np.array_equal(Tk, T)),
                   "stable_dt_for_negative_alpha": stable_dt(-alpha, dx),
                   "guard": "heat.py:32-33 stable_dt returns inf for alpha_max <= 0; heat.py:159 'if not np.isfinite(dt) or dt <= 0: return T, info'"},
    "tag": "[BY CONSTRUCTION] input guards that silently return T unchanged; not a measurement"}
with open(os.path.join(C.HERE, "res_fourier_v2.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1))
