"""U3 — Mentat leaves (PLATINUM 21). Every constant and dimensionless leaf used by the U3 run, with its Mentat source.

Run from the sigma-ground repo root:  python -X utf8 <this file>
Writes mentat_leaves.json next to this file. Touches nothing in the repo.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
sys.path.insert(0, r"D:/Aaron/development/sigma-ground")
from mentat_client import mentat  # noqa: E402

N, M, B = 128, 1.0e12, 5000.0
out = {}


def leaf(key, tool, args):
    v, s = mentat(tool, args)
    out[key] = dict(value=v, source=s, tool=tool, args=args)
    print(f"{key:28s} = {v!r:40s} [{s}]", flush=True)
    return v


G = leaf("G", "lookup_constant", {"name": "G"})
leaf("k_B", "lookup_constant", {"name": "k_B"})
# design's dimensionless groups (Mentat sympy)
leaf("trelax_over_tcross_N128", "simplify_expr", {"expression": "N(0.1*128/log(128))"})
leaf("dE_over_K_formula", "simplify_expr", {"expression": "(1+d)^2-1"})
leaf("virial_K_of_E_softened", "solve_equation", {"equation": "2*K + w*(E-K) = 0", "variable": "K"})
# Plummer-sphere dimensionless factors (no Mentat Plummer tool: gap 7 / utility 4) — evaluated by Mentat sympy
leaf("plummer_W_factor_3pi_32", "simplify_expr", {"expression": "N(3*pi/32)"})
leaf("plummer_K_factor_3pi_64", "simplify_expr", {"expression": "N(3*pi/64)"})
leaf("t_cr_HM_over_unit", "simplify_expr", {"expression": "N((3*pi/32)^(-3/2))"})
leaf("r_half_over_b", "simplify_expr", {"expression": "N(1/sqrt(2^(2/3)-1))"})
leaf("v_rms_over_sqrt_GM_b", "simplify_expr", {"expression": "N(sqrt(3*pi/32))"})
r_h = float(out["r_half_over_b"]["value"]) * B
leaf("T_orb_at_r_half_s", "orbital_period", {"semimajor_axis_m": r_h, "central_mass_kg": M / 2})
f_b = float(leaf("plummer_Menc_frac_at_b", "simplify_expr", {"expression": "N(2^(-3/2))"}))
leaf("T_orb_at_b_s", "orbital_period", {"semimajor_axis_m": B, "central_mass_kg": M * f_b})

# gas side: the repo's equipartition formula (repo engines are Mentat)
from sigma_ground.field.interface.statistical import heat_capacity_equipartition  # noqa: E402
C_gas = heat_capacity_equipartition(3, n=N)
out["C_gas_equipartition_J_per_K"] = dict(value=C_gas, source="sigma_ground.field.interface.statistical.heat_capacity_equipartition(3, n=128)")
print("C_gas =", C_gas)

# consistency: the engine's own G is the one Mentat reports
from sigma_ground.field.interface import nbody  # noqa: E402
out["engine_G_equals_mentat_G"] = bool(nbody._G == G)
print("engine _G == Mentat G:", out["engine_G_equals_mentat_G"])

json.dump(out, open(os.path.join(HERE, "mentat_leaves.json"), "w", encoding="utf-8"), indent=1)
