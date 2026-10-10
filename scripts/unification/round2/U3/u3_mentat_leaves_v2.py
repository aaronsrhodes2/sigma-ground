"""U3 repair (v2) - Mentat leaves (PLATINUM 21). Re-fetches every v1 leaf and adds the leaves the skeptic repairs need:
crossing-time conventions, window lengths in units of the orbital period at r_half, the Plummer moment-of-inertia
integrand's large-r behaviour, and closed forms for the small-sample statistics (chi-square with 2 dof, F(2, 6)).

Run from the sigma-ground repo root:  python -X utf8 <this file>
Writes mentat_leaves_v2.json next to this file. Touches nothing in the repo.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
sys.path.insert(0, r"D:/Aaron/development/sigma-ground")
from mentat_client import mentat  # noqa: E402

N, M, B = 128, 1.0e12, 5000.0
out = {}


def leaf(key, tool, args, note=""):
    v, s = mentat(tool, args)
    out[key] = dict(value=v, source=s, tool=tool, args=args, note=note)
    print(f"{key:34s} = {str(v)[:60]:60s} [{s}]", flush=True)
    return v


# ---- v1 leaves, re-fetched ----
G = leaf("G", "lookup_constant", {"name": "G"})
leaf("k_B", "lookup_constant", {"name": "k_B"})
leaf("trelax_over_tcross_N128", "simplify_expr", {"expression": "N(0.1*128/log(128))"})
leaf("dE_over_K_formula", "simplify_expr", {"expression": "(1+d)^2-1"})
leaf("virial_K_of_E_softened", "solve_equation", {"equation": "2*K + w*(E-K) = 0", "variable": "K"},
     "fixed-w approximation: leaves out E dw/dE")
leaf("t_cr_HM_over_unit", "simplify_expr", {"expression": "N((3*pi/32)^(-3/2))"},
     "Heggie-Mathieu t_cr = G M^(5/2)/(2|E|)^(3/2) = 2 R_V/v_rms, in design units sqrt(b^3/GM)")
leaf("r_half_over_b", "simplify_expr", {"expression": "N(1/sqrt(2^(2/3)-1))"})
leaf("v_rms_over_sqrt_GM_b", "simplify_expr", {"expression": "N(sqrt(3*pi/32))"})
r_h = float(out["r_half_over_b"]["value"]) * B
leaf("T_orb_at_r_half_s", "orbital_period", {"semimajor_axis_m": r_h, "central_mass_kg": M / 2})
# ---- v2: crossing-time conventions (design unit = sqrt(b^3/GM)) ----
leaf("R_V_over_b", "simplify_expr", {"expression": "N(16/(3*pi))"}, "Plummer virial radius GM^2/(2|W|)")
leaf("R_V_over_v_rms_units", "simplify_expr", {"expression": "N((16/(3*pi))/sqrt(3*pi/32))"},
     "half-diameter crossing time R/v with R = R_V")
leaf("r_half_over_v_rms_units", "simplify_expr", {"expression": "N((2^(2/3)-1)^(-1/2)/sqrt(3*pi/32))"},
     "half-diameter crossing time R/v with R = r_half")
leaf("HM_check_2RV_over_vrms", "simplify_expr", {"expression": "N(2*(16/(3*pi))/sqrt(3*pi/32))"},
     "should equal t_cr_HM_over_unit")
leaf("run_3trelax_in_trelax_RV", "simplify_expr", {"expression": "N(3/((16/(3*pi))/sqrt(3*pi/32)))"},
     "design's 3 t_relax run in relaxation times 0.1N/lnN x R_V/v_rms")
leaf("run_3trelax_in_trelax_rh", "simplify_expr", {"expression": "N(3/((2^(2/3)-1)^(-1/2)/sqrt(3*pi/32)))"},
     "design's 3 t_relax run in relaxation times 0.1N/lnN x r_half/v_rms")
leaf("run_3trelax_in_trelax_HM", "simplify_expr", {"expression": "N(3/(3*pi/32)^(-3/2))"},
     "design's 3 t_relax run in relaxation times 0.1N/lnN x Heggie-Mathieu t_cr")
leaf("N_over_8lnN_N128", "simplify_expr", {"expression": "N(128/(8*log(128)))"},
     "textbook derivation t_relax/t_cross = N/(8 lnN) with t_cross = R/v; 0.1N/lnN is its rounded form")
# ---- v2: window lengths against the circular-orbit period at r_half (design units) ----
leaf("T_orb_r_half_units", "simplify_expr", {"expression": "N(2*pi*sqrt(2*(2^(2/3)-1)^(-3/2)))"},
     "Kepler period at r_half around M/2, in design units; a proxy, not a measured breathing period")
leaf("ext_window_units", "simplify_expr", {"expression": "N((3749-2375)/300)"}, "extended window rows 2375..3749")
leaf("ext_window_over_T_orb", "simplify_expr",
     {"expression": "N(((3749-2375)/300)/(2*pi*sqrt(2*(2^(2/3)-1)^(-3/2))))"})
leaf("post_kick_run_over_T_orb", "simplify_expr",
     {"expression": "N((3749/300)/(2*pi*sqrt(2*(2^(2/3)-1)^(-3/2))))"})
leaf("pre_window_over_T_orb", "simplify_expr",
     {"expression": "N(((2374-1584)/300)/(2*pi*sqrt(2*(2^(2/3)-1)^(-3/2))))"})
# ---- v2: Plummer moment of inertia: integrand of I = int r^2 dM is x^4 (1+x^2)^(-5/2) (x = r/b) ----
leaf("plummer_I_integrand_large_x", "series_expansion",
     {"expression": "x^4/sqrt(1+x*x)^5", "variable": "x", "point": "oo", "order": 3},
     "~1/x at large x, so I of an untruncated Plummer sphere diverges logarithmically: halo-dominated")
leaf("plummer_I_partial_integral", "integrate_expr",
     {"expression": "x^4/sqrt(1+x*x)^5", "variable": "x", "lower": "0", "upper": "X"})
# ---- v2: small-sample statistics closed forms ----
leaf("chi2_2dof_q025", "simplify_expr", {"expression": "N(-2*log(0.975))"}, "chi-square 2 dof, 2.5 % quantile")
leaf("chi2_2dof_q975", "simplify_expr", {"expression": "N(-2*log(0.025))"}, "chi-square 2 dof, 97.5 % quantile")
leaf("F_2_6_survival", "integrate_expr", {"expression": "(1+x/3)^(-4)", "variable": "x", "lower": "f", "upper": "oo"},
     "F(2,6) pdf is (1+x/3)^(-4); survival = (1+f/3)^(-3)")
leaf("t2_pdf_norm", "integrate_expr", {"expression": "1/sqrt(2+x*x)^3", "variable": "x", "lower": "-oo", "upper": "oo"},
     "Student t with 2 dof: pdf = (2+x^2)^(-3/2)")
leaf("t2_survival", "integrate_expr", {"expression": "1/sqrt(2+x*x)^3", "variable": "x", "lower": "t", "upper": "oo"},
     "one-sided p for the seed-mean t statistic (2 dof)")
leaf("F_2_6_pdf_norm", "integrate_expr", {"expression": "(1+x/3)^(-4)", "variable": "x", "lower": "0", "upper": "oo"})

from sigma_ground.field.interface.statistical import heat_capacity_equipartition  # noqa: E402
out["C_gas_equipartition_J_per_K"] = dict(value=heat_capacity_equipartition(3, n=N),
                                          source="sigma_ground.field.interface.statistical.heat_capacity_equipartition(3, n=128)")
from sigma_ground.field.interface import nbody  # noqa: E402
out["engine_G_equals_mentat_G"] = bool(nbody._G == G)
v1 = json.load(open(os.path.join(HERE, "mentat_leaves.json"), encoding="utf-8"))
out["v1_leaves_identical"] = {k: str(v1[k]["value"]) == str(out[k]["value"]) for k in v1 if k in out and isinstance(v1[k], dict)}
print("v1 identical:", out["v1_leaves_identical"])
json.dump(out, open(os.path.join(HERE, "mentat_leaves_v2.json"), "w", encoding="utf-8"), indent=1)
