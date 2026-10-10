"""U13 v2 analytic rows, every number through Mentat (integrate_expr / simplify_expr / constants).

1. Continuum identity for the ENGINE's AV form (sigma_ground/dynamics/continuum.py:556-605, Monaghan 1992 mu-form,
   no beta term, eta^2 = 0.01 h^2) under pure homologous compression v = -k r:
     mu_ij = -k h r^2/(r^2+eta^2),  Pi_ij = alpha c_s k h r^2/(r^2+eta^2)/rho,  pressure pair term 2 c_s^2/rho
     => V_visc/V_press = (alpha k h / 2 c_s) f_eta,  f_eta = int q^3 W'(q) q^2/(q^2+0.01) dq / int q^3 W'(q) dq
     => C = zeta/(rho alpha c_s h) = f_eta / 6      (1/6 exactly as eta -> 0, for any kernel)
2. Signal-velocity form Pi = -alpha c_s (v_ij . rhat)/rho (Monaghan 1997 with v_sig ~ 2 c_s and its 1/2):
     C = (2/9) <q>_W, <q>_W = int 4 pi q^3 W dq (M4 kernel).  Context for Price 2012's "zeta ~ alpha v_sig h/6".
3. Early free-fall expectation for the ratio:  k(t) = (G M/R^3) t = (pi^2/8) t/t_ff^2 (uniform sphere),
     ratio = alpha (pi^2/16) h/(c_s t_ff) (t/t_ff) f_eta.
4. Thermal virial ratio 2K_th/|W| = 5 c_s^2 R/(G M);  3P/(rho c^2) = 3 c_s^2/c^2;  Knudsen number of the iron swarm.
5. Time-step alternatives for the flagged cfl_timestep defect: 0.3 h/c_s, 0.3 sqrt(h/g_edge), 0.25 sqrt(h/g_edge).
Kernel: M4 cubic spline, sigma_ground/dynamics/fluid/kernel.py, W = (1/(pi h^3)) (1 - 3q^2/2 + 3q^3/4) for q<1,
(1/(4 pi h^3)) (2-q)^3 for 1<=q<2.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
from mentat_client import mentat  # noqa: E402

SRC = {}


def leaf(key, tool, args):
    v, s = mentat(tool, args)
    SRC[key] = {"tool": tool, "args": args, "value": v, "source": s}
    return v


def num(key, expr):
    return float(leaf(key, "simplify_expr", {"expression": f"N({expr})"}))


Win, Wout = "(1/pi)*(1 - 3*q**2/2 + 3*q**3/4)", "(1/(4*pi))*(2 - q)**3"
dWin, dWout = "(1/pi)*(-3*q + 9*q**2/4)", "-(3/(4*pi))*(2 - q)**2"


def integ(key, inner, outer):
    a = leaf(key + "_q0_1", "integrate_expr", {"expression": inner, "variable": "q", "lower": "0", "upper": "1"})
    b = leaf(key + "_q1_2", "integrate_expr", {"expression": outer, "variable": "q", "lower": "1", "upper": "2"})
    return num(key, f"({a}) + ({b})")


res = {}
res["kernel_norm"] = integ("norm", f"4*pi*q**2*{Win}", f"4*pi*q**2*{Wout}")
res["I_press"] = integ("I_press", f"4*pi*q**3*{dWin}", f"4*pi*q**3*{dWout}")
res["I_visc_eta"] = integ("I_visc_eta", f"4*pi*q**3*{dWin}*q**2/(q**2+1/100)",
                          f"4*pi*q**3*{dWout}*q**2/(q**2+1/100)")
res["mean_q_W"] = integ("mean_q", f"4*pi*q**3*{Win}", f"4*pi*q**3*{Wout}")
res["f_eta"] = num("f_eta", f"{res['I_visc_eta']!r}/{res['I_press']!r}")
res["C_engine_mu_form"] = num("C_mu", f"{res['f_eta']!r}/6")
res["C_engine_mu_form_eta_to_0"] = num("C_mu_eta0", "1/6")
res["C_vsig_form"] = num("C_vsig", f"(2/9)*{res['mean_q_W']!r}")

G = leaf("G", "lookup_constant", {"name": "G"})
c = leaf("c", "lookup_constant", {"name": "c"})
M, R, N = 1.0e12, 5000.0, 64
vref = leaf("v_ref", "circular_orbit_velocity", {"central_mass_kg": M, "radius_m": R})
rho_fe = leaf("rho_iron", "density", {"material": "iron"})
cs = 0.40 * vref
h = 1.5 * (4.0 / 3.0 * math.pi * R ** 3 / N) ** (1.0 / 3.0)
tff = num("t_ff", f"sqrt(3*pi/(32*{G!r}*(3*{M!r}/(4*pi*{R!r}**3))))")
res.update(c_s_m_s=cs, h0_m=h, t_ff_s=tff)
res["freefall_ratio_per_t_over_tff"] = num("ff_coeff", f"(pi**2/16)*{h!r}/({cs!r}*{tff!r})*{res['f_eta']!r}")
res["freefall_ratio_step50_dt_tff_over_600"] = num("ff_step50", f"{res['freefall_ratio_per_t_over_tff']!r}*50/600")
res["thermal_virial_ratio_2K_over_W"] = num("thermal_virial", f"5*{cs!r}**2*{R!r}/({G!r}*{M!r})")
res["three_P_over_rho_c2"] = num("threeP_rho_c2", f"3*{cs!r}**2/{c!r}**2")
mp = M / N
a = num("particle_radius_m", f"(3*{mp!r}/(4*pi*{rho_fe!r}))**(1/3)")
mfp = num("mean_free_path_m", f"(4*pi*{R!r}**3/3)/({N}*pi*(2*{a!r})**2)")
res.update(particle_radius_m=a, mean_free_path_m=mfp, Knudsen=mfp / R, Knudsen_with_sqrt2=mfp / R / math.sqrt(2))
g_edge = num("g_edge", f"{G!r}*{M!r}/{R!r}**2")
C_MAT = json.load(open(os.path.join(HERE, "u13_results.json"), encoding="utf-8"))["material_sound_speed_iron_m_s"]
res["c_material_iron_from_engine_m_s"] = C_MAT          # parcel.sound_speed set by the engine's interface cascade
res["dt_cfl_material_s"] = num("dt_cfl_material", f"0.3*{h!r}/{C_MAT!r}")
res["dt_cs_only_s"] = num("dt_cs_only", f"0.3*{h!r}/{cs!r}")
res["dt_accel_0p3_s"] = num("dt_accel_0p3", f"0.3*sqrt({h!r}/{g_edge!r})")
res["dt_accel_0p25_s"] = num("dt_accel_0p25", f"0.25*sqrt({h!r}/{g_edge!r})")
for k in ("dt_cfl_material_s", "dt_cs_only_s", "dt_accel_0p3_s", "dt_accel_0p25_s"):
    res[k[:-2] + "_over_tff"] = res[k] / tff
res["dt_used_tff_over_600_over_dt_cfl"] = (tff / 600) / res["dt_cfl_material_s"]
res["dt_cs_only_over_dt_cfl"] = res["dt_cs_only_s"] / res["dt_cfl_material_s"]
res["design_100_steps_with_cs_only_cfl_span_tff"] = 100 * res["dt_cs_only_s"] / tff
res["mentat_sources"] = SRC
with open(os.path.join(HERE, "u13_derive_v2.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k != "mentat_sources"}, indent=1))
