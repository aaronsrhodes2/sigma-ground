"""U6 v2 (repair round) -- interpretation numbers requested by the skeptics, every one from Mentat (PLATINUM 21).
All [DERIVED], none of them RUN on the solver.  Writes physics_v2.json.  Run from the sigma-ground repo root.

Blocks
  I   energy/virial identity for an ISOLATED self-gravitating star (Newtonian bookkeeping, flat box):
        Q = int (eps - 3P) dV,  eps = rest + internal energy density (the trace of T uses the TOTAL energy density)
        int rho0 c^2 dV = M0 c^2 - K;  E = M0 c^2 + K + U + W;  scalar virial  Idd/2 = 2K + 3 Pi + W  (Pi = int P dV)
        =>  Q = E - Idd/2  and the far field  r sigma = (2 G xi / c^4) [Q + Idd/6] = (2 G xi/c^4)(E - Idd/3)
  II  heating at fixed rest density: d(eps - 3P) = dP/(Gamma - 1) - 3 dP = -3 dP (Gamma - 4/3)/(Gamma - 1)
  III homologous adiabatic one-zone check (Gamma symbolic) and the breathing frequency
  IV  metric weight (1 - 2 Phi/c^2) of design sec. 2.1 on the PRESCRIBED homologous mass mode (zeroth order in omega R/c)
  V   corrected air-sound arrival time; caveat numbers of v1 re-fetched
"""
import sys, os, json
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
from mentat_client import mentat

HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(os.path.join(HERE, "mentat_leaves_v2.json"), encoding="utf-8"))
c = float(L["c"]["value"]); w = L["choices"]["w_c"]; a = L["choices"]["a_m"]; eps = L["choices"]["eps"]
I2r = "((pi**3 - 6*pi)/pi**3)"
B = {}
def rec(k, tool, args):
    v, s = mentat(tool, args); B[k] = {"value": v, "source": s, "tool": tool, "args": args}; print(f"{k:48s} {v} | {s}", flush=True); return v

# ---- I. identity ----
rec("I_identity_Q", "simplify_expr", {"expression": "(M0*c**2 - K) + (E - M0*c**2 - K - W) - (Idd/2 - 2*K - W)"})
rec("I_far_field_isolated_flat_bracket", "simplify_expr", {"expression": "(E - Idd/2) + Idd/6"})
rec("I_far_field_isolated_flat_coefficient", "simplify_expr", {"expression": "(8*pi*G*xi/c**4)/(4*pi)*(-Idd/3)"})
rec("I_ratio_isolated_to_U6_mass_formula", "simplify_expr", {"expression": "(-Idd/3)/(Idd/6)"})
# covariant box (rho* bookkeeping, int sqrt(-g) rho0 c^2 = M* c^2 + 2W - K): adds 2 dW
rec("I_identity_Q_covariant", "simplify_expr", {"expression": "2*W + (M0*c**2 - K) + (E - M0*c**2 - K - W) - (Idd/2 - 2*K - W)"})

# ---- II. heating at fixed rest density ----
rec("II_heating_factor_symbolic", "simplify_expr", {"expression": "(dP/(Gam - 1) - 3*dP)/(-3*dP)"})
for g, gs in (("2", "2"), ("5/3", "5/3"), ("4/3", "4/3")):
    rec(f"II_heating_factor_Gamma_{g.replace('/', '_')}", "simplify_expr", {"expression": f"N((3*({gs}) - 4)/(3*(({gs}) - 1)), 15)"})

# ---- III. n = 1 potential, |W|, homologous one-zone check, breathing frequency ----
rec("III_poisson_check_n1_potential", "simplify_expr",
    {"expression": "diff(r**2*diff(-4*pi*G*rho_c*a**2*(1 + sin(r/a)/(r/a)), r), r)/r**2 - 4*pi*G*rho_c*sin(r/a)/(r/a)"})
rec("III_int_sinx_x_plus_sinx", "integrate_expr", {"expression": "sin(x)*(x + sin(x))", "variable": "x", "lower": "0", "upper": "pi"})
# W = (1/2) int rho Phi dV = -(1/2) 16 pi^2 G rho_c^2 a^5 (3 pi/2);  M c^2 = 4 pi^2 rho_c a^3 c^2;  rho_c = w c^2/(6 pi G a^2)
rec("III_absW_over_Mc2_symbolic", "simplify_expr",
    {"expression": "(8*pi**2*G*(w*c**2/(6*pi*G*a**2))**2*a**5*(3*pi/2))/(4*pi**2*(w*c**2/(6*pi*G*a**2))*a**3*c**2)"})
rec("III_virial_equilibrium_3Pi_minus_absW", "simplify_expr", {"expression": "3*(w/6) - w/2"})
rec("III_homol_dQ_dlam", "simplify_expr",
    {"expression": "diff((absW/(3*(Gam - 1)))*lam**(-3*(Gam - 1))*(4 - 3*Gam), lam).subs(lam, 1)"})
rec("III_homol_d_halfIdd_dlam", "simplify_expr",
    {"expression": "diff(3*(Gam - 1)*(absW/(3*(Gam - 1)))*lam**(-3*(Gam - 1)) - absW/lam, lam).subs(lam, 1)"})
rec("III_homol_dE_dlam", "simplify_expr",
    {"expression": "diff((absW/(3*(Gam - 1)))*lam**(-3*(Gam - 1)) - absW/lam, lam).subs(lam, 1)"})
rec("III_breathing_omegaR_over_c_Gamma2", "simplify_expr", {"expression": f"N(sqrt((3*2 - 4)*({w}/2)/{I2r}), 15)"})
rec("III_breathing_over_cs_centre", "simplify_expr", {"expression": f"N(sqrt((3*2 - 4)*({w}/2)/{I2r})/sqrt(2*{w}/3), 15)"})
# isolated star at its own breathing frequency, amplitude in units of xi r_s eps
rec("III_amp_isolated_flat_over_xirs_eps", "simplify_expr", {"expression": f"N((2/3)*(3*2 - 4)*({w}/2), 15)"})
rec("III_amp_isolated_cov_over_xirs_eps", "simplify_expr", {"expression": f"N(2*({w}/2) + (2/3)*(3*2 - 4)*({w}/2), 15)"})
rec("III_amp_press_design_over_xirs_eps", "simplify_expr", {"expression": f"N({w}/2, 15)"})
rec("III_isolated_flat_over_Apress", "simplify_expr", {"expression": f"N((2/3)*(3*2 - 4)*({w}/2)/({w}/2), 15)"})
rec("III_isolated_cov_over_Apress", "simplify_expr", {"expression": f"N((2*({w}/2) + (2/3)*(3*2 - 4)*({w}/2))/({w}/2), 15)"})

# ---- IV. metric weight on the prescribed homologous mass mode ----
# rho* bookkeeping: d int sqrt(-g) rho0 c^2 dV = 2 dW = 2|W| eps sin  ->  amplitude 2 (|W|/Mc^2) eps xi r_s = w eps xi r_s
# weight on the flat-volume U6 density held homologous: d int (-2 Phi rho) dV = -4 dW  ->  2 w eps xi r_s
rec("IV_metric_rhostar_over_Apress", "simplify_expr", {"expression": f"N((2*({w}/2))/({w}/2), 15)"})
rec("IV_metric_flatrho_over_Apress", "simplify_expr", {"expression": f"N((4*({w}/2))/({w}/2), 15)"})
for q in (0.01, 0.02, 0.04, 0.08):
    rec(f"IV_metric_rhostar_over_Amass_wR{q}", "simplify_expr", {"expression": f"N((2*({w}/2))/({q}**2*{I2r}/3), 15)"})
    rec(f"IV_metric_flatrho_over_Amass_wR{q}", "simplify_expr", {"expression": f"N((4*({w}/2))/({q}**2*{I2r}/3), 15)"})

# ---- V. air-sound arrival (v1 said 1.3 s) and caveat numbers ----
cs_air = float(L["cs_air_293K_m_s"]["value"])
rec("V_t_sonic_air_s", "simplify_expr", {"expression": f"N(29*pi*{a}/{cs_air}, 15)"})
rec("V_t_run_6T_wR0.01_s", "simplify_expr", {"expression": f"N(6*2*pi*pi*{a}/(0.01*{c}), 15)"})
rec("V_crossover_omegaR_over_c", "simplify_expr", {"expression": f"N(sqrt(3*{w}/(2*{I2r})), 15)"})
rec("V_crossover_over_cs_centre", "simplify_expr", {"expression": f"N(sqrt(3*{w}/(2*{I2r}))/sqrt(2*{w}/3), 15)"})
rec("V_omega_cut_s-1_lambda_34p5km", "simplify_expr", {"expression": f"N({c}/34500, 15)"})
rec("V_omega_max_swept_s-1", "simplify_expr", {"expression": f"N(0.08*{c}/(pi*{a}), 15)"})
rec("V_omega_min_swept_s-1", "simplify_expr", {"expression": f"N(0.01*{c}/(pi*{a}), 15)"})
rec("V_yukawa_at_30R_lambda_34p5km", "simplify_expr", {"expression": f"N(exp(-30*pi*{a}/34500), 6)"})

json.dump(B, open(os.path.join(HERE, "physics_v2.json"), "w", encoding="utf-8"), indent=1)
print("wrote physics_v2.json")
