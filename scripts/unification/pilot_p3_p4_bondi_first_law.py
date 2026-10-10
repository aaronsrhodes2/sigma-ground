"""Pilots P3 + P4 (leaf quantities via Mentat where a tool exists; derivation via sympy).

P3 — Bondi (1952) isothermal accretion: the sonic point is an acoustic horizon (Unruh 1981, Visser 1998).
     r_sonic = GM/(2 c_s^2);  du/dx|_sonic = -1 on the transonic branch (L'Hopital, checked with sympy);
     kappa_acoustic = |d(c_s - v)/dr| = c_s / r_sonic = 2 c_s^3 / (G M);  T_analog = hbar kappa / (2 pi k_B)
     => T_analog / T_Hawking = 8 (c_s / c)^3   (Schwarzschild: kappa = c^3/(4GM), T_H = hbar c^3/(8 pi G M k_B)).
P4 — First law of black-hole mechanics dE = T dS and the Smarr relation E = 2TS; Verlinde (2011) entropic chain.
"""
import json, math, os, subprocess, sys
import sympy as sp
from sigma_ground.field.constants import G, C, HBAR, K_B, L_PLANCK
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tools'))
from mentat_client import mentat   # PLATINUM §21: leaf quantities from Mentat (skippytel-mentat container)

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
M = 1.989e30
T_H, src_T = mentat("hawking_temperature", {"mass_kg": M})
r_s, src_r = mentat("schwarzschild_radius", {"mass_kg": M})
S_bh, src_S = mentat("bekenstein_hawking_entropy", {"mass_kg": M})
S_bh = S_bh["entropy_k_B"] * K_B if isinstance(S_bh, dict) else S_bh   # tool returns S/k_B

# --- P3: sympy check of the Bondi transonic slope ---
u, x = sp.symbols('u x', positive=True); s = sp.symbols('s')  # slope may be negative
F = u**2/2 - sp.log(u) - 2*sp.log(x) - 2/x + sp.Rational(3, 2)
quad = sp.expand((sp.diff(F, u, 2)*s**2 + 2*sp.diff(F, u, x)*s + sp.diff(F, x, 2)).subs({u: 1, x: 1}))
slopes = sp.solve(quad, s)
rows = []
for cs in (1e3, 1e4, 1e5):
    r_son = G*M/(2*cs*cs); kappa = cs/r_son; T_ac = HBAR*kappa/(2*math.pi*K_B)
    rows.append(dict(c_s=cs, r_sonic_m=r_son, kappa_1_s=kappa, T_analog_K=T_ac, ratio=T_ac/T_H, predicted_8_cs_c_cubed=8*(cs/C)**3))
# --- P4: first law + Smarr + Verlinde (all with repo constants; T_H, r_s, S from Mentat) ---
# first law with the ANALYTIC derivative dS/dM = 2S/M (S is exactly quadratic in M); a central difference at
# h = 1e-6 M only reports float cancellation (1 - 6.9e-11), not physics.
T_first_law = C**2 / (2 * S_bh / M)
smarr = 2*T_H*S_bh/(M*C**2)
m, r = 1.0, 1.496e11
N = 4*math.pi*r*r/L_PLANCK**2                       # Verlinde bit count N = A/l_P^2 (with A/(4 l_P^2) the ratio below is 4: a convention)
T_screen = 2*M*C**2/(N*K_B)                         # equipartition screen temperature (NOT the Unruh temperature)
F_v = T_screen*(2*math.pi*K_B*m*C/HBAR); F_n = G*M*m/r**2
a_r = G*M/r**2
T_unruh, src_unruh = mentat("unruh_temperature", {"acceleration_m_s2": a_r})   # Mentat leaf; F_v/F_n == T_screen/T_unruh identically
out = dict(sources={"T_H": src_T, "r_s": src_r, "S_BH": src_S},
           bondi_slopes_at_sonic_point=[str(v) for v in slopes], bondi_rows=rows,
           first_law_T_over_T_H=T_first_law/T_H, smarr_2TS_over_Mc2=smarr, verlinde_F_over_Newton=F_v/F_n,
           unruh_over_screen_T=T_unruh/T_screen, unruh_source=src_unruh,
           note="all four are consistency checks by construction (one constants set; L_PLANCK = sqrt(hbar G/c^3)); rows 3 and 4 are one identity")
print(json.dumps(out, indent=1))
