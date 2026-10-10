"""Stage 5 — L3, same dictionary, turned from [LIT] into a run: does a fluid falling onto a mass make a horizon that sound
cannot escape?

NEW UTILITY (labelled, per the program's rule): a 1-D spherical, isothermal, finite-volume Euler solver with a point-mass
gravity source. The repo's SPH engine cannot resolve a transonic radial profile at affordable N, so this ~70-line solver
stands in; every leaf quantity still comes from Mentat (G, a solar mass), and the predictions it is scored against are
Bondi 1952's exact ones:
    r_sonic = GM / 2c_s²,    accretion eigenvalue λ_c = e^{3/2}/4 = 1.1204 (γ = 1),    Ṁ = 4π λ_c (GM)² ρ_∞ / c_s³.

Equations (spherical, isothermal P = ρc_s²), conservative form in u = (ρr², ρvr²):
    ∂_t(ρr²) + ∂_r(ρvr²) = 0
    ∂_t(ρvr²) + ∂_r(ρv²r² + Pr²) = 2Pr − GMρ
HLL fluxes (isothermal wave speeds v ± c_s), second-order MUSCL with minmod, SSP-RK2. Outer boundary held at (ρ_∞, v = 0);
inner boundary is supersonic outflow (zero-gradient), so nothing comes back from it.

Experiment, decided before the run:
  1. relax from rest to the steady state; pass if r_sonic and λ_c match Bondi to < 3 %;
  2. inject a small outward-moving acoustic pulse (δρ/ρ = 1e-3, Gaussian) at r₀ = 0.5 r_sonic and at r₀ = 2 r_sonic on the
     relaxed flow; track the pulse's leading characteristic. PASS if the inside pulse never reaches r_sonic (swept inward)
     and the outside pulse reaches 4 r_sonic. FAIL (no horizon) if the inside pulse crosses r_sonic outward.
Units: everything is scaled by r_sonic and c_s (the flow is self-similar in those), with GM fixed by Mentat's solar mass —
the dimensional numbers are reported for c_s = 1e4 m/s.
"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from mentat_client import mentat

G, G_src = mentat("lookup_constant", {"name": "G"})
sun, sun_src = mentat("solar_system_body", {"body_name": "sun"})
M = sun["mass_kg"]; CS = 1.0e4
GM = G * M; R_SON = GM / (2 * CS * CS); LAM_C = math.exp(1.5) / 4

# dimensionless: r in r_sonic, v in c_s, t in r_sonic/c_s  →  GM = 2 (since r_son = GM/2c² = 1)
GMd, cs = 2.0, 1.0
r_in, r_out, NR = 0.15, 12.0, 600
edges = np.geomspace(r_in, r_out, NR + 1); r = np.sqrt(edges[:-1] * edges[1:]); dr = np.diff(edges)

def prim(U):
    rho = U[0] / r ** 2; v = U[1] / np.maximum(U[0], 1e-300); return rho, v

def minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(abs(a), abs(b)), 0.0)

def rhs(U):
    rho, v = prim(U)
    # ghost cells: inner zero-gradient (supersonic inflow leaves), outer fixed ambient
    rho_g = np.concatenate(([rho[0], rho[0]], rho, [1.0, 1.0])); v_g = np.concatenate(([v[0], v[0]], v, [0.0, 0.0]))
    sl_r = minmod(rho_g[1:-1] - rho_g[:-2], rho_g[2:] - rho_g[1:-1]); sl_v = minmod(v_g[1:-1] - v_g[:-2], v_g[2:] - v_g[1:-1])
    rc, vc = rho_g[1:-1], v_g[1:-1]
    rL, vL = (rc + 0.5 * sl_r)[:-1], (vc + 0.5 * sl_v)[:-1]       # left state at each interface
    rR, vR = (rc - 0.5 * sl_r)[1:], (vc - 0.5 * sl_v)[1:]         # right state
    e2 = np.concatenate(([edges[0]], edges, [edges[-1]]))[1:-1] ** 2
    fL = np.array([rL * vL * e2, (rL * vL * vL + rL * cs * cs) * e2]); fR = np.array([rR * vR * e2, (rR * vR * vR + rR * cs * cs) * e2])
    uL = np.array([rL * e2, rL * vL * e2]); uR = np.array([rR * e2, rR * vR * e2])
    sL = np.minimum(vL - cs, vR - cs); sR = np.maximum(vL + cs, vR + cs)
    F = np.where(sL >= 0, fL, np.where(sR <= 0, fR, (sR * fL - sL * fR + sL * sR * (uR - uL)) / np.maximum(sR - sL, 1e-300)))
    dU = -(F[:, 1:] - F[:, :-1]) / dr
    dU[1] += 2 * rho * cs * cs * r - GMd * rho
    return dU

def step(U, dt):
    U1 = U + dt * rhs(U); return 0.5 * (U + U1 + dt * rhs(U1))

rho0 = np.ones(NR); U = np.array([rho0 * r ** 2, np.zeros(NR)])
dt = 0.3 * dr.min() / (cs + 3.0)
t, T_relax = 0.0, 60.0
while t < T_relax:
    U = step(U, dt); t += dt
rho, v = prim(U)
mach = -v / cs
i_son = int(np.argmin(abs(mach - 1.0)))
r_son_num = float(r[i_son])
mdot = float(np.median((-4 * math.pi * rho * v * r ** 2)[(r > 0.5) & (r < 5)]))
lam_num = mdot / (4 * math.pi * GMd ** 2)          # ρ_∞ = c_s = 1
steady = dict(r_sonic_num_over_bondi=r_son_num, lambda_c_num=lam_num, lambda_c_bondi=LAM_C, lambda_rel_err=lam_num / LAM_C - 1,
              mdot_spread=float(np.std((-4 * math.pi * rho * v * r ** 2)[(r > 0.5) & (r < 5)]) / mdot))

def pulse_run(r0, amp=1e-3, width=0.04, T=6.0):
    """Superpose an outgoing acoustic pulse (δρ = amp·ρ·g, δv = +c_s·amp·g, the outgoing characteristic) and track it."""
    U0 = U.copy(); rh, vh = prim(U0)
    gss = np.exp(-0.5 * ((r - r0) / width) ** 2)
    rp = rh * (1 + amp * gss); vp = vh + cs * amp * gss
    Up = np.array([rp * r ** 2, rp * vp * r ** 2]); Ub = U0.copy()
    tt, track = 0.0, []
    while tt < T:
        Up = step(Up, dt); Ub = step(Ub, dt); tt += dt
        if len(track) == 0 or tt - track[-1][0] > 0.02:
            dr_ = prim(Up)[0] / prim(Ub)[0] - 1
            k = int(np.argmax(dr_)); track.append((tt, float(r[k]), float(dr_[k])))
    rmax = max(p[1] for p in track); rlast = track[-1][1]
    return dict(r0=r0, max_radius_reached=rmax, final_radius=rlast, track=track[::5])

inside = pulse_run(0.5); outside = pulse_run(2.0)
verdict = dict(inside_escaped=inside["max_radius_reached"] > 1.0, outside_reached_4=outside["max_radius_reached"] > 4.0)
kappa = CS / R_SON
out = dict(sources=dict(G=G_src, sun=sun_src), dimensional=dict(c_s=CS, M=M, r_sonic_m=R_SON, kappa_1_s=kappa,
           T_analog_K_from_mentat=mentat("unruh_temperature", {"acceleration_m_s2": kappa * 299792458.0})[0]),
           steady=steady, pulses=dict(inside=inside, outside=outside), verdict=verdict,
           profile=dict(r=r.tolist()[::10], mach=mach.tolist()[::10], rho=rho.tolist()[::10]))
os.makedirs("misc/unification", exist_ok=True)
json.dump(out, open("misc/unification/stage5_sonic_horizon.json", "w"), indent=1)
print(json.dumps(dict(steady=steady, verdict=verdict, inside_max=inside["max_radius_reached"], outside_max=outside["max_radius_reached"],
                      dimensional=out["dimensional"]), indent=1))
