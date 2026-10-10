"""U7 — post-hoc diagnostic: the engine's SPH pressure operator linearised for plane waves on the s = 2h/3 lattice
(an infinite 1-D chain of lattice planes, no free surfaces, no viscosity), integrated with velocity Verlet at the
engine's dt, source plane displaced by 0.05 s. Applies the SAME arrival statistics as the engine runs, to separate
'what the statistic does on a dispersive pulse' from 'what the finite bar / viscosity / nonlinearity do'.
Operator: delta_rho_n = m sum_l G_l (u_n - u_{n-l}),  a_n = -(m K / rho0^3) sum_l G_l delta_rho_{n-l},
G_l = sum over the lattice plane at offset l of dW/dx (engine kernel grad_W). [DERIVED numerics, not the engine]"""
import json, math, sys
import numpy as np
sys.path.insert(0, r"D:/Aaron/development/sigma-ground")
sys.path.insert(0, r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U7")
from sigma_ground.dynamics.fluid.kernel import W, grad_W
from sigma_ground.dynamics.continuum import material_properties
from u7_sph_analyse import first_cross, first_peak, ols_loglog, model_compare
OUT = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U7"
s = 0.01; h = 1.5 * s
pr = material_properties("water_ice", 300.0); rho0, K, cL = pr["density_kg_m3"], pr["bulk_modulus_Pa"], pr["sound_speed_m_s"]
c_t = math.sqrt(K / rho0)
R = 4
m = rho0 / sum(W(s * math.sqrt(i * i + j * j + k * k), h) for i in range(-R, R + 1) for j in range(-R, R + 1) for k in range(-R, R + 1))
G = {l: sum(grad_W(l * s, j * s, k * s, h)[0] for j in range(-R, R + 1) for k in range(-R, R + 1)) for l in range(-R, R + 1) if l != 0}
N = 401; c0 = N // 2
dt = 0.3 * h / cL                     # the engine's cfl_timestep at t = 0 (c_max = cascade c_L)
def accel(u):
    drho = np.zeros(N)
    for l, g in G.items():
        drho += m * g * (u - np.roll(u, l))
    a = np.zeros(N)
    for l, g in G.items():
        a += -(m * K / rho0 ** 3) * g * np.roll(drho, l)
    return a
u = np.zeros(N); u[c0] = 0.05 * s; v = np.zeros(N)
nsteps = 50
V = np.zeros((nsteps + 1, N)); a = accel(u)
for n in range(1, nsteps + 1):
    u = u + v * dt + 0.5 * a * dt * dt
    a1 = accel(u); v = v + 0.5 * (a + a1) * dt; a = a1
    V[n] = v
t = np.arange(nsteps + 1) * dt
rows = {}
for d in range(1, 11):
    y = np.abs(V[:, c0 + d]); j = first_peak(y)
    a_, b_, cc = y[j - 1], y[j], y[j + 1]; den = a_ - 2 * b_ + cc
    tp = t[j] + (0.5 * (a_ - cc) / den * dt if den != 0 else 0.0)
    rows[d] = {"t_peak": float(tp), "t_0.5": first_cross(t[: j + 1], y[: j + 1], 0.5 * y[j]),
               "t_0.25": first_cross(t[: j + 1], y[: j + 1], 0.25 * y[j]), "t_0.1": first_cross(t[: j + 1], y[: j + 1], 0.1 * y[j])}
dd = np.arange(4, 11) * s; fits = {}
for key in ("t_0.1", "t_0.25", "t_0.5", "t_peak"):
    tt = np.array([rows[d][key] for d in range(4, 11)])
    lin = np.polyfit(tt, dd, 1)
    fits[key] = dict(p_loglog=ols_loglog(dd, tt), speed_over_c_tait=float(lin[0] / c_t), offset_d0_over_s=float(lin[1] / s),
                     rms_ratio_parabolic_over_hyperbolic=model_compare(dd, tt)["rms_ratio_parabolic_over_hyperbolic"])
out = dict(dt=dt, c_tait=c_t, G=G, rows=rows, fits=fits)
json.dump(out, open(f"{OUT}/sph_1d_linear.json", "w"), indent=1)
print(json.dumps(fits, indent=1))
