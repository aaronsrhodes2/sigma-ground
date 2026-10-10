"""Pilot P2b — the virial identity measured directly: ½ d²I/dt² = 2K + 3∫P dV + U at t → 0.
For a cold isothermal uniform cloud (K = 0, P = ρ c_s²): sign(d²I/dt²) = sign(3 M c_s² − |U₀|), with |U₀| the
DISCRETE potential energy of the realisation (measured, not the analytic 3GM²/5R). No free parameter.
Usage: python scripts/unification/pilot_p2b_virial_curvature.py [N] [f1,f2,...] [n_steps]
"""
import json, math, sys
import numpy as np
from sigma_ground.dynamics.seeding import seed_rotating_cloud, to_continuum, G_CODATA as G
from sigma_ground.dynamics.continuum import continuum_step, _sph_density_sum
from sigma_ground.dynamics.phase_e import diagnostics, _sph_state

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
F_LIST = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0.40, 0.45, 0.50, 0.55]
N_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
M, R = 1.0e12, 5000.0
rho = 3 * M / (4 * math.pi * R ** 3)
t_ff = math.sqrt(3 * math.pi / (32 * G * rho))
v_ref = math.sqrt(G * M / R)
h_res = 1.5 * (4 / 3 * math.pi * R ** 3 / N) ** (1 / 3)

def moment_of_inertia(scene):
    ms, ps, _ = _sph_state(scene)
    ps = np.array(ps); ms = np.array(ms); c = (ms[:, None] * ps).sum(0) / ms.sum()
    return float((ms * ((ps - c) ** 2).sum(1)).sum())

rows = []
for f in F_LIST:
    bodies = seed_rotating_cloud(N, cloud_mass_kg=M, cloud_radius_m=R, omega_rad_s=0.0,
                                 particle_density_kg_m3=920.0, jitter_frac=0.0, seed=7)   # cold: K(0) = 0 exactly
    cs = f * v_ref
    scene = to_continuum(bodies, material_key='water_ice', G=G, gravity_softening_m=0.02 * R,
                         eos='isothermal', sound_speed_m_s=cs, smoothing_h=h_res)
    d0 = diagnostics(*_sph_state(scene), G); U0 = abs(d0['pe_j'])
    dt = t_ff / 2000.0; n = N_STEPS                  # first 0.05 t_ff (100 steps), finely sampled
    ts, Is = [0.0], [moment_of_inertia(scene)]
    for k in range(n):
        continuum_step(scene, dt); ts.append((k + 1) * dt); Is.append(moment_of_inertia(scene))
    ts, Is = np.array(ts), np.array(Is)
    a, b, c0 = np.polyfit(ts, Is, 2)                 # I(t) ≈ I0 + ½ I'' t²  →  I'' = 2a
    I_dd = 2 * a
    predicted = 2 * (3 * M * cs * cs - U0)           # d²I/dt² = 2(2K + 3∫P dV + U) with K = 0, ∫P dV = M c_s²
    rows.append(dict(f=f, N=N, thermal_over_U0=3 * M * cs * cs / U0, U0_discrete_over_analytic=U0 / (0.6 * G * M * M / R),
                     I_ddot_measured=I_dd, I_ddot_virial=predicted, ratio=I_dd / predicted if predicted else None,
                     sign_agrees=bool(np.sign(I_dd) == np.sign(predicted)), fit_rel_resid=float(np.std(Is - np.polyval([a, b, c0], ts)) / np.ptp(Is))))
    print(json.dumps(rows[-1]), flush=True)
