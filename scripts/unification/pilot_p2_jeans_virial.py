"""Feasibility probe — Jeans/virial: SPH self-gravitating isothermal cloud, sweep c_s across the virial threshold.
Isothermal virial theorem (P = rho c_s^2): 2K_kin + 3 M c_s^2 + U = 0, U = -(3/5) G M^2 / R for a uniform sphere
=> support when c_s^2 >~ G M / (5 R)  i.e. f = c_s / sqrt(GM/R) >~ 1/sqrt(5) = 0.447.
"""
import math, time, json, sys
from sigma_ground.dynamics.seeding import seed_rotating_cloud, to_continuum, G_CODATA as G
from sigma_ground.dynamics.continuum import continuum_step
from sigma_ground.dynamics.phase_e import diagnostics, _sph_state

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
F_LIST = [float(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
M, R = 1.0e12, 5000.0
rho = 3 * M / (4 * math.pi * R**3)
t_ff = math.sqrt(3 * math.pi / (32 * G * rho))
v_ref = math.sqrt(G * M / R)
results = []
for f in F_LIST:
    bodies = seed_rotating_cloud(N, cloud_mass_kg=M, cloud_radius_m=R, omega_rad_s=0.0,
                                 particle_density_kg_m3=920.0, jitter_frac=0.02, seed=7)
    cs = f * v_ref
    h_res = 1.5 * (4/3*math.pi*R**3/N)**(1/3)   # resolve the cloud: ~30 kernel neighbours, not the packing radius
    scene = to_continuum(bodies, material_key='water_ice', G=G, gravity_softening_m=0.02 * R,
                         eos='isothermal', sound_speed_m_s=cs,
                         smoothing_h=h_res)
    dt = t_ff / 400.0; n_steps = 800   # 2 free-fall times
    d0 = diagnostics(*_sph_state(scene), G)
    from sigma_ground.dynamics.continuum import _sph_density_sum
    _sph_density_sum(scene.continuum_parcels(), scene.h)
    nnb = sum(len(p.neighbors) for p in scene.continuum_parcels())/N
    t0 = time.time()
    for k in range(n_steps):
        continuum_step(scene, dt)
    d1 = diagnostics(*_sph_state(scene), G)
    K_kin = d1['ke_j']; U = d1['pe_j']; therm = 3 * M * cs * cs
    U0_analytic = 0.6 * G * M * M / R
    row = dict(f=f, cs=cs, h=h_res, mean_neighbours=nnb, U0_discrete_over_analytic=abs(d0['pe_j']) / U0_analytic,
               thermal_over_U0_discrete=3 * M * cs * cs / abs(d0['pe_j']), r_med_ratio=d1['r_med_m'] / d0['r_med_m'],
               r_max_ratio=d1['r_max_m'] / d0['r_max_m'],
               virial_kin=2 * K_kin / abs(U), virial_thermal=therm / abs(U),
               virial_total=(2 * K_kin + therm) / abs(U), wall_s=round(time.time() - t0, 1))
    print(json.dumps(row), flush=True)
    results.append(row)
json.dump(results, open(sys.argv[2], 'w'), indent=1) if len(sys.argv) > 2 else None
