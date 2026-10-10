"""Feasibility probe P5 — negative heat capacity of a self-gravitating N-body system vs positive C_V of an ideal gas.
Two uniform spheres with isotropic velocities scaled to virial ratio Q = 2K/|U| = 1 at two total energies E1 < E2 (same mass, different radius).
After relaxing ~4 t_ff, T_eff := (2/3) <K>/(N k_B). Prediction (virial): T_eff ∝ -E  => (T2-T1)/(E2-E1) < 0.
Ideal gas (repo statistical.heat_capacity_equipartition): dT/dE = +1/((3/2) N k_B) > 0.
"""
import math, random, time, json, sys
import numpy as np
from sigma_ground.field.constants import G, K_B
from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem
from sigma_ground.field.interface.statistical import heat_capacity_equipartition
N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
M = 1.0e12
def make(R, seed):
    rng = random.Random(seed); m = M / N; pos = []; 
    for i in range(N):
        r = R * rng.random() ** (1/3); ct = 2*rng.random()-1; st = math.sqrt(1-ct*ct); ph = 2*math.pi*rng.random()
        pos.append([r*st*math.cos(ph), r*st*math.sin(ph), r*ct])
    pos = np.array(pos); pos -= pos.mean(0)
    # PE
    U = 0.0
    for i in range(N):
        for j in range(i+1, N):
            U -= G*m*m/np.linalg.norm(pos[i]-pos[j])
    vel = np.array([[rng.gauss(0,1) for _ in range(3)] for _ in range(N)]); vel -= vel.mean(0)
    K = 0.5*m*(vel**2).sum(); vel *= math.sqrt(0.5*abs(U)/K)   # virial: 2K = |U|
    bodies = [CelestialBody(mass_kg=m, position_m=pos[i], velocity_m_s=vel[i], radius_m=1.0, love_number_k2=0.0) for i in range(N)]
    return NBodySystem(bodies, softening_m=0.02*R), m
def ke(sys_): return sum(b.kinetic_energy() for b in sys_.bodies)
rows = []
for R in (5000.0, 10000.0):
    sysn, m = make(R, 11); rho = 3*M/(4*math.pi*R**3); t_ff = math.sqrt(3*math.pi/(32*G*rho)); dt = t_ff/300
    E0 = sysn.total_energy(); t0 = time.time(); Ks = []
    for k in range(int(4*300)):
        sysn.step(dt)
        if k >= 2*300: Ks.append(ke(sysn))
    E1 = sysn.total_energy(); Kbar = float(np.mean(Ks)); U = E1 - Kbar
    T_eff = (2/3)*Kbar/(N*K_B)
    rows.append(dict(R=R, E_J=E1, E_drift=abs(E1/E0-1), K_mean=Kbar, virial_2K_over_U=2*Kbar/abs(U), T_eff_K=T_eff, wall_s=round(time.time()-t0,1)))
    print(json.dumps(rows[-1]), flush=True)
dT_dE = (rows[1]['T_eff_K']-rows[0]['T_eff_K'])/(rows[1]['E_J']-rows[0]['E_J'])
C_grav = 1/dT_dE
C_gas = heat_capacity_equipartition(3, n=N)  # J/K for N monatomic particles (check repo semantics)
print(json.dumps(dict(dT_dE_gravity=dT_dE, C_gravity_J_per_K=C_grav, C_ideal_gas_J_per_K=C_gas, sign_gravity='NEGATIVE' if C_grav<0 else 'positive')))
json.dump(dict(rows=rows, dT_dE_gravity=dT_dE, C_gravity=C_grav, C_gas=C_gas), open(sys.argv[2],'w'), indent=1) if len(sys.argv)>2 else None
