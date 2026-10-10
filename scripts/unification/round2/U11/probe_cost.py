"""Timing / feasibility probe for U11 (no physics claims; measures engine cost and cfl_timestep)."""
import time, json, numpy as np
from sigma_ground.dynamics.continuum import ContinuumParcel, ContinuumScene, continuum_step, cfl_timestep
from sigma_ground.dynamics.scene import GroundPlane
from sigma_ground.dynamics.vec import Vec3
cs, g, H = 1.0, 1.0, 1.0
out = {}
for N in (128, 256):
    rng = np.random.default_rng(3)
    ps = [ContinuumParcel('water_ice', 0.02, temperature=300.0,
          position=Vec3(rng.uniform(0, 1), rng.uniform(0, 4), rng.uniform(0, 1)),
          velocity=Vec3(0, 0, 0), label=f'p{i}') for i in range(N)]
    sc = ContinuumScene(ps, gravity=Vec3(0, -g, 0), ground=GroundPlane(), smoothing_h=0.1 * H,
                        eos='isothermal', sound_speed_m_s=cs)
    t0 = time.perf_counter(); dt = cfl_timestep(sc); t_cfl = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(5): continuum_step(sc, dt)
    t_step = (time.perf_counter() - t0) / 5
    sc.heat_conduction = False
    t0 = time.perf_counter()
    for _ in range(5): continuum_step(sc, dt)
    t_step_nc = (time.perf_counter() - t0) / 5
    t0 = time.perf_counter()
    for p in sc.continuum_parcels(): p.update_properties()
    t_up = time.perf_counter() - t0
    out[N] = dict(cfl_dt=dt, material_sound_speed=ps[0].sound_speed, t_cfl=t_cfl, t_step=t_step,
                  t_step_noconduction=t_step_nc, t_update_props_all=t_up, mass=ps[0].mass)
print(json.dumps(out, indent=1))
