import time, cProfile, pstats, io, numpy as np
from sigma_ground.dynamics.continuum import ContinuumParcel, ContinuumScene, continuum_step
from sigma_ground.dynamics.scene import GroundPlane
from sigma_ground.dynamics.vec import Vec3
N, h, L = 800, 0.4, 1.6
rng = np.random.default_rng(3)
ps = [ContinuumParcel('water_ice', 0.02, temperature=300.0,
      position=Vec3(rng.uniform(0, L), rng.exponential(1.0), rng.uniform(0, L)),
      velocity=Vec3(0, 0, 0), label=f'p{i}') for i in range(N)]
sc = ContinuumScene(ps, gravity=Vec3(0, -1.0, 0), ground=GroundPlane(), smoothing_h=h,
                    eos='isothermal', sound_speed_m_s=1.0)
sc.heat_conduction = False
pr = cProfile.Profile(); pr.enable()
t0 = time.perf_counter()
for _ in range(2):
    sc._vv_accel = None
    continuum_step(sc, 0.05)
print('per step', (time.perf_counter() - t0) / 2)
pr.disable(); s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumtime').print_stats(12); print(s.getvalue()[:3000])
print('mean neighbors', np.mean([len(p.neighbors) for p in ps]))
