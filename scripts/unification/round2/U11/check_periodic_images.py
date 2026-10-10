"""Sanity check of the driver-side 3x3 lateral image tiling used by u11_sph.py --mode periodic3d.
A lateral lattice (4 x 4 per layer, spacing h, L = 4h) with jittered layers in y: with exact lateral periodicity
every parcel of a layer must get the same SPH density and zero lateral acceleration (to roundoff), and a
lateral translation of the whole set by an arbitrary offset must not change any density."""
import json, math
import numpy as np
from sigma_ground.dynamics.continuum import ContinuumParcel, ContinuumScene, continuum_step
from sigma_ground.dynamics.vec import Vec3

h = 0.4; L = 4 * h; n = 4; sp = L / n


def make(offset):
    ps = []
    for iy, y in enumerate([0.5, 0.83, 1.21, 1.55, 1.9]):
        for ix in range(n):
            for iz in range(n):
                ps.append(ContinuumParcel('water_ice', 0.02, 300.0,
                                          position=Vec3((ix * sp + offset[0]) % L, y, (iz * sp + offset[1]) % L),
                                          velocity=Vec3(0, 0, 0)))
    sc = ContinuumScene(ps, gravity=Vec3(0, 0, 0), ground=False, smoothing_h=h, eos='isothermal', sound_speed_m_s=1.0)
    sc.energy_equation = False
    real = list(ps)
    imgs = []
    for sx in (-1, 0, 1):
        for sz in (-1, 0, 1):
            if (sx, sz) == (0, 0):
                continue
            for p in real:
                q = ContinuumParcel.__new__(ContinuumParcel)
                q.__dict__.update(p.__dict__)
                q.position = Vec3(p.position.x + sx * L, p.position.y, p.position.z + sz * L)
                q.velocity = Vec3(0, 0, 0); q.neighbors = []
                imgs.append(q)
    sc.parcels = real + imgs
    continuum_step(sc, 1e-9)
    acc = sc._vv_accel[:len(real)]
    rho = np.array([p.sph_density for p in real]).reshape(5, n * n)
    ax = np.array([abs(v.x) for v in acc]); az = np.array([abs(v.z) for v in acc]); ay = np.array([v.y for v in acc]).reshape(5, n * n)
    return rho, ax, az, ay


r0, ax0, az0, ay0 = make((0.0, 0.0))
r1, ax1, az1, ay1 = make((0.137, 0.291))
out = dict(layer_rho_spread_rel=float(np.max((r0.max(1) - r0.min(1)) / r0.mean(1))),
           max_lateral_acc=float(max(ax0.max(), az0.max())), typical_vertical_acc=float(np.abs(ay0).mean()),
           translation_rho_change_rel=float(np.max(np.abs(np.sort(r1, 1) - np.sort(r0, 1)) / r0)),
           layer_ay_spread=float(np.max(ay0.max(1) - ay0.min(1))))
print(json.dumps(out, indent=1))
json.dump(out, open(r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U11/check_periodic_images.json", 'w'), indent=1)
