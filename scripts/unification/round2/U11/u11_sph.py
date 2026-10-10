"""U11 SPH leg: isothermal SPH gas under uniform g -> rho ~ exp(-z/H), H = c_s^2/g, on the UNMODIFIED engine.

Modes
  literal       design verbatim: 3-D, N uniform in 1 x 4 x 1 H, dt = cfl_timestep(scene) every step
  literal_cs    same geometry, dt = 0.3 h / max(c_s, v_max) (engine CFL formula, isothermal c_s substituted, driver-side)
  line          planar-parallel column: every parcel on the line x = z = 0 (grad_W has zero lateral components,
                so the column is confined by symmetry); dt as literal_cs
  periodic3d    3-D, lateral periodicity by a full 3x3 image tiling rebuilt every step (driver-side); image
                accelerations injected from their real counterparts through the engine's own velocity-Verlet
                cache so images drift exactly with their originals; dt as literal_cs

Scaled units (a disclosed Choice): c_s, g given; H = c_s^2/g (Mentat sympy.dsolve, see u11_formula.py).
Parcels: 'water_ice', radius 0.02 (design) - the material only sets the parcel mass; with eos='isothermal'
the pressure is rho_sph c_s^2 and T is uniform, so buoyancy and conduction are identically zero.
"""
import sys, json, math, time, argparse
import numpy as np
from sigma_ground.dynamics.continuum import ContinuumParcel, ContinuumScene, continuum_step, cfl_timestep
from sigma_ground.dynamics.scene import GroundPlane
from sigma_ground.dynamics.vec import Vec3

ap = argparse.ArgumentParser()
ap.add_argument('--mode', required=True, choices=['literal', 'literal_cs', 'line', 'periodic3d'])
ap.add_argument('--N', type=int, default=128)
ap.add_argument('--h', type=float, default=0.1)          # absolute smoothing length
ap.add_argument('--g', type=float, default=1.0)
ap.add_argument('--cs', type=float, default=1.0)
ap.add_argument('--steps', type=int, default=2000)
ap.add_argument('--max_steps', type=int, default=2000)
ap.add_argument('--seed', type=int, default=3)
ap.add_argument('--init', default='uniform')            # 'uniform' on [0, 4H] or 'exp:<H0 in units of H>'
ap.add_argument('--L', type=float, default=None)        # lateral box (literal: 1.0; periodic3d: 4h)
ap.add_argument('--K_line', type=float, default=None)   # integral of the 3-D kernel along a line, times h^2 (Mentat)
ap.add_argument('--out', required=True)
ap.add_argument('--record_every', type=int, default=50)
ap.add_argument('--alpha', type=float, default=1.0)     # artificial-viscosity coefficient (engine default 1.0)
a = ap.parse_args()

H = a.cs ** 2 / a.g
rng = np.random.default_rng(a.seed)
Z = 4.0 * H
if a.mode in ('literal', 'literal_cs'):
    L = a.L or 1.0
elif a.mode == 'periodic3d':
    L = a.L or 4.0 * a.h
    assert L >= 4.0 * a.h - 1e-12, "periodic box must be >= 4h (no double counting of images)"
else:
    L = 0.0


def draw_y(n):
    if a.init == 'uniform':
        return rng.uniform(0.0, Z, n)
    H0 = float(a.init.split(':')[1]) * H
    u = rng.uniform(0.0, 1.0, n)                           # exponential truncated at 4H
    return -H0 * np.log(1.0 - u * (1.0 - math.exp(-Z / H0)))


ys0 = draw_y(a.N)
ps = []
for i in range(a.N):
    if a.mode == 'line':
        pos = Vec3(0.0, float(ys0[i]), 0.0)
    else:
        pos = Vec3(float(rng.uniform(0, L)), float(ys0[i]), float(rng.uniform(0, L)))
    ps.append(ContinuumParcel('water_ice', 0.02, temperature=300.0, position=pos,
                              velocity=Vec3(0, 0, 0), label=f'p{i}'))
m = ps[0].mass
sc = ContinuumScene(ps, gravity=Vec3(0, -a.g, 0), ground=GroundPlane(), smoothing_h=a.h,
                    eos='isothermal', sound_speed_m_s=a.cs)
if a.mode == 'periodic3d':
    # isothermal: the first-law booking only feeds energy_bath_J (no dynamical effect); off so the
    # pair lists do not have to follow the image set
    sc.energy_equation = False

real = list(ps)
SHIFTS = [(sx, sz) for sx in (-1, 0, 1) for sz in (-1, 0, 1) if (sx, sz) != (0, 0)]


def build_images(acc_real):
    """3x3 lateral tiling: wrap reals into [0, L), then 8 shifted copies of each. Images carry the real's
    velocity; their velocity-Verlet start acceleration is the real's (cache injection)."""
    for p in real:
        p.position = Vec3(p.position.x % L, p.position.y, p.position.z % L)
    imgs = []
    acc = list(acc_real) if acc_real is not None else None
    for (sx, sz) in SHIFTS:
        for k, p in enumerate(real):
            q = ContinuumParcel.__new__(ContinuumParcel)
            q.__dict__.update(p.__dict__)
            q.position = Vec3(p.position.x + sx * L, p.position.y, p.position.z + sz * L)
            q.velocity = Vec3(p.velocity.x, p.velocity.y, p.velocity.z)
            q.neighbors = []
            imgs.append(q)
            if acc is not None:
                acc.append(acc_real[k])
    sc.parcels = real + imgs
    sc._vv_accel = acc


def ke():
    return sum(0.5 * p.mass * (p.velocity.x ** 2 + p.velocity.y ** 2 + p.velocity.z ** 2) for p in real)


def window_fit(y, rho):
    sel = (y > 0.8 * H) & (y < 2.8 * H) & np.isfinite(rho) & (rho > 0)
    if sel.sum() < 3:
        return dict(n_fit=int(sel.sum()), H_fit=float('nan'), H_fit_over_H=float('nan'), rms=float('nan'))
    c = np.polyfit(y[sel], np.log(rho[sel]), 1)
    res = np.log(rho[sel]) - np.polyval(c, y[sel])
    return dict(n_fit=int(sel.sum()), H_fit=float(-1.0 / c[0]), H_fit_over_H=float(-1.0 / c[0] / H),
                rms=float(np.sqrt(np.mean(res ** 2))), intercept=float(c[1]))


def snapshot():
    y = np.array([p.position.y for p in real])
    rho = np.array([p.sph_density for p in real])
    out = dict(fit=window_fit(y, rho), y=y.tolist(), rho=rho.tolist(),
               nnb=[len(p.neighbors) for p in real],
               vy=[p.velocity.y for p in real],
               ke_parcels=[0.5 * p.mass * (p.velocity.x ** 2 + p.velocity.y ** 2 + p.velocity.z ** 2) for p in real])
    if a.mode in ('literal', 'literal_cs', 'periodic3d'):
        xs = np.array([p.position.x for p in real]); zs = np.array([p.position.z for p in real])
        out['lateral_std'] = float(np.sqrt(np.var(xs) + np.var(zs)))
        out['lateral_extent'] = [float(xs.min()), float(xs.max()), float(zs.min()), float(zs.max())]
        out['x'] = xs.tolist(); out['z'] = zs.tolist()
    if a.mode == 'line':
        o = np.argsort(y); ys = y[o]
        lam = np.full(len(ys), np.nan)
        lam[1:-1] = 2.0 * m / (ys[2:] - ys[:-2])            # counted linear density, kernel-free
        out['counted_fit'] = window_fit(ys, lam)
        out['counted_lambda'] = [None if not np.isfinite(v) else float(v) for v in lam]
        out['y_sorted'] = ys.tolist()
        if a.K_line:
            K = a.K_line / a.h ** 2
            n_above = np.arange(len(ys))[::-1].astype(float)   # parcels strictly above, sorted ascending
            weight = a.g * K * m * (n_above + 0.5)
            P = a.cs ** 2 * rho[o]
            r = P / weight
            sel = (ys > 0.8 * H) & (ys < 2.8 * H)
            out['hydrostatic'] = dict(ratio_mean=float(np.mean(r[sel])),
                                      ratio_rms_dev=float(np.sqrt(np.mean((r[sel] - 1) ** 2))),
                                      ratio_all=r.tolist())
    if a.mode == 'periodic3d':
        edges = np.linspace(0, 4 * H, 21)
        cnt, _ = np.histogram(y, edges)
        zc = 0.5 * (edges[1:] + edges[:-1])
        nden = cnt * m / (L * L * (edges[1] - edges[0]))
        sel = (zc > 0.8 * H) & (zc < 2.8 * H) & (cnt > 0)
        if sel.sum() >= 3:
            cfit = np.polyfit(zc[sel], np.log(nden[sel]), 1)
            hc = float(-1 / cfit[0])
        else:
            hc = float('nan')
        out['counted_fit'] = dict(H_fit=hc, H_fit_over_H=hc / H, zc=zc.tolist(), counted_rho=nden.tolist())
        o = np.argsort(y); ys = y[o]
        n_above = np.arange(len(ys))[::-1].astype(float)
        weight = a.g * m * (n_above + 0.5) / (L * L)
        P = a.cs ** 2 * rho[o]
        r = P / weight
        selh = (ys > 0.8 * H) & (ys < 2.8 * H)
        out['hydrostatic'] = dict(ratio_mean=float(np.mean(r[selh])),
                                  ratio_rms_dev=float(np.sqrt(np.mean((r[selh] - 1) ** 2))))
    return out


hist = []
ke_peak = 0.0
first_settle = None
snap_budget = None
t0 = time.perf_counter()
acc_real = None
dts = []
step = 0
while step < a.max_steps:
    if a.mode == 'periodic3d':
        build_images(acc_real)
    if a.mode == 'literal':
        dt = cfl_timestep(sc)
    else:
        vmax = max(p.velocity.length() for p in real)
        dt = 0.3 * a.h / max(a.cs, vmax, 1e-10)
    continuum_step(sc, dt, alpha_visc=a.alpha)
    dts.append(dt)
    if a.mode == 'periodic3d':
        acc_real = sc._vv_accel[:len(real)]
        sc.parcels = list(real)
    step += 1
    k = ke()
    ke_peak = max(ke_peak, k)
    settled = k < 1e-4 * ke_peak
    if settled and first_settle is None:
        first_settle = dict(step=step, t=sc.time)
    if step % a.record_every == 0 or step == 1:
        y = np.array([p.position.y for p in real]); rho = np.array([p.sph_density for p in real])
        k_free = sum(0.5 * p.mass * (p.velocity.x ** 2 + p.velocity.y ** 2 + p.velocity.z ** 2)
                     for p in real if p.position.y >= 2.0 * p.radius)   # excludes parcels in ground contact
        rec = dict(step=step, t=sc.time, ke=k, ke_over_peak=k / ke_peak if ke_peak > 0 else 0.0,
                   ke_free_over_peak=k_free / ke_peak if ke_peak > 0 else 0.0,
                   fit=window_fit(y, rho), ymax=float(y.max()), ymean=float(y.mean()),
                   mean_nnb=float(np.mean([len(p.neighbors) for p in real])), wall_s=time.perf_counter() - t0)
        if a.mode in ('literal', 'literal_cs', 'periodic3d'):
            xs = np.array([p.position.x for p in real]); zs = np.array([p.position.z for p in real])
            rec['lateral_std'] = float(np.sqrt(np.var(xs) + np.var(zs)))
        hist.append(rec)
        brief = {kk: rec[kk] for kk in ('step', 't', 'ke_over_peak', 'ymax', 'mean_nnb', 'wall_s')}
        brief['H_fit'] = rec['fit'].get('H_fit')
        print(json.dumps(brief), flush=True)
    if step == a.steps:
        snap_budget = snapshot()
        snap_budget.update(step=step, t=sc.time, ke_over_peak=k / ke_peak)
    if step >= a.steps and settled:
        break

final = snapshot()
final.update(step=step, t=sc.time, ke_over_peak=ke() / ke_peak)
res = dict(args=vars(a), H=H, L=L, parcel_mass=m, ke_peak=ke_peak, first_settle=first_settle,
           dt_first=dts[0], dt_last=dts[-1], dt_min=min(dts), total_time=sc.time, steps=step,
           wall_s=time.perf_counter() - t0, material_sound_speed=real[0].sound_speed,
           cfl_timestep_engine=cfl_timestep(sc),
           history=hist, at_step_budget=snap_budget, final=final)
json.dump(res, open(a.out, 'w'), indent=1)
print('DONE', a.out, json.dumps(final['fit']), 'settled', first_settle, flush=True)
