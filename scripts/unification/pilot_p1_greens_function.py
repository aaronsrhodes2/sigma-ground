"""Pilot P1 — Green's-function identity: steady Fourier heat (repo FVM law) vs Newtonian potential.

k∇²T = −q  (steady heat, volumetric source q)      ∇²φ = 4πGρ  (Newtonian gravity)
⇒ with q = q0 inside a uniform sphere and ρ = ρ_m inside the same sphere,  T − T0 = −φ · q0/(4πG ρ_m k).
Volumetric heating is applied by operator splitting between diffuse_fvm calls (fixed point = discrete Poisson).
Walls are Dirichlet-pinned to the exact Newtonian values (a harmonic function is fixed by its boundary data).
"""
import sys, time, numpy as np
from sigma_ground.dynamics.fields.heat import diffuse_fvm, stable_dt
from sigma_ground.field.constants import G

def run(n, dx, R_cells, rho_m=8000.0, kk=50.0, rho_cp=3.5e6, q0=1e6, n_tau=8, verbose=True):
    c = (n - 1) / 2; xs = (np.arange(n) - c) * dx
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij'); r = np.sqrt(X**2 + Y**2 + Z**2)
    R = R_cells * dx; inside = r <= R
    M = rho_m * inside.sum() * dx**3
    phi = np.where(r >= R, -G * M / np.maximum(r, 1e-12), -G * M * (3 * R * R - r * r) / (2 * R**3))
    s = q0 / (4 * np.pi * G * rho_m * kk); T_exact = 300.0 - phi * s
    alpha = kk / rho_cp
    T = np.full((n, n, n), 300.0); k = np.full((n, n, n), kk); rcp = np.full((n, n, n), rho_cp)
    wall = np.zeros((n, n, n), bool); wall[0] = wall[-1] = True; wall[:, 0] = wall[:, -1] = True; wall[:, :, 0] = wall[:, :, -1] = True
    L = n * dx; tau = L * L / (3 * np.pi**2 * alpha)
    dt_split = stable_dt(alpha, dx) * 20
    t = 0.0; t0 = time.time(); history = []
    while t < n_tau * tau:
        T, _ = diffuse_fvm(T, k, rcp, dx, total_time=dt_split, ambient_mask=wall, ambient_T=T_exact, max_substeps=10**6)
        T[inside] += q0 * dt_split / rho_cp; t += dt_split
        ext = (r > R + 1.5 * dx) & ~wall
        history.append((t / tau, float((np.abs(T - T_exact)[ext] / np.abs(T_exact - 300.0)[ext]).max())))
    ext = (r > R + 1.5 * dx) & ~wall
    rel = np.abs(T - T_exact)[ext] / np.abs(T_exact - 300.0)[ext]
    out = dict(n=n, dx=dx, R_cells=R_cells, tau_s=tau, wall_s=time.time() - t0,
               ext_max_rel=float(rel.max()), ext_median_rel=float(np.median(rel)),
               transient=[(round(a, 2), b) for a, b in history[::max(1, len(history)//8)]])
    if verbose: print(out)
    return out

if __name__ == '__main__':
    a = run(16, 0.0175, 3.0); b = run(32, 0.00875, 6.0)
    print('convergence ratio (coarse/fine max rel err):', round(a['ext_max_rel'] / b['ext_max_rel'], 2))
