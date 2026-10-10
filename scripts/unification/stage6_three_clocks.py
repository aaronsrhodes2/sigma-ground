"""Stage 6b — the three clocks: one spatial operator, three time structures.

A source of constant power Q is switched on at t = 0. At a fixed distance r the field is:
  elliptic (Newtonian gravity, steady Poisson):  u(t) = u_ss                for all t > 0       (instantaneous)
  parabolic (heat, Fourier):                      u(t) = u_ss · erfc(r / 2√(αt))                 (Carslaw & Jaeger, point source)
  hyperbolic (sound; gravity in GR):              u(t) = u_ss · θ(t − r/c)                       (retarded, switched-on source)
with the SAME steady value u_ss = Q/(4πk r) — the shared Laplacian of Stages 1–3.

The parabolic curve is also computed on the repo's own engine, `dynamics.fields.heat.diffuse_fvm`, with a small
switched-on spherical source and walls far away (pinned to the initial value), and compared with the erfc law until the
walls are felt. The elliptic and hyperbolic curves are exact closed forms (stated as such); the repo has no wave solver.
Scaled units: α = 1 m²/s, k = 1 W/(m·K), c = 1 m/s, r = 6 m — so diffusion arrives on r²/α = 36 s, sound at r/c = 6 s.
"""
import json, math, os
import numpy as np
from scipy.special import erfc
from sigma_ground.dynamics.fields.heat import diffuse_fvm, stable_dt

n, dx = 64, 1.0
kk, rho_cp = 1.0, 1.0; alpha = kk / rho_cp
c0 = (n - 1) / 2; xs = (np.arange(n) - c0) * dx
X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij"); r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
src = r <= 1.6 * dx; Vsrc = src.sum() * dx ** 3
Q = 1.0; q = Q / Vsrc
wall = np.zeros((n, n, n), bool); wall[0] = wall[-1] = True; wall[:, 0] = wall[:, -1] = True; wall[:, :, 0] = wall[:, :, -1] = True
r_obs = 6.0
obs = np.abs(r - r_obs) < 0.5 * dx
T = np.zeros((n, n, n)); k = np.full((n, n, n), kk); rcp = np.full((n, n, n), rho_cp)
dt_split = stable_dt(alpha, dx) * 3
times, vals, t = [], [], 0.0
t_end = 220.0
while t < t_end:
    T, _ = diffuse_fvm(T, k, rcp, dx, total_time=dt_split, ambient_mask=wall, ambient_T=0.0, max_substeps=10 ** 6)
    T[src] += q * dt_split / rho_cp; t += dt_split
    times.append(t); vals.append(float(T[obs].mean()))
u_ss = Q / (4 * math.pi * kk * r_obs)
times = np.array(times); vals = np.array(vals) / u_ss
t_wall = (c0 - r_obs) ** 2 / (4 * alpha)       # rough time for the walls to be felt at r_obs
early = times < 0.5 * t_wall
err = float(np.max(np.abs(vals[early] - erfc(r_obs / (2 * np.sqrt(alpha * times[early]))))[times[early] > 2.0]))
out = dict(r_obs=r_obs, alpha=alpha, c=1.0, diffusion_time=r_obs ** 2 / alpha, sound_time=r_obs / 1.0, t_wall_estimate=t_wall,
           engine_vs_erfc_max_abs_err_before_walls=err, t=times[::20].tolist(), engine=vals[::20].tolist())
json.dump(out, open("misc/unification/stage6_three_clocks.json", "w"), indent=1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
tt = np.logspace(-1, np.log10(t_end), 600)
fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=180); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.plot(tt, np.ones_like(tt), color=S1, lw=2.4, label="elliptic — Newtonian gravity: the field is there at once")
ax.plot(tt, (tt >= r_obs / 1.0).astype(float), color=S3, lw=2.4, linestyle=(0, (6, 3)), label="hyperbolic — sound, and gravity in GR: arrives at t = r/c")
ax.plot(tt, erfc(r_obs / (2 * np.sqrt(alpha * tt))), color=S2, lw=2.4, linestyle=(0, (2, 2)), label="parabolic — heat (Fourier): erfc(r / 2√αt)")
ax.plot(times[::15], vals[::15], color=S2, marker="o", ms=3.5, lw=0, label="parabolic — the repo's FVM heat law (64³, switched-on source)")
ax.axvline(t_wall, color=MUTED, lw=1, linestyle=":"); ax.text(t_wall * 1.05, 0.05, "walls felt\n(finite box)", fontsize=8, color=MUTED)
ax.set_xscale("log"); ax.set_xlim(0.1, t_end); ax.set_ylim(-0.03, 1.12)
ax.set_xlabel("time since the source switched on  (s; α = 1 m²/s, c = 1 m/s, r = 6 m)"); ax.set_ylabel("field at r / its steady value Q/4πkr")
ax.set_title("Fig. 6b — Three clocks, one Laplacian: the same steady field, reached instantly, by a wavefront, or by diffusion\n"
             "engine vs erfc before the walls: max |Δ| = %.3f" % err, fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="center left"); ax.grid(color=GRID, lw=0.6, which="both")
fig.tight_layout(); fig.savefig("misc/unification/fig06b_three_clocks.png", bbox_inches="tight", facecolor=SURFACE)
print(json.dumps({k: v for k, v in out.items() if k not in ("t", "engine")}, indent=1))
