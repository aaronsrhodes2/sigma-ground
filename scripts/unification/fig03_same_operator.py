"""Figure 3 — Stage 3 (L1, same operator): (a) the repo's FVM heat law vs the analytic Newtonian potential, radial profile
with residual; (b) the transient approach to the Poisson solution in units of the corrected box mode τ_eff; (c) the n = 1
polytrope: σ(r) with and without the pressure term (weak-field rows only; w_c = 0.5 shown dashed as outside validity).
Leaf constants: G from Mentat lookup_constant = sigma_ground.field.constants (verified equal); the material values are
free scale parameters of the identity and cancel."""
import math, sys, os, importlib.util, matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from sigma_ground.dynamics.fields.heat import diffuse_fvm, stable_dt
from sigma_ground.field.constants import G, C
from sigma_ground.inventory.core.sigma import XI

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

# ---------- (a)+(b): P1 at 32^3 with the split kick at 1x stable_dt (the honest setting after verification) ----------
def run_p1(n, dx, R_cells, kick_mult=1, n_tau=12, rho_m=8000.0, kk=50.0, rho_cp=3.5e6, q0=1e6):
    c = (n - 1) / 2; xs = (np.arange(n) - c) * dx
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij'); r = np.sqrt(X**2 + Y**2 + Z**2)
    R = R_cells * dx; inside = r <= R; M = rho_m * inside.sum() * dx**3
    phi = np.where(r >= R, -G * M / np.maximum(r, 1e-12), -G * M * (3 * R * R - r * r) / (2 * R**3))
    s = q0 / (4 * np.pi * G * rho_m * kk); T_exact = 300.0 - phi * s
    alpha = kk / rho_cp
    T = np.full((n, n, n), 300.0); k = np.full((n, n, n), kk); rcp = np.full((n, n, n), rho_cp)
    wall = np.zeros((n, n, n), bool); wall[0] = wall[-1] = True; wall[:, 0] = wall[:, -1] = True; wall[:, :, 0] = wall[:, :, -1] = True
    tau_eff = ((n - 1) / n) ** 2 * (n * dx) ** 2 / (3 * np.pi ** 2 * alpha)
    dt_split = stable_dt(alpha, dx) * kick_mult; t = 0.0; hist = []
    ext = (r > R + 1.5 * dx) & ~wall
    while t < n_tau * tau_eff:
        T, _ = diffuse_fvm(T, k, rcp, dx, total_time=dt_split, ambient_mask=wall, ambient_T=T_exact, max_substeps=10**6)
        T[inside] += q0 * dt_split / rho_cp; t += dt_split
        hist.append((t / tau_eff, float((np.abs(T - T_exact)[ext] / np.abs(T_exact - 300.0)[ext]).max())))
    return r, T, T_exact, R, wall, np.array(hist)

r, T, T_exact, R, wall, hist = run_p1(32, 0.00875, 6.0, kick_mult=1)
bins = np.linspace(0, r[~wall].max(), 40); idx = np.digitize(r, bins)
prof_fvm = np.array([np.mean((T - 300.0)[(idx == i) & ~wall]) if np.any((idx == i) & ~wall) else np.nan for i in range(1, len(bins))])
prof_ex = np.array([np.mean((T_exact - 300.0)[(idx == i) & ~wall]) if np.any((idx == i) & ~wall) else np.nan for i in range(1, len(bins))])
rc = 0.5 * (bins[1:] + bins[:-1])

# ---------- (c): P7 polytrope at 28^3 (resolution-stable per verification), with the solver from pilot_p7 ----------
spec = importlib.util.spec_from_file_location("p7", os.path.join(os.path.dirname(__file__), "pilot_p7_sigma_three_fields.py"))
src = open(spec.origin, encoding="utf-8").read().split("n, dx, a_cells = 40, 1000.0, 5.0")[0]   # reuse poisson_fvm only
ns = {}; exec(compile(src, "p7_solver", "exec"), ns); poisson_fvm = ns["poisson_fvm"]
n7, dx7, a_cells = 28, 1000.0, 3.5
a = a_cells * dx7; R7 = math.pi * a; c0 = (n7 - 1) / 2; xs = (np.arange(n7) - c0) * dx7
X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij'); r7 = np.sqrt(X**2 + Y**2 + Z**2)
xi7 = np.maximum(r7 / a, 1e-9); shape = np.where(r7 < R7, np.sin(xi7) / xi7, 0.0)
curves = {}
for w_c, style in ((1e-3, "-"), (0.1, "-"), (0.5, "--")):
    rho_c = w_c * C**2 / (6 * math.pi * G * a**2); K = 2 * math.pi * G * a**2
    rho = rho_c * shape; P = K * rho**2
    Phi_n, _, _ = poisson_fvm(rho / rho_c, dx7); Psi_n, _, _ = poisson_fvm((rho - 3 * P / C**2) / rho_c, dx7)
    Phi = -4 * math.pi * G * rho_c * Phi_n; Psi = -4 * math.pi * G * rho_c * Psi_n
    sig = -(2 * XI / C**2) * Psi; sig_rho = -(2 * XI / C**2) * Phi
    b7 = np.linspace(0, 1.25 * R7, 16); i7 = np.digitize(r7, b7)
    prof = lambda F: np.array([np.mean(F[i7 == i]) if np.any(i7 == i) else np.nan for i in range(1, len(b7))])
    curves[w_c] = (0.5 * (b7[1:] + b7[:-1]) / R7, prof(sig), prof(sig_rho), style, float(sig[r7 < 1.2 * dx7].mean() / sig_rho[r7 < 1.2 * dx7].mean()))

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=180); fig.patch.set_facecolor(SURFACE)
ax = axes[0]; ax.set_facecolor(SURFACE)
ax.plot(rc / R, prof_ex, color=S1, lw=2, label="analytic Newtonian −Φ·s  (G via Mentat)")
ax.plot(rc / R, prof_fvm, color=S2, lw=0, marker="o", ms=4, label="repo FVM heat law, steady state")
ax.axvline(1.0, color=GRID, lw=1); ax.text(1.02, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] else 1, "sphere edge", fontsize=8, color=MUTED, va="top")
ax.set_xlabel("r / R"); ax.set_ylabel("T − T₀  (K)"); ax.set_title("(a) one operator: steady heat vs gravity\n32³, exact wall data, kick at 1× stable_dt", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8); ax.grid(color=GRID, lw=0.8)
ax2 = ax.twinx(); ax2.plot(rc / R, 100 * (prof_fvm - prof_ex) / np.nanmax(np.abs(prof_ex)), color=MUTED, lw=1, alpha=0.6)
ax2.set_ylabel("residual (% of peak)", color=MUTED); ax2.tick_params(colors=MUTED)
ax = axes[1]; ax.set_facecolor(SURFACE)
ax.semilogy(hist[:, 0], 100 * hist[:, 1], color=S1, lw=2)
ax.set_xlabel("t / τ_eff   (τ_eff = ((n−1)/n)² L² / 3π²α, the box mode)"); ax.set_ylabel("max exterior deviation from Poisson (%)")
ax.set_title("(b) the difference: heat must relax; the potential is instantaneous\n(the curve is the box's Dirichlet mode, e^(−t/τ))", fontsize=10, color=INK)
ax.grid(color=GRID, lw=0.8, which="both")
ax = axes[2]; ax.set_facecolor(SURFACE)
cols = {1e-3: S3, 0.1: S1, 0.5: S2}
for w_c, (xr, sg, sr, style, ratio_c) in curves.items():
    norm = np.nanmax(np.abs(sr))
    ax.plot(xr, sg / norm, color=cols[w_c], lw=2, linestyle=style, label=f"w_c = {w_c:g}: with −3P/c²  (centre ratio {ratio_c:.3f})" + ("  outside validity" if w_c == 0.5 else ""))
    ax.plot(xr, sr / norm, color=cols[w_c], lw=1, linestyle=":", alpha=0.8)
ax.plot([], [], color=MUTED, lw=1, linestyle=":", label="ρ-only field (dotted; identical shape for every w_c)")
ax.axvline(1.0, color=GRID, lw=1); ax.set_xlabel("r / R"); ax.set_ylabel("σ(r) / max|σ_ρ-only|")
ax.set_title("(c) σ-bridge on an n = 1 polytrope: the pressure term\nlowers central σ by 0.609·w_c (two sources, one solver)", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8); ax.grid(color=GRID, lw=0.8)
fig.suptitle("Fig. 3 — L1, same operator: true, and true of every static linear field", fontsize=12, color=INK)
fig.tight_layout(); fig.savefig("misc/unification/fig03_same_operator.png", bbox_inches="tight", facecolor=SURFACE)
print("saved; P1 steady exterior max rel err (1x kick, 12 tau):", float(hist[-1, 1]), "| polytrope central ratios Ψ_c/Φ_c:", {w: v[4] for w, v in curves.items()}, "(closed form 1 − 0.609 w_c)")
