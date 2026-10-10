"""Pilot P7 — the σ-bridge as three fields of one operator, on a Lane–Emden n = 1 polytrope.

∇²Φ = 4πG ρ            (Newtonian potential)
∇²Ψ = 4πG (ρ − 3P/c²)  (σ-potential: σ = −(2ξ/c²) Ψ, from □σ = −ξR, static weak field)
∇²T = −q/k             (steady Fourier; the repo's FVM heat law is used as the Poisson solver, per pilot P1)

n = 1 polytrope: ρ = ρ_c sin(r/a)/(r/a), P = K ρ², a² = K/(2πG), R = πa. Choosing w_c = 3P_c/(ρ_c c²) fixes ρ_c.
Predictions (exact for n = 1): ∫P dV = ½ K ρ_c M  ⇒  Q_Ψ/Q_Φ = 1 − w_c/2 (monopole);  compactness r_s/R = (4/3) w_c.
"""
import json, math, sys
import numpy as np
from sigma_ground.dynamics.fields.heat import diffuse_fvm, stable_dt
from sigma_ground.field.constants import G, C
from sigma_ground.inventory.core.sigma import XI   # only surviving copy of ξ (excised from field/ in e66615b)

def poisson_fvm(s, dx, n_tau=8):
    """Solve ∇²v = −s (v → Q/4πr outside) with the repo's conservative FVM heat law run to steady state."""
    n = s.shape[0]; c = (n - 1) / 2; xs = (np.arange(n) - c) * dx
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij'); r = np.sqrt(X**2 + Y**2 + Z**2)
    Q = float(s.sum() * dx**3); v_ext = Q / (4 * np.pi * np.maximum(r, dx / 2))
    kk, rho_cp = 1.0, 1.0; alpha = kk / rho_cp
    wall = np.zeros((n, n, n), bool); wall[0] = wall[-1] = True; wall[:, 0] = wall[:, -1] = True; wall[:, :, 0] = wall[:, :, -1] = True
    T = np.zeros((n, n, n)); k = np.full((n, n, n), kk); rcp = np.full((n, n, n), rho_cp)
    tau = (n * dx)**2 / (3 * np.pi**2 * alpha); dt_split = stable_dt(alpha, dx) * 20; t = 0.0
    while t < n_tau * tau:
        T, _ = diffuse_fvm(T, k, rcp, dx, total_time=dt_split, ambient_mask=wall, ambient_T=v_ext, max_substeps=10**6)
        T += s * dt_split / rho_cp; t += dt_split
    return T, r, Q

n, dx, a_cells = 40, 1000.0, 5.0
a = a_cells * dx; R = math.pi * a
c0 = (n - 1) / 2; xs = (np.arange(n) - c0) * dx
X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij'); r = np.sqrt(X**2 + Y**2 + Z**2)
xi = np.maximum(r / a, 1e-9); shape = np.where(r < R, np.sin(xi) / xi, 0.0)
rows = []
for w_c in (1e-3, 1e-1, 5e-1):
    rho_c = w_c * C**2 / (6 * math.pi * G * a**2); K = 2 * math.pi * G * a**2
    rho = rho_c * shape; P = K * rho**2
    s_phi = rho; s_psi = rho - 3 * P / C**2
    Phi_n, _, Q_phi = poisson_fvm(s_phi / rho_c, dx); Psi_n, _, Q_psi = poisson_fvm(s_psi / rho_c, dx)
    Phi = -4 * math.pi * G * rho_c * Phi_n; Psi = -4 * math.pi * G * rho_c * Psi_n
    sigma = -(2 * XI / C**2) * Psi
    inside = r < R
    M = float(rho.sum() * dx**3); r_s = 2 * G * M / C**2
    # interior shape test: pressure term is more centrally concentrated than ρ → Ψ/(1−w_c/2) ≠ Φ inside
    ratio_mono = Q_psi / Q_phi
    shape_dev = float(np.max(np.abs(Psi / ratio_mono - Phi)[inside]) / np.max(np.abs(Phi)))
    rows.append(dict(w_c=w_c, rho_c_kg_m3=rho_c, M_kg=M, R_m=R, compactness_rs_over_R=r_s / R, predicted_compactness=4 * w_c / 3,
                     Q_psi_over_Q_phi=ratio_mono, predicted_1_minus_wc_over_2=1 - w_c / 2,
                     interior_shape_deviation_frac=shape_dev,
                     sigma_centre=float(sigma[inside].min() if sigma[inside].min() < 0 else sigma[inside].max()),
                     sigma_centre_rho_only=float((-(2 * XI / C**2) * Phi)[inside][np.argmax(np.abs(Phi[inside]))]),
                     sigma_surface_field_eq=XI * r_s / R))
    print(json.dumps(rows[-1]), flush=True)
