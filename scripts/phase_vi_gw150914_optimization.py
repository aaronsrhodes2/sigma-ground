"""
Phase VI — GW150914/H1 template optimisation at 16 kHz.

Scans ringdown start time (t_start) and QNM parameter space (f_QNM, τ_QNM)
to find the maximum achievable ρ_syn without pycbc IMR subtraction.

Three-stage scan:
  1. t_start ∈ [0.5, 8.0] ms  at nominal (f, τ) — find t*
  2. (f, τ) grid around GW150914 posterior  at t* — find (f*, τ*)
  3. Joint (t*, f*, τ*) optimal — report best achievable ρ_syn

Physics:
  The QNM signal at time t_start ms after merger has amplitude
  A_det × exp(−t_start/τ_QNM).  Earlier start → higher ρ_syn.
  Constraint: cannot start inside the IMR merger (t < 0 after merger peak).
  Without pycbc IMR subtraction, starting too early contaminates ρ_obs
  (merger signal leaks into QNM window) while ρ_syn is clean.
  Phase VI quantifies the gap between current pipeline and pycbc.

GW150914/H1 published QNM posterior (Isi et al. 2019; GWTC-1):
  f_220 ∈ [235, 267] Hz   (68 % CI; MAP ≈ 251 Hz)
  τ_220 ∈ [3.0,  5.5] ms  (68 % CI; MAP ≈ 4.0 ms)

Output:
  - Figure:  misc/bh_phase_vi_gw150914_optimization.png
  - Verdict: misc/bh_phase_vi_gw150914_optimization_results.md
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from sigma_ground.field.interface.ligo_echo_search import (
    EVENTS, fetch_strain, read_strain, find_merger_sample,
    estimate_psd, bandpass, whiten,
)
from sigma_ground.field.constants import ETA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────────────────
G_SI    = 6.67430e-11
C_SI    = 299792458.0
M_SUN   = 1.98892e30
MPC_M   = 3.085677581e22

# GW150914 LVK MAP parameters (Phase III)
M_REM   = 62.2          # M☉
E_RAD   = 3.0           # M☉c²
D_L_MPC = 410.0
IOTA    = 163.0         # degrees
F_RD    = 0.15

# GW150914 QNM nominal (Phase III)
F_NOM   = 251.0         # Hz
TAU_NOM = 4.00e-3       # s

TMPL_DUR_MS   = 20.0    # ms — template window duration (fixed)
N_BG_SAMPLES  = 300     # background calibration draws (seed 42)
BG_LO_S, BG_HI_S = 2.0, 12.0   # pre-merger window [s]

ETA_VAL = ETA           # 0.4153

# σ_conv model predictions
GAMMA_PREDS = {
    "GR":           1.0000,
    "exp":          0.8395,
    "sigma_coh":    round(1.0 - ETA_VAL / 2.0, 4),
    "linear_cbrt":  round(ETA_VAL ** (1.0 / 3.0), 4),
}
PRED_COLORS = {
    "GR": "#ff7b72", "exp": "#f0883e",
    "sigma_coh": "#3fb950", "linear_cbrt": "#bc8cff",
}

# ── Scan grids ────────────────────────────────────────────────────────────────
TSTART_GRID_MS = np.arange(0.5, 8.1, 0.5)       # 16 values
F_GRID_HZ      = np.arange(225.0, 281.0, 3.0)   # 19 values
TAU_GRID_MS    = np.arange(2.5,  6.6,  0.25)    # 17 values


# ── Physics helpers ───────────────────────────────────────────────────────────
def a22_source_m(M_f_kg, eps_rd, f_hz, tau_s):
    Q = math.pi * f_hz * tau_s
    return math.sqrt(5.0 * G_SI * eps_rd * M_f_kg / (math.pi * C_SI * f_hz * Q))


def r_rms_sky_avg(iota_deg):
    c = math.cos(math.radians(iota_deg))
    return math.sqrt((1.0 / 5.0) * ((1.0 + c * c) ** 2 / 4.0 + c * c))


def detector_amplitude(f_hz, tau_s):
    """GR-predicted detector amplitude A_det [m/m] at ringdown onset."""
    M_f_kg  = M_REM * M_SUN
    eps_rd  = F_RD * E_RAD / M_REM
    d_L_m   = D_L_MPC * MPC_M
    R_rms   = r_rms_sky_avg(IOTA)
    A22     = a22_source_m(M_f_kg, eps_rd, f_hz, tau_s)
    return (A22 / d_L_m) * R_rms


# ── Data loading (once) ───────────────────────────────────────────────────────
print("Loading GW150914/H1 at 16 kHz …", flush=True)
path   = fetch_strain("GW150914", "H1", fs_khz=16)
strain, t0_gps, fs = read_strain(path)
merger_idx = find_merger_sample("GW150914", t0_gps, fs)

psd_len   = int(8.0 * fs)
psd_start = max(0, merger_idx - int(2.0 * fs) - psd_len)
psd_freqs, psd_vals = estimate_psd(strain, fs, psd_start, psd_len)
bp = bandpass(strain, fs, lo=35.0, hi=min(fs / 2.0 * 0.45, 1500.0))

n_tmpl = int(TMPL_DUR_MS * 1e-3 * fs)


# ── Matched-filter helper ─────────────────────────────────────────────────────
def compute_snr(tstart_ms, f_hz, tau_s):
    """Return (rho_obs, rho_syn) for GW150914/H1.

    The GR injection is h_inj(t) = A_det × exp(−(t+tstart)/τ) × cos(2πf(t+tstart))
    so that the injected signal has already decayed by exp(−tstart/τ) at the
    start of the observation window — matching the physical ringdown.

    The same template exp(−t/τ)cos(2πft) is used for both the matched filter
    on the data and the injection normalisation.
    """
    tau = tau_s
    t_arr   = np.arange(n_tmpl) / fs
    env     = np.exp(-t_arr / tau)
    tc_raw  = env * np.cos(2.0 * np.pi * f_hz * t_arr)
    ts_raw  = env * np.sin(2.0 * np.pi * f_hz * t_arr)
    tc_wht  = whiten(tc_raw, fs, psd_freqs, psd_vals)
    ts_wht  = whiten(ts_raw, fs, psd_freqs, psd_vals)
    norm    = math.sqrt(np.dot(tc_wht, tc_wht) + np.dot(ts_wht, ts_wht))
    if norm <= 0:
        return float("nan"), float("nan")

    def quad_amp(wht_seg):
        return math.sqrt((np.dot(wht_seg, tc_wht) / norm) ** 2 +
                         (np.dot(wht_seg, ts_wht) / norm) ** 2)

    # Observed amplitude
    rd_start = merger_idx + int(tstart_ms * 1e-3 * fs)
    seg = bp[rd_start: rd_start + n_tmpl]
    if len(seg) < n_tmpl:
        return float("nan"), float("nan")
    amp_obs = quad_amp(whiten(seg, fs, psd_freqs, psd_vals))

    # Background sigma_noise
    rng = np.random.default_rng(seed=42)
    bg_amps = []
    for _ in range(N_BG_SAMPLES):
        idx = merger_idx - int(rng.uniform(BG_LO_S, BG_HI_S) * fs)
        if idx >= 0:
            bseg = bp[idx: idx + n_tmpl]
            if len(bseg) == n_tmpl:
                bg_amps.append(quad_amp(whiten(bseg, fs, psd_freqs, psd_vals)))
    if not bg_amps:
        return float("nan"), float("nan")
    sigma_noise = float(np.mean(bg_amps)) / math.sqrt(math.pi / 2.0)

    # GR injection — amplitude decayed by exp(−tstart/τ) at window start
    t_off   = tstart_ms * 1e-3
    A_det   = detector_amplitude(f_hz, tau_s)
    h_inj   = A_det * np.exp(-(t_arr + t_off) / tau) * np.cos(2.0 * np.pi * f_hz * (t_arr + t_off))
    amp_syn = quad_amp(whiten(h_inj, fs, psd_freqs, psd_vals))

    rho_obs = amp_obs / sigma_noise if sigma_noise > 0 else float("nan")
    rho_syn = amp_syn / sigma_noise if sigma_noise > 0 else float("nan")
    return rho_obs, rho_syn


# ── Stage 1: t_start scan at nominal (f, τ) ──────────────────────────────────
print(f"Stage 1: scanning t_start ∈ [{TSTART_GRID_MS[0]:.1f}, {TSTART_GRID_MS[-1]:.1f}] ms …", flush=True)
rho_obs_ts, rho_syn_ts = [], []
for ts in TSTART_GRID_MS:
    ro, rs = compute_snr(ts, F_NOM, TAU_NOM)
    rho_obs_ts.append(ro)
    rho_syn_ts.append(rs)
    print(f"  t_start={ts:.1f}ms  ρ_obs={ro:.3f}  ρ_syn={rs:.4f}", flush=True)

rho_obs_ts = np.array(rho_obs_ts)
rho_syn_ts = np.array(rho_syn_ts)
tstart_opt_idx = int(np.nanargmax(rho_syn_ts))
tstart_opt_ms  = float(TSTART_GRID_MS[tstart_opt_idx])
rho_syn_ts_opt = float(rho_syn_ts[tstart_opt_idx])
print(f"\n  → Optimal t_start = {tstart_opt_ms:.1f} ms  ρ_syn = {rho_syn_ts_opt:.4f}\n")

# ── Stage 2: (f, τ) scan at t_start_opt ──────────────────────────────────────
print(f"Stage 2: scanning (f, τ) at t_start = {tstart_opt_ms:.1f} ms …", flush=True)
rho_syn_grid = np.full((len(F_GRID_HZ), len(TAU_GRID_MS)), float("nan"))
rho_obs_grid = np.full_like(rho_syn_grid, float("nan"))

for i, f in enumerate(F_GRID_HZ):
    for j, tau_ms in enumerate(TAU_GRID_MS):
        ro, rs = compute_snr(tstart_opt_ms, f, tau_ms * 1e-3)
        rho_syn_grid[i, j] = rs
        rho_obs_grid[i, j] = ro
    best_in_row = np.nanmax(rho_syn_grid[i])
    print(f"  f={f:.0f}Hz  best ρ_syn = {best_in_row:.4f}", flush=True)

max_idx = np.unravel_index(np.nanargmax(rho_syn_grid), rho_syn_grid.shape)
f_opt     = float(F_GRID_HZ[max_idx[0]])
tau_opt   = float(TAU_GRID_MS[max_idx[1]])
rho_syn_opt = float(rho_syn_grid[max_idx])
rho_obs_opt = float(rho_obs_grid[max_idx])
gamma_opt   = rho_obs_opt / rho_syn_opt if rho_syn_opt > 0 else float("nan")

print(f"\n  → Optimal (f, τ, t_start) = ({f_opt:.0f} Hz, {tau_opt:.2f} ms, {tstart_opt_ms:.1f} ms)")
print(f"     ρ_syn = {rho_syn_opt:.4f}   ρ_obs = {rho_obs_opt:.4f}   Γ = {gamma_opt:.3f}")

# Phase III reference at (f=251, τ=4ms, t=3ms)
rho_obs_ref = 1.000   # from Phase III/IV @ 16 kHz
rho_syn_ref = 1.543

improvement_syn = rho_syn_opt / rho_syn_ref
improvement_pycbc_needed = 5.0 / rho_syn_opt   # factor needed to reach ρ_syn=5

print(f"\nPhase III reference: ρ_syn = {rho_syn_ref:.3f}")
print(f"Phase VI optimised:  ρ_syn = {rho_syn_opt:.3f}  ({improvement_syn:.2f}× improvement)")
print(f"Gap to ρ_syn=5: need {improvement_pycbc_needed:.2f}× further gain (pycbc target)")


# ── Rician posterior at Phase III vs Phase VI ρ_syn ──────────────────────────
from scipy.special import i0 as bessel_i0

def rician_logpdf(A, s):
    if A <= 0:
        return -1e30
    return math.log(A) - (A**2 + s**2) / 2.0 + float(np.log(bessel_i0(A * s) + 1e-300))

def posterior_gamma(rho_obs_val, rho_syn_val, n_pts=2000):
    G_arr = np.linspace(0.0, 3.0, n_pts)
    logp  = np.array([rician_logpdf(rho_obs_val, G * rho_syn_val) for G in G_arr])
    logp -= logp.max()
    post  = np.exp(logp)
    norm  = np.trapezoid(post, G_arr)
    post /= norm
    cdf   = np.cumsum(post) * (G_arr[1] - G_arr[0])
    cdf  /= cdf[-1]
    return G_arr, post, cdf

G_ref, post_ref, cdf_ref = posterior_gamma(rho_obs_ref, rho_syn_ref)
G_opt, post_opt, cdf_opt = posterior_gamma(rho_obs_opt, rho_syn_opt)

# Hypothetical ρ_syn = 5 (pycbc target), ρ_obs scaled by Γ_map from current obs
gamma_current = rho_obs_ref / rho_syn_ref   # = 0.648
rho_obs_pycbc = gamma_current * 5.0         # assume same Γ
G_pyc, post_pyc, cdf_pyc = posterior_gamma(rho_obs_pycbc, 5.0)

def p_less_than(cdf, G_arr, val):
    idx = np.searchsorted(G_arr, val)
    return float(cdf[min(idx, len(cdf)-1)])

print("\n── Posterior comparison ──────────────────────────────────────────")
for label, G_arr, cdf in [("Phase III (ρ_syn=1.543)", G_ref, cdf_ref),
                            (f"Phase VI  (ρ_syn={rho_syn_opt:.2f})", G_opt, cdf_opt),
                            ("pycbc est (ρ_syn=5.0)", G_pyc, cdf_pyc)]:
    p_gr     = p_less_than(cdf, G_arr, 1.0)
    p_sc     = p_less_than(cdf, G_arr, 0.7924)
    print(f"  {label}: P(Γ<1)={p_gr:.3f}  P(Γ<0.79)={p_sc:.3f}")


# ── Summary printout ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase VI — GW150914/H1 Optimisation Summary")
print("=" * 60)
print(f"Optimal t_start    = {tstart_opt_ms:.1f} ms  (vs 3.0 ms in Phase III)")
print(f"Optimal f_QNM      = {f_opt:.0f} Hz  (nominal: {F_NOM:.0f} Hz)")
print(f"Optimal τ_QNM      = {tau_opt:.2f} ms  (nominal: {TAU_NOM*1e3:.2f} ms)")
print(f"ρ_syn (Phase III)  = {rho_syn_ref:.3f}  (t=3ms, nominal template)")
print(f"ρ_syn (Phase VI)   = {rho_syn_opt:.3f}  ({improvement_syn:.2f}× improvement)")
print(f"ρ_syn gap to B>10  = {improvement_pycbc_needed:.2f}× remaining (target: 5.0)")
print(f"Γ at optimum       = {gamma_opt:.3f}")
print("=" * 60)


# ── Figure ────────────────────────────────────────────────────────────────────
DARK_BG    = "#0d1117"
LIGHT_TEXT = "#e6edf3"
ACCENT     = "#58a6ff"
ORANGE     = "#f0883e"
GREEN      = "#3fb950"
YELLOW     = "#e3b341"

fig = plt.figure(figsize=(15, 10), facecolor=DARK_BG)

ax_ts  = fig.add_axes([0.06, 0.56, 0.38, 0.36])   # t_start scan
ax_ft  = fig.add_axes([0.52, 0.56, 0.44, 0.36])   # (f, τ) heat map
ax_p   = fig.add_axes([0.06, 0.08, 0.88, 0.38])   # posteriors

for ax in [ax_ts, ax_ft, ax_p]:
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=LIGHT_TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")

# --- t_start scan ---
ax_ts.plot(TSTART_GRID_MS, rho_syn_ts, color=ACCENT, lw=1.5, marker="o",
           ms=4, label="ρ_syn (GR predicted)")
ax_ts.plot(TSTART_GRID_MS, rho_obs_ts, color=ORANGE, lw=1.2, marker="s",
           ms=4, ls="--", label="ρ_obs (observed)")
ax_ts.axvline(tstart_opt_ms, color=YELLOW, lw=0.9, ls=":", alpha=0.9,
              label=f"t* = {tstart_opt_ms:.1f} ms")
ax_ts.axvline(3.0, color=LIGHT_TEXT, lw=0.7, ls="--", alpha=0.5, label="Phase III (3 ms)")
ax_ts.axhline(5.0, color=GREEN, lw=0.8, ls=":", alpha=0.7, label="pycbc target (ρ_syn=5)")
ax_ts.set_xlabel("Ringdown start t_start [ms]", color=LIGHT_TEXT, fontsize=9)
ax_ts.set_ylabel("SNR", color=LIGHT_TEXT, fontsize=9)
ax_ts.set_title("ρ_syn & ρ_obs vs ringdown start time\n(f=251 Hz, τ=4.0 ms)",
                color=LIGHT_TEXT, fontsize=9)
ax_ts.legend(fontsize=7, facecolor=DARK_BG, labelcolor=LIGHT_TEXT, framealpha=0.7)
ax_ts.set_xlim(TSTART_GRID_MS[0], TSTART_GRID_MS[-1])

# --- (f, τ) heat map ---
FF, TT = np.meshgrid(TAU_GRID_MS, F_GRID_HZ)
pcm = ax_ft.pcolormesh(TT, FF, rho_syn_grid, cmap="inferno", shading="auto")
ax_ft.scatter([f_opt], [tau_opt], color="white", s=60, marker="*", zorder=5,
              label=f"opt ({f_opt:.0f} Hz, {tau_opt:.1f} ms), ρ_syn={rho_syn_opt:.3f}")
ax_ft.axvline(F_NOM, color=LIGHT_TEXT, lw=0.7, ls="--", alpha=0.6, label="Nominal f")
ax_ft.axhline(TAU_NOM * 1e3, color=LIGHT_TEXT, lw=0.7, ls=":", alpha=0.6, label="Nominal τ")
cb = plt.colorbar(pcm, ax=ax_ft, pad=0.02)
cb.ax.tick_params(colors=LIGHT_TEXT, labelsize=7)
cb.set_label("ρ_syn", color=LIGHT_TEXT, fontsize=8)
ax_ft.set_xlabel("f_QNM [Hz]", color=LIGHT_TEXT, fontsize=9)
ax_ft.set_ylabel("τ_QNM [ms]", color=LIGHT_TEXT, fontsize=9)
ax_ft.set_title(f"ρ_syn over (f_QNM, τ_QNM) at t* = {tstart_opt_ms:.1f} ms",
                color=LIGHT_TEXT, fontsize=9)
ax_ft.legend(fontsize=7, facecolor=DARK_BG, labelcolor=LIGHT_TEXT, framealpha=0.7,
             loc="upper left")

# --- Posterior comparison ---
ax_p.plot(G_ref, post_ref, color=ACCENT, lw=1.5, ls="--",
          label=f"Phase III  ρ_syn={rho_syn_ref:.2f}  (t=3ms, nominal)")
ax_p.plot(G_opt, post_opt, color=YELLOW, lw=1.8,
          label=f"Phase VI   ρ_syn={rho_syn_opt:.2f}  (t={tstart_opt_ms:.1f}ms, optimal)")
ax_p.plot(G_pyc, post_pyc, color=GREEN, lw=1.8, ls="-.",
          label=f"pycbc est  ρ_syn=5.00  (hypothetical)")

for name, G_val in GAMMA_PREDS.items():
    ax_p.axvline(G_val, color=PRED_COLORS[name], lw=0.9, ls=":", alpha=0.85)
    ax_p.text(G_val + 0.01, ax_p.get_ylim()[1] * 0.98 if ax_p.get_ylim()[1] > 0 else 1,
              name, color=PRED_COLORS[name], fontsize=7, va="top")

ax_p.set_xlabel("Γ  (amplitude ratio A_obs / A_GR)", color=LIGHT_TEXT, fontsize=9)
ax_p.set_ylabel("p(Γ | data)", color=LIGHT_TEXT, fontsize=9)
ax_p.set_title("Single-event Rician posterior on Γ: Phase III vs Phase VI optimum vs pycbc forecast",
               color=LIGHT_TEXT, fontsize=9)
ax_p.legend(fontsize=8, facecolor=DARK_BG, labelcolor=LIGHT_TEXT, framealpha=0.7)
ax_p.set_xlim(0.0, 2.5)
ax_p.set_ylim(bottom=0)

fig.suptitle(
    "Phase VI — GW150914/H1 @ 16 kHz  |  Ringdown template optimisation  |  Sensitivity ceiling without pycbc",
    color=LIGHT_TEXT, fontsize=11, y=0.98,
)

out_dir  = os.path.join(os.path.dirname(__file__), "..", "misc")
fig_path = os.path.join(out_dir, "bh_phase_vi_gw150914_optimization.png")
fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print(f"\nFigure saved → {fig_path}")
print(f"Size: {os.path.getsize(fig_path)/1024:.1f} KB")
