"""
Phase VIII — Sensitivity forecast: B(sigma_coh/GR) vs ρ_syn and catalog depth.

Phase VII established the corrected null: B = 1.20, no evidence.
Phase VIII answers: what improvement is needed, and from where?

Three forecast axes:
  A. Single-event B vs ρ_syn (GW150914/H1 pycbc improvement path)
  B. N-event B vs catalog depth (GWTC-3 coherent stack scaling)
  C. Combined: pycbc on GW150914 + N GWTC-3 events

Method:
  For each scenario, compute E[ΔlnL] under both hypotheses:
    - Under H_GR    : ρ_obs ~ Rician(ρ_syn, 1)      [GR true]
    - Under H_sc    : ρ_obs ~ Rician(Γ_sc × ρ_syn, 1)  [sigma_coh true]

  E[ΔlnL | H] = ∫ p(ρ_obs | H) × [lnL_sc(ρ_obs) − lnL_GR(ρ_obs)] dρ_obs

  E[ΔlnL] from N independent events scales as N × E[ΔlnL_per_event].
  B_expected = exp(N × E[ΔlnL_per_event]).

Current corrected baseline (Phase VII):
  GW150914/H1:  ρ_syn_corr = 0.729,  observed ΔlnL = +0.052
  GW190521/L1:  ρ_syn_corr = 1.099,  observed ΔlnL = +0.208
  GW190521/H1:  ρ_syn_corr = 0.762,  observed ΔlnL = −0.094
  Combined:     ΔlnL = +0.183,  B = 1.20

Output:
  - Figure: misc/bh_phase_viii_sensitivity_forecast.png
  - Verdict: misc/bh_phase_viii_sensitivity_forecast_results.md
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import i0 as bessel_i0

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from sigma_ground.field.constants import ETA

ETA_VAL   = float(ETA)
GAMMA_SC  = round(1.0 - ETA_VAL / 2.0, 4)   # 0.7924
GAMMA_GR  = 1.0000

# ── Corrected Phase VII baseline (11 events) ─────────────────────────────────
BASELINE_DLNL = 0.1827   # combined corrected ΔlnL
BASELINE_B    = math.exp(BASELINE_DLNL)
BASELINE_CORR_RHOSYNS = [
    0.7289, 0.0833, 0.0009, 0.0001,   # GW150914 H1/L1, GW151226 H1/L1
    0.4528, 0.0854, 0.0079,            # GW170814 H1/L1/V1
    0.1619, 0.0204,                    # GW170104 H1/L1
    0.7623, 1.0995,                    # GW190521 H1/L1
]

# ── Rician helpers ────────────────────────────────────────────────────────────
def rician_logpdf(A, s):
    if A <= 0:
        return -1e30
    s = max(s, 0.0)
    return math.log(A) - (A**2 + s**2) / 2.0 + float(np.log(bessel_i0(A * s) + 1e-300))

def rician_pdf(A, s):
    return float(np.exp(rician_logpdf(A, s)))


def expected_delta_lnl(rho_syn, gamma_true, n_pts=500):
    """
    E[ΔlnL | gamma_true] = ∫ Rician(A; gamma_true*rho_syn) × [lnL_sc(A) - lnL_GR(A)] dA

    s_true = gamma_true × rho_syn  (signal SNR under H_true)
    s_sc   = GAMMA_SC  × rho_syn  (sigma_coh prediction)
    s_gr   = GAMMA_GR  × rho_syn  (GR prediction)
    """
    s_true = gamma_true * rho_syn
    s_sc   = GAMMA_SC  * rho_syn
    s_gr   = GAMMA_GR  * rho_syn

    # Integrate over A ∈ [0, max(s_true + 5, 4)] — covers > 99.9% of Rician mass
    A_max = max(s_true + 5.0, 4.0)
    A_arr = np.linspace(1e-6, A_max, n_pts)
    dA    = A_arr[1] - A_arr[0]

    integrand = np.array([
        rician_pdf(A, s_true) * (rician_logpdf(A, s_sc) - rician_logpdf(A, s_gr))
        for A in A_arr
    ])
    return float(np.trapezoid(integrand, A_arr))


# ── A. Single-event forecast: GW150914/H1 pycbc path ─────────────────────────
print("A. Single-event forecast (GW150914/H1) …")
rho_syn_range = np.concatenate([
    np.linspace(0.5,  2.0,  20),
    np.linspace(2.0,  5.0,  20),
    np.linspace(5.0, 20.0,  30),
])

e_dlnl_gr_single = np.array([expected_delta_lnl(s, GAMMA_GR) for s in rho_syn_range])
e_dlnl_sc_single = np.array([expected_delta_lnl(s, GAMMA_SC) for s in rho_syn_range])

# Include fixed contribution from the other 10 corrected events
dlnl_other_10 = BASELINE_DLNL - 0.0519   # subtract GW150914/H1 contribution

# Total expected ΔlnL = this event + other 10 (unchanged)
total_dlnl_gr = e_dlnl_gr_single + dlnl_other_10
total_dlnl_sc = e_dlnl_sc_single + dlnl_other_10

# Key milestones on the pycbc path
PYCBC_RANGE = [0.729, 2.0, 5.0, 10.0]  # ρ_syn values
pycbc_B_gr  = [math.exp(expected_delta_lnl(s, GAMMA_GR) + dlnl_other_10)
               for s in PYCBC_RANGE]
pycbc_B_sc  = [math.exp(expected_delta_lnl(s, GAMMA_SC) + dlnl_other_10)
               for s in PYCBC_RANGE]

print(f"  ρ_syn  |  B (if GR true)  |  B (if sigma_coh true)")
for s, bg, bsc in zip(PYCBC_RANGE, pycbc_B_gr, pycbc_B_sc):
    print(f"  {s:5.2f}  |  {bg:8.3f}         |  {bsc:8.3f}")


# ── B. GWTC-3 coherent stack forecast ────────────────────────────────────────
print("\nB. GWTC-3 coherent stack forecast …")

# Typical corrected event: M_f~50 M☉, τ~3.5ms → K=exp(-3/3.5)=0.42
# Biased ρ_syn from Phase II: ρ̄_QNM = 1.075; corrected: ~1.075×0.42 ≈ 0.45
RHO_SYN_TYPICAL_CORR = 0.45   # representative corrected ρ_syn per event

e_dlnl_gr_evt = expected_delta_lnl(RHO_SYN_TYPICAL_CORR, GAMMA_GR)
e_dlnl_sc_evt = expected_delta_lnl(RHO_SYN_TYPICAL_CORR, GAMMA_SC)

N_events = np.arange(0, 5001, 25)
N_events[0] = 1

# Start from Phase VII corrected baseline and add N new events
b_gwtc_gr = np.array([math.exp(BASELINE_DLNL + n * e_dlnl_gr_evt) for n in N_events])
b_gwtc_sc = np.array([math.exp(BASELINE_DLNL + n * e_dlnl_sc_evt) for n in N_events])

# Find N to reach B > 10
n_for_B10_gr = float("nan")
n_for_B10_sc = float("nan")
for n in N_events:
    b = math.exp(BASELINE_DLNL + n * e_dlnl_gr_evt)
    if b > 10.0 and math.isnan(n_for_B10_gr):
        n_for_B10_gr = n
    b = math.exp(BASELINE_DLNL + n * e_dlnl_sc_evt)
    if b > 10.0 and math.isnan(n_for_B10_sc):
        n_for_B10_sc = n

print(f"  E[ΔlnL per event] under GR: {e_dlnl_gr_evt:+.5f}  (per ρ_syn_typ={RHO_SYN_TYPICAL_CORR})")
print(f"  E[ΔlnL per event] under sigma_coh: {e_dlnl_sc_evt:+.5f}")
print(f"  N to reach B>10 (if GR true): {n_for_B10_gr}")
print(f"  N to reach B>10 (if sigma_coh true): {n_for_B10_sc}")


# ── C. Combined: pycbc on GW150914 + N GWTC-3 events ────────────────────────
print("\nC. Combined forecast (pycbc GW150914 + GWTC-3) …")

# pycbc boost: GW150914/H1 ρ_syn → 5.0
RHO_SYN_PYCBC = 5.0
e_dlnl_gr_pyc = expected_delta_lnl(RHO_SYN_PYCBC, GAMMA_GR)
e_dlnl_sc_pyc = expected_delta_lnl(RHO_SYN_PYCBC, GAMMA_SC)
boost_gr = e_dlnl_gr_pyc - expected_delta_lnl(0.729, GAMMA_GR)
boost_sc = e_dlnl_sc_pyc - expected_delta_lnl(0.729, GAMMA_SC)

baseline_after_pycbc_gr = BASELINE_DLNL + boost_gr
baseline_after_pycbc_sc = BASELINE_DLNL + boost_sc

b_combined_gr = np.array([math.exp(baseline_after_pycbc_gr + n * e_dlnl_gr_evt) for n in N_events])
b_combined_sc = np.array([math.exp(baseline_after_pycbc_sc + n * e_dlnl_sc_evt) for n in N_events])

n_for_B10_combined_gr = float("nan")
n_for_B10_combined_sc = float("nan")
for n in N_events:
    b = math.exp(baseline_after_pycbc_gr + n * e_dlnl_gr_evt)
    if b > 10.0 and math.isnan(n_for_B10_combined_gr):
        n_for_B10_combined_gr = n
    b = math.exp(baseline_after_pycbc_sc + n * e_dlnl_sc_evt)
    if b > 10.0 and math.isnan(n_for_B10_combined_sc):
        n_for_B10_combined_sc = n

print(f"  After pycbc (ρ_syn 0.729→5.0): ΔlnL boost = {boost_sc:+.4f} (sigma_coh)")
print(f"  B immediately after pycbc (if GR true): {math.exp(baseline_after_pycbc_gr):.2f}")
print(f"  B immediately after pycbc (if sigma_coh true): {math.exp(baseline_after_pycbc_sc):.2f}")
print(f"  N extra events for B>10 after pycbc (if GR true): {n_for_B10_combined_gr}")
print(f"  N extra events for B>10 after pycbc (if sigma_coh true): {n_for_B10_combined_sc}")


# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Phase VIII — Sensitivity Forecast Summary")
print("=" * 70)
print(f"Current corrected baseline:  ΔlnL = {BASELINE_DLNL:+.3f}  B = {BASELINE_B:.2f}")
print()
print("To reach B > 10 (strong evidence) if sigma_coh is true:")
print(f"  Option A: GW150914/H1 ρ_syn → 5.0  alone  → B = {math.exp(baseline_after_pycbc_sc):.1f} (after pycbc)")
print(f"            ρ_syn needed for B>10 alone: ")

# Find single-event ρ_syn for B>10 if sigma_coh true
for s in np.linspace(1.0, 25.0, 500):
    if math.exp(expected_delta_lnl(s, GAMMA_SC) + dlnl_other_10) > 10.0:
        print(f"            ρ_syn ≥ {s:.1f}")
        break

print(f"  Option B: GWTC-3 stack ({RHO_SYN_TYPICAL_CORR:.2f}/event):  need {n_for_B10_combined_sc} extra events + pycbc")
print(f"            or {n_for_B10_sc} events without pycbc")
print()
print("To reach B > 10 if GR is true (rule out sigma_coh):")
print(f"  Need {n_for_B10_combined_gr} extra GWTC-3 events after pycbc, or {n_for_B10_gr} without pycbc")
print("=" * 70)


# ── Figure ────────────────────────────────────────────────────────────────────
DARK_BG    = "#0d1117"
LIGHT_TEXT = "#e6edf3"
ACCENT     = "#58a6ff"
ORANGE     = "#f0883e"
GREEN      = "#3fb950"
RED        = "#ff7b72"
PURPLE     = "#bc8cff"
YELLOW     = "#e3b341"

fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor=DARK_BG)
fig.subplots_adjust(left=0.06, right=0.97, wspace=0.32)

for ax in axes:
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=LIGHT_TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.axhline(10.0, color=YELLOW, lw=0.8, ls="--", alpha=0.7, label="B = 10 (strong evidence)")
    ax.axhline(1.0,  color=LIGHT_TEXT, lw=0.5, ls=":",  alpha=0.4, label="B = 1 (no preference)")

# Panel A: single-event ρ_syn sweep
ax = axes[0]
ax.plot(rho_syn_range, np.exp(total_dlnl_sc), color=GREEN,  lw=2.0, label="if sigma_coh true")
ax.plot(rho_syn_range, np.exp(total_dlnl_gr), color=RED,    lw=2.0, label="if GR true")
ax.axvline(0.729, color=ACCENT,  lw=1.0, ls=":", label=f"Current ρ_syn = 0.73")
ax.axvline(1.423, color=ORANGE,  lw=1.0, ls=":", label=f"t* opt ρ_syn = 1.42")
ax.axvline(5.0,   color=GREEN,   lw=1.0, ls="--", alpha=0.6, label=f"pycbc ρ_syn ≈ 5")
ax.set_xlabel("ρ_syn for GW150914/H1 (corrected)", color=LIGHT_TEXT, fontsize=9)
ax.set_ylabel("Bayes factor B(sigma_coh / GR)", color=LIGHT_TEXT, fontsize=9)
ax.set_title("Panel A — GW150914/H1 pycbc improvement path\n(other 10 events fixed at corrected values)",
             color=LIGHT_TEXT, fontsize=9)
ax.set_yscale("log")
ax.set_xlim(0.5, 10.0)
ax.set_ylim(0.5, 200)
ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor=LIGHT_TEXT, framealpha=0.7)

# Panel B: GWTC-3 stack (no pycbc)
ax = axes[1]
ax.plot(N_events, b_gwtc_sc, color=GREEN, lw=2.0, label="if sigma_coh true")
ax.plot(N_events, b_gwtc_gr, color=RED,   lw=2.0, label="if GR true")
ax.axvline(90,  color=PURPLE, lw=0.9, ls=":", alpha=0.8, label="~90 BBH (GWTC-3)")
ax.axvline(180, color=PURPLE, lw=0.9, ls="--", alpha=0.6, label="~180 det-events")
ax.set_xlabel("N additional corrected events (ρ_syn≈0.45/event)", color=LIGHT_TEXT, fontsize=9)
ax.set_ylabel("Bayes factor B(sigma_coh / GR)", color=LIGHT_TEXT, fontsize=9)
ax.set_title(f"Panel B — GWTC-3 stack only\n(no pycbc; ρ_syn_typ={RHO_SYN_TYPICAL_CORR}/event)",
             color=LIGHT_TEXT, fontsize=9)
ax.set_yscale("log")
ax.set_xlim(0, 5000)
ax.set_ylim(0.5, 200)
ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor=LIGHT_TEXT, framealpha=0.7)

# Panel C: pycbc + GWTC-3
ax = axes[2]
ax.plot(N_events, b_combined_sc, color=GREEN,  lw=2.0, label="if sigma_coh true")
ax.plot(N_events, b_combined_gr, color=RED,    lw=2.0, label="if GR true")
ax.plot(N_events, b_gwtc_sc,    color=GREEN,   lw=1.0, ls="--", alpha=0.4, label="no pycbc (ref)")
ax.plot(N_events, b_gwtc_gr,    color=RED,     lw=1.0, ls="--", alpha=0.4)
ax.axvline(90,   color=PURPLE, lw=0.9, ls=":",  alpha=0.8, label="~90 BBH (GWTC-3)")
ax.axvline(3500, color=ORANGE, lw=0.9, ls="--", alpha=0.5, label="~3500 (O4+O5 projection)")
ax.set_xlabel("N additional corrected events after pycbc", color=LIGHT_TEXT, fontsize=9)
ax.set_ylabel("Bayes factor B(sigma_coh / GR)", color=LIGHT_TEXT, fontsize=9)
ax.set_title(f"Panel C — pycbc (ρ_syn 0.73→5.0) + GWTC-3/O4 stack\nSolid = with pycbc, dashed = without",
             color=LIGHT_TEXT, fontsize=9)
ax.set_yscale("log")
ax.set_xlim(0, 5000)
ax.set_ylim(0.5, 200)
ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor=LIGHT_TEXT, framealpha=0.7)

fig.suptitle(
    "Phase VIII — Sensitivity Forecast: B(sigma_coh/GR) Roadmap  |  Phase VII corrected baseline",
    color=LIGHT_TEXT, fontsize=11, y=0.99,
)

out_dir  = os.path.join(os.path.dirname(__file__), "..", "misc")
fig_path = os.path.join(out_dir, "bh_phase_viii_sensitivity_forecast.png")
fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print(f"\nFigure saved → {fig_path}")
print(f"Size: {os.path.getsize(fig_path)/1024:.1f} KB")
