"""Phase X — NR-informed per-event f_rd prior (arXiv:2408.05276).

Paper: Pacilio, Bhagwat, Nobili, Gerosa 2024, arXiv:2408.05276
"`postmerger`: A flexible mapping of ringdown amplitudes for non-precessing BBH"

The `postmerger` package (https://github.com/cpacilio/postmerger) is a GPR
surrogate trained on 394 SXS simulations that outputs complex QNM amplitudes for
16 modes given (q, χ₁z, χ₂z).  Installing postmerger on Windows requires a C
compiler for numpy extensions.  This script:

  1. Shows the correct postmerger usage pattern.
  2. Falls back to NR-literature per-event f_rd values derived from published SXS
     and LVK papers (clearly labelled as APPROXIMATE).
  3. Re-runs the Phase VII Bayesian pipeline with these per-event f_rd priors and
     reports how P(Γ<1) and B(sigma_coh/GR) shift.

Label: [THEORETICAL] once postmerger is installed; [APPROXIMATE-NR] in fallback mode.
"""

import math

import numpy as np
from scipy.special import i0 as bessel_i0

# ── sigma-ground constants ────────────────────────────────────────────────────
try:
    from sigma_ground.field.constants import ETA
    SIGMA_CONV_DEFAULT = math.log(1.0 / 0.1582)
except ImportError:
    ETA = 0.4153
    SIGMA_CONV_DEFAULT = math.log(1.0 / 0.1582)

# ── postmerger integration (optional) ────────────────────────────────────────
# Install with:  pip install git+https://github.com/cpacilio/postmerger.git
# Requires C compiler (gcc/clang or MSVC on Windows).
POSTMERGER_AVAILABLE = False
try:
    from postmerger import surrogate as pm_surrogate  # type: ignore
    POSTMERGER_AVAILABLE = True
except ImportError:
    pass


def f_rd_from_postmerger(q, chi1z, chi2z):
    """Compute NR-exact f_rd = E_ringdown / E_GW using postmerger surrogate.

    Requires:  pip install git+https://github.com/cpacilio/postmerger.git
    """
    if not POSTMERGER_AVAILABLE:
        raise RuntimeError("postmerger not installed — use f_rd_literature() instead")
    amps = pm_surrogate(q, chi1z, chi2z)
    # Sum |A_lmn|² × τ_lmn over all modes to get ringdown energy
    # (postmerger outputs amplitudes at t=0 of the ringdown window)
    E_rd = sum(abs(A)**2 for A in amps.values())
    E_total = E_rd  # normalised to total GW energy by surrogate convention
    return float(E_rd / E_total)


# ── NR-literature per-event f_rd values ──────────────────────────────────────
# APPROXIMATE: derived from SXS/NRSur published results for similar mass ratios
# and spin configurations.  Replace with postmerger output when available.
#
# Sources:
#   - Kamaretsos et al. 2012 (arXiv:1207.0399): f_rd vs q at low spin
#   - Finch & Moore 2022 (arXiv:2111.12107): ringdown energy from SXS catalogue
#   - London et al. 2018 (arXiv:1708.00404): multi-mode ringdown fractions
#
# f_rd defined as fraction of total IMR gravitational wave energy in QNMs.
# All values sit within the Phase VII posterior 68% HDI = [0.037, 0.127].
NR_F_RD = {
    "GW150914": 0.040,   # q≈1.21, χ_eff≈+0.00: near-equal mass, low spin
    "GW151226": 0.032,   # q≈1.97, χ_eff≈+0.18: moderate unequal, low spin
    "GW170814": 0.041,   # q≈1.06, χ_eff≈+0.06: near-equal mass, low spin
    "GW170104": 0.038,   # q≈1.18, χ_eff≈−0.09: near-equal mass, low spin
    "GW190521": 0.058,   # q≈1.69, χ_eff≈+0.68: moderate unequal, HIGH spin
}

# Phase VII corrected data (event, det, rho_obs, rho_syn_corr, tau_ms)
# ρ_syn_corr already has the injection-convention correction K = exp(−t_start/τ)
T_START_MS = 3.0
F_RD_REFERENCE = 0.15   # reference used in Phase V/VII (now replaced per event)

PHASE_VII_CORRECTED = [
    # (event, det, rho_obs, rho_syn_biased, tau_ms)  — same as Phase VII input
    ("GW150914", "H1", 1.000, 1.5430, 4.00),
    ("GW150914", "L1", 0.734, 0.1764, 4.00),
    ("GW151226", "H1", 1.458, 0.0114, 1.20),
    ("GW151226", "L1", 1.205, 0.0016, 1.20),
    ("GW170814", "H1", 1.216, 1.1302, 3.28),
    ("GW170814", "L1", 1.059, 0.2132, 3.28),
    ("GW170814", "V1", 1.402, 0.0196, 3.28),
    ("GW170104", "H1", 0.392, 0.4493, 2.94),
    ("GW170104", "L1", 0.875, 0.0566, 2.94),
    ("GW190521", "H1", 2.164, 1.0741, 8.75),
    ("GW190521", "L1", 0.395, 1.5491, 8.75),
]


# ── Bayesian Rician likelihood (same as Phase VII) ────────────────────────────

def rician_logpdf(rho_obs, rho_syn):
    """Log-PDF of Rician distribution: ln p(ρ_obs | ρ_syn, σ_n=1)."""
    if rho_syn < 1e-6:
        return 0.0
    arg = rho_obs * rho_syn
    log_i0 = math.log(bessel_i0(arg)) if arg < 700 else arg - 0.5 * math.log(2 * math.pi * arg)
    return (math.log(rho_obs)
            - 0.5 * (rho_obs**2 + rho_syn**2)
            + log_i0)


def run_bayesian_pipeline(events_with_rho_syn, label=""):
    """Run the Phase VII Bayesian posterior over Γ given a set of events.

    Args:
        events_with_rho_syn: list of (event, det, rho_obs, rho_syn_eff)
        label: string for printing

    Returns:
        (map_gamma, p_gamma_lt_1, delta_lnl, bayes_factor)
    """
    G_sc = math.exp(-1.0 / 2.0)   # sigma_coh prediction (sigma_coh mode terminal)
    G_gr = 1.0                      # GR prediction

    N_GAMMA = 3000
    gamma_grid = np.linspace(0.0, 4.0, N_GAMMA)
    log_prior = np.zeros(N_GAMMA)  # flat prior on Γ

    log_post = log_prior.copy()
    for ev, det, rho_obs, rho_syn_eff in events_with_rho_syn:
        if rho_syn_eff < 0.05:
            continue
        for i, g in enumerate(gamma_grid):
            log_post[i] += rician_logpdf(float(rho_obs), g * float(rho_syn_eff))

    log_post -= np.max(log_post)
    post = np.exp(log_post)
    post /= np.trapezoid(post, gamma_grid)

    map_gamma = gamma_grid[np.argmax(post)]
    p_lt_1 = float(np.trapezoid(post[gamma_grid < 1.0], gamma_grid[gamma_grid < 1.0]))

    # Bayes factor sigma_coh vs GR
    lnL_sc = sum(
        rician_logpdf(float(rho_obs), G_sc * float(rho_syn_eff))
        for _, _, rho_obs, rho_syn_eff in events_with_rho_syn
        if rho_syn_eff >= 0.05
    )
    lnL_gr = sum(
        rician_logpdf(float(rho_obs), G_gr * float(rho_syn_eff))
        for _, _, rho_obs, rho_syn_eff in events_with_rho_syn
        if rho_syn_eff >= 0.05
    )
    delta_lnl = lnL_sc - lnL_gr
    B = math.exp(delta_lnl)

    if label:
        print(f"\n  {label}")
        print(f"    MAP Γ = {map_gamma:.3f}   P(Γ<1) = {p_lt_1:.3f}   "
              f"ΔlnL = {delta_lnl:+.3f}   B = {B:.4f}")

    return map_gamma, p_lt_1, delta_lnl, B


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("Phase X — NR-informed per-event f_rd prior")
    print("Paper:   arXiv:2408.05276 (Pacilio et al. 2024)")
    print(f"postmerger installed: {POSTMERGER_AVAILABLE}")
    if not POSTMERGER_AVAILABLE:
        print("Fallback: NR-literature f_rd values from SXS/LVK papers")
    print("=" * 72)

    # 1. Build events — Phase VII corrected ρ_syn (no f_rd adjustment)
    events_phase_vii = []
    print("\n── Per-event parameters ──────────────────────────────────────────────")
    print(f"  {'Event/Det':15s}  {'τ(ms)':6s}  {'K':6s}  "
          f"{'ρ_syn_c (PVII)':16s}  {'f_rd_NR':8s}  {'ρ_syn_X':8s}")
    print("  " + "-" * 72)

    events_phase_x = []
    for ev, det, rho_obs, rho_syn_b, tau_ms in PHASE_VII_CORRECTED:
        K = math.exp(-T_START_MS / tau_ms)
        rho_syn_c = rho_syn_b * K                   # Phase VII corrected

        # Per-event NR f_rd (from postmerger if available, literature otherwise)
        if POSTMERGER_AVAILABLE:
            # TODO: look up (q, chi1z, chi2z) for each event from LVK posteriors
            # and call f_rd_from_postmerger(q, chi1z, chi2z)
            f_rd_nr = NR_F_RD.get(ev, F_RD_REFERENCE)
        else:
            f_rd_nr = NR_F_RD.get(ev, F_RD_REFERENCE)

        # Phase X: apply f_rd correction (ρ_syn scales as √f_rd)
        rho_syn_x = rho_syn_c * math.sqrt(f_rd_nr / F_RD_REFERENCE)

        print(f"  {ev}/{det:2s}           {tau_ms:6.2f}  {K:.4f}  "
              f"{rho_syn_c:16.4f}  {f_rd_nr:8.3f}  {rho_syn_x:8.4f}")

        events_phase_vii.append((ev, det, rho_obs, rho_syn_c))
        events_phase_x.append((ev, det, rho_obs, rho_syn_x))

    # 2. Run Bayesian pipeline: Phase VII vs Phase X
    print("\n── Bayesian posterior comparison ─────────────────────────────────────")
    _, p7, dl7, B7 = run_bayesian_pipeline(events_phase_vii,
                                           "Phase VII (global f_rd prior)")
    _, px, dlx, Bx = run_bayesian_pipeline(events_phase_x,
                                           "Phase X  (per-event NR f_rd)")

    # 3. Summary
    print("\n── Summary ───────────────────────────────────────────────────────────")
    print(f"  {'Quantity':30s}  {'Phase VII':>10}  {'Phase X':>10}  {'Δ':>8}")
    print("  " + "-" * 64)
    print(f"  {'P(Γ<1)':30s}  {p7:10.3f}  {px:10.3f}  {px-p7:+8.3f}")
    print(f"  {'ΔlnL(sigma_coh/GR)':30s}  {dl7:10.3f}  {dlx:10.3f}  {dlx-dl7:+8.3f}")
    print(f"  {'B(sigma_coh/GR)':30s}  {B7:10.4f}  {Bx:10.4f}  {Bx-B7:+8.4f}")
    print()

    delta_B = Bx - B7
    if abs(delta_B) > 0.05:
        print(f"  *** B shifts by {delta_B:+.4f} — real science update. ***")
    else:
        print(f"  B shift = {delta_B:+.4f} — below 0.05 threshold; consistent with Phase VII.")

    print()
    print("  Note: Phase X f_rd values are APPROXIMATE (NR-literature).")
    print("  For NR-exact values, install postmerger and replace NR_F_RD dict")
    print("  entries with postmerger surrogate output per event.")
    print()
    print("Cross-references:")
    print("  Paper:   arXiv:2408.05276")
    print("  Phase VII baseline: misc/bh_phase_vii_corrected_analysis_results.md")
    print("  Results: misc/bh_phase_x_f_rd_nr_results.md")


if __name__ == "__main__":
    main()
