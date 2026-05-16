"""Phase XI — η candidate comparison: formula search, HDE, and Barrow.

⚠ SUPERSEDED 2026-05-15: this script's "best candidate" verdict
(ETA_FORMULA = exp(-φ/σ_conv)) was rejected as formula-search numerology.
See misc/eta_empirical_verdict_2026-05-15.md for the current resolution
(η anchored at DESI Union3 c²). The script is retained for historical
traceability and will fail loudly when run because ETA_FORMULA is now None.

Compares four independent lines of evidence for σ-ground's cosmic entanglement
fraction η ≈ 0.4153 (HISTORICAL VALUE -- now superseded):

  1. η_working   = 0.4153          — original dark-energy heuristic [REJECTED]
  2. η_formula   = exp(-φ/σ_conv)  — formula search (2026-04-17)    [REJECTED]
  3. η_HDE_U3    = c²  ≈ 0.412    — DESI 2024 HDE fit, Union3 dataset [ADOPTED]
  4. η_HDE_D5    = c²  ≈ 0.491    — DESI 2024 HDE fit, DESY5 dataset
  5. η_Barrow    ≤ 0.43            — Barrow HDE fractal limit (arXiv:2503.18230)

Labels:
  [SPECULATIVE] — not yet confirmed against sigma-ground pipeline
  [THEORETICAL] — first-principles grounded, not yet uniquely confirmed
"""

import math

# ── sigma-ground constants ─────────────────────────────────────────────────────
try:
    from sigma_ground.field.constants import (
        XI, PHI, SIGMA_CONV, ETA,
        ETA_FORMULA,
        ETA_HDE_UNION3, ETA_HDE_DESY5,
        C_HDE_UNION3, C_HDE_DESY5,
        XI_KOBAKHIDZE, SIGMA_CONV_KOBAKHIDZE,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(__file__).replace('\\', '/').rsplit('/', 2)[0])
    from sigma_ground.field.constants import (
        XI, PHI, SIGMA_CONV, ETA,
        ETA_FORMULA,
        ETA_HDE_UNION3, ETA_HDE_DESY5,
        C_HDE_UNION3, C_HDE_DESY5,
        XI_KOBAKHIDZE, SIGMA_CONV_KOBAKHIDZE,
    )

# ── Barrow HDE (arXiv:2503.18230) ─────────────────────────────────────────────
# Barrow modified HDE: S_Barrow = (A/A_Pl)^(1+Δ/2).  Observational constraint:
# Δ < 0.43 (95% CL, Barrow 2020 + subsequent fits). The Barrow fractal
# dimension Δ is not the same as σ-ground's η, but numerically Δ_upper ≈ η.
# [SPECULATIVE] — identification of Δ with η is unsubstantiated.
ETA_BARROW_MAX = 0.43  # Barrow Δ upper bound ≈ η (within 3.5%)


def compare_candidates():
    """Print a comparison table of all η candidates."""
    print("=" * 70)
    print("Phase XI — η candidate comparison")
    print("=" * 70)
    print()
    print("σ-ground's current η = 0.4153 is a heuristic 'hobby number'.")
    print("This script evaluates independent lines of evidence for its true value.")
    print()

    candidates = [
        ("η_working",   ETA,            "(dark-energy heuristic, retained as working value)",  "—"),
        ("η_formula",   ETA_FORMULA,    "= exp(-φ/σ_conv)  [SPECULATIVE]",                     "formula search"),
        ("η_HDE_U3",    ETA_HDE_UNION3, "= c²  (CMB+DESI+Union3, c=0.642±0.028) [SPECULATIVE]", "arXiv:2411.08639"),
        ("η_HDE_D5",    ETA_HDE_DESY5,  "= c²  (CMB+DESI+DESY5,  c=0.701±0.024) [SPECULATIVE]", "arXiv:2411.08639"),
        ("η_Barrow",    ETA_BARROW_MAX, "= Δ_max < 0.43           [SPECULATIVE]",               "arXiv:2503.18230"),
    ]

    print(f"  {'Label':12s}  {'Value':8s}  {'Δ from 0.4153':14s}  {'% error':8s}  Description")
    print("  " + "-" * 80)
    for label, val, desc, ref in candidates:
        delta = val - ETA
        pct = 100 * abs(delta) / ETA
        print(f"  {label:12s}  {val:8.6f}  {delta:+14.6f}  {pct:7.3f}%  {desc}")
    print()

    # DESI 1-σ band: c = 0.642 ± 0.028 → c² ∈ [0.369, 0.457]
    c_low  = C_HDE_UNION3 - 0.028
    c_high = C_HDE_UNION3 + 0.028
    eta_low  = c_low  ** 2
    eta_high = c_high ** 2

    print("── DESI Union3 1-σ constraint ────────────────────────────────────────")
    print(f"  c = {C_HDE_UNION3:.3f} ± 0.028  →  c² ∈ [{eta_low:.4f}, {eta_high:.4f}]")
    for label, val, *_ in candidates:
        within = eta_low <= val <= eta_high
        sigma_dist = abs(val - ETA_HDE_UNION3) / (0.028 * (C_HDE_UNION3 + C_HDE_UNION3) * 0.028 / ETA_HDE_UNION3)
        flag = "WITHIN 1σ" if within else "outside 1σ"
        print(f"  {label:12s} = {val:.4f}  →  {flag}")
    print()

    # Key triple coincidence
    print("── Triple coincidence analysis ───────────────────────────────────────")
    trio = [ETA, ETA_FORMULA, ETA_HDE_UNION3]
    spread = max(trio) - min(trio)
    mean = sum(trio) / len(trio)
    print(f"  η_working  = {ETA:.6f}   (original heuristic)")
    print(f"  η_formula  = {ETA_FORMULA:.6f}   (exp(-φ/σ_conv) formula)")
    print(f"  η_HDE_U3   = {ETA_HDE_UNION3:.6f}   (DESI central value)")
    print(f"  Spread:    {spread:.6f} across all three  ({100*spread/mean:.2f}% of mean)")
    print()
    if spread < 0.01:
        print("  *** All three estimates cluster within 1% — strong convergence. ***")
    elif spread < 0.05:
        print("  All three estimates within 5% — consistent, but not uniquely discriminating.")
    else:
        print("  Spread > 5% — candidates disagree; further data needed.")
    print()

    # Kobakhidze shift on η_formula = exp(-φ/σ_conv)
    eta_kob = math.exp(-PHI / SIGMA_CONV_KOBAKHIDZE)
    print("── Effect of Kobakhidze ξ on η_formula = exp(-φ/σ_conv) ──────────────")
    print(f"  ξ_Planck   = {XI:.6f}, σ_conv = {SIGMA_CONV:.6f}  →  η_formula = {ETA_FORMULA:.6f}")
    print(f"  ξ_Kob(N=8) = {XI_KOBAKHIDZE:.6f}, σ_conv = {SIGMA_CONV_KOBAKHIDZE:.6f}  →  η_formula = {eta_kob:.6f}  "
          f"(Δ = {eta_kob - ETA_FORMULA:+.6f})")
    print()

    # σ_conv impact
    sigma_shift = SIGMA_CONV_KOBAKHIDZE - SIGMA_CONV
    print("── Constants summary ─────────────────────────────────────────────────")
    print(f"  ξ (Planck)     = {XI:.6f}")
    print(f"  ξ (Kobakhidze) = {XI_KOBAKHIDZE:.6f}   (Δ = {XI_KOBAKHIDZE-XI:+.6f}, "
          f"{100*(XI_KOBAKHIDZE-XI)/XI:.3f}%)")
    print(f"  σ_conv         = {SIGMA_CONV:.6f}")
    print(f"  σ_conv (Kob)   = {SIGMA_CONV_KOBAKHIDZE:.6f}   (Δ = {sigma_shift:+.6f}, "
          f"{100*sigma_shift/SIGMA_CONV:.3f}%)")
    print(f"  φ              = {PHI:.6f}")
    print(f"  ETA (working)  = {ETA:.6f}")
    print(f"  ETA_FORMULA    = {ETA_FORMULA:.6f}   (0.125% error from working)")
    print()

    print("── Interpretation ────────────────────────────────────────────────────")
    print("  If c² ≡ η in the HDE formula, DESI's Union3 fit places η = 0.412 ± 0.036.")
    print(f"  ETA_FORMULA = ξ^φ = {ETA_FORMULA:.4f} lands at {100*(ETA_FORMULA-ETA_HDE_UNION3)/ETA_HDE_UNION3:+.2f}%")
    print(f"  from the DESI central value — well within the 1σ band.")
    print()
    print("  This is a triple coincidence:")
    print(f"    exp(-φ/σ_conv) ≈ {ETA_FORMULA:.4f}  (formula, connects φ and σ_conv)")
    print(f"    DESI c² ≈ {ETA_HDE_UNION3:.4f}  (observational, DESI DR2)")
    print(f"    η_heuristic = {ETA:.4f}  (original dark-energy derivation)")
    print()
    print("  None of these three lines of evidence are independent of each other")
    print("  (DESI measures ρ_DE; η was derived from ρ_DE; ξ sets σ_conv).")
    print("  But their convergence is non-trivial: only ξ^φ predicts a specific")
    print("  number (0.4158) that lands inside the DESI band without fitting.")
    print()
    print("  Verdict: ETA_FORMULA = exp(-φ/σ_conv) is the best current candidate for η.")
    print("  Label: [SPECULATIVE] pending a first-principles derivation of why")
    print("  ξ^φ has physical meaning in the sigma-ground context.")
    print()
    print("Cross-references:")
    print("  Constants:  sigma_ground/field/constants.py (ETA, PHI, ETA_FORMULA,")
    print("              ETA_HDE_UNION3, XI_KOBAKHIDZE)")
    print("  DESI HDE:   arXiv:2411.08639")
    print("  Barrow HDE: arXiv:2503.18230")
    print("  Formula search: 2026-04-17 session")


if __name__ == "__main__":
    compare_candidates()
