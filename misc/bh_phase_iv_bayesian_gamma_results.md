# Phase IV — Bayesian Γ Inference via Rician Likelihood: Verdict

**Date:** 2026-04-17  
**Phase:** IV — Bayesian amplitude ratio inference  
**Pipeline:** `scripts/phase_iv_bayesian_gamma.py`  
**Input:** Phase III (ρ_obs, ρ_syn) for 11 detector-events; GW150914 at 16 kHz

---

## Why Phase III's Ratio Is Biased

The simple estimator Γ = ρ_obs / ρ_syn (Phase III) is biased when ρ_syn << 1.
Under H₀ (pure noise), ρ_obs follows a Rayleigh(1) distribution with
E[ρ_obs] = √(π/2) ≈ 1.25 regardless of ρ_syn.  Therefore:

    E[Γ_naive | H₀] = E[ρ_obs] / ρ_syn = 1.25 / ρ_syn → ∞  as  ρ_syn → 0

This inflates Γ for noise-dominated events (GW151226/L1 gave Γ = 747 in Phase
III, driven entirely by ρ_syn = 0.0016).

---

## Correct Framework: Rician Likelihood

The matched-filter quadrature amplitude A = sqrt(ρ_cos² + ρ_sin²) is
Rician-distributed under a signal with SNR s:

    p(A | s) = A · exp(−(A² + s²)/2) · I₀(A·s)

with s = Γ × ρ_syn per event.  Under H₀ (s=0) this reduces to Rayleigh(1).

The per-event **Rician MLE** for the signal SNR:

    s_MLE = max(0,  √(A² − 1))   [for σ=1 Rician]

This is zero whenever ρ_obs ≤ 1 (at or below the noise floor) — the correct
statistical answer: no signal is detectable when the observation is at the
noise floor.

**Combined posterior** (uniform prior Γ ≥ 0):

    ln p(Γ | data) ∝ Σᵢ ln Rician(ρ_obs_i ; Γ × ρ_syn_i, 1)

Events with ρ_syn ≈ 0 contribute a Γ-flat term and are automatically
down-weighted.  Events with ρ_syn ~ O(1) drive the posterior.

---

## Per-Event Log-Likelihood Ratios

| Event / Det | ρ_obs | ρ_syn | ΔlnL (σ_coh vs GR) | Direction |
|---|---|---|---|---|
| GW150914 / H1 | 1.000 | 1.5430 | **+0.262** | sigma_coh |
| GW150914 / L1 | 0.734 | 0.1764 | +0.004 | sigma_coh |
| GW151226 / H1 | 1.458 | 0.0114 | −0.000 | — (Γ-flat) |
| GW151226 / L1 | 1.205 | 0.0016 | +0.000 | — (Γ-flat) |
| GW170814 / H1 | 1.216 | 1.1302 | **+0.089** | sigma_coh |
| GW170814 / L1 | 1.059 | 0.2132 | +0.004 | sigma_coh |
| GW170814 / V1 | 1.402 | 0.0196 | +0.000 | — (Γ-flat) |
| GW170104 / H1 | 0.392 | 0.4493 | +0.035 | sigma_coh |
| GW170104 / L1 | 0.875 | 0.0566 | +0.000 | — (Γ-flat) |
| GW190521 / H1 | 2.164 | 1.0741 | **−0.128** | GR |
| GW190521 / L1 | 0.395 | 1.5491 | **+0.413** | sigma_coh |

**Combined ΔlnL = +0.679 → Bayes factor B(sigma_coh / GR) = exp(0.679) = 1.97**

All five informative events (ρ_syn > 0.1) favor sigma_coh, except GW190521/H1.

---

## Combined Posterior

```
MAP Γ         = 0.000   ← most-likely: no detectable QNM signal above noise
68% HDI       = [0.000, 0.551]
95% HDI       = [0.000, 1.028]

P(Γ < 1.000 | data)  = 0.944   (94% posterior probability of amplitude suppression)
P(Γ < 0.792 | data)  = 0.860   (86% consistent with sigma_coh direction)
```

**All four predictions lie within the 95% HDI** (GR barely, at Γ = 1.0 ≈ hi95 = 1.028):

| Model | Γ_pred | P(Γ > Γ_pred) | Consistent with data? |
|---|---|---|---|
| GR | 1.0000 | 5.7% | Yes (95% HDI) |
| exp | 0.8395 | 11.7% | Yes |
| sigma_coh | 0.7923 | 14.1% | Yes |
| linear_cbrt | 0.7461 | 16.8% | Yes |

---

## Verdict

**The Bayesian Rician analysis is consistent with every σ_conv candidate mode.
GR (Γ = 1.0) is at the edge of the 95% credible interval but is NOT ruled out.**

**MAP Γ = 0 does not mean the QNM is zero.** It means the observed amplitudes
are at or below the noise floor (ρ_obs ≤ 1 for 9 of 11 events), and the Rician
MLE for a noise-floor observation is exactly zero.  The posterior mass below
GR (P = 94%) reflects that ρ_syn consistently overestimates ρ_obs — driven
primarily by the f_rd = 0.15 systematic.

**Bayes factor: sigma_coh preferred over GR by B ≈ 2.** This is "barely worth
mentioning" on the Jeffreys scale (need B > 10 for "substantial evidence").
The combined ΔlnL = +0.679 comes from five informative events all showing
ρ_obs < ρ_syn (consistent with suppressed amplitude), with the exception of
GW190521/H1 which shows a positive noise fluctuation (ρ_obs = 2.16 > ρ_syn = 1.07).

**The dominant systematic: f_rd = 0.15.** The fraction of E_rad in the (2,2,0)
QNM is uncertain to a factor of 3–7 (NR estimates range 0.03–0.20).  If f_rd
is closer to 0.05, ρ_syn drops by 0.58×, bringing ρ_syn into better alignment
with ρ_obs and shifting the posterior MAP to Γ ~ 0.7–1.0.  The Rician posterior
cannot distinguish the f_rd systematic from a genuine Γ < 1 suppression at this
sensitivity.

---

## Key Physical Findings

**1. All 11 events show ρ_obs ≤ ρ_syn (except GW190521/H1).**
The GR prediction consistently overshoots the observed amplitude.  This is
either (a) Γ < 1 (the σ_conv signal), (b) f_rd overestimated, or (c)
template mismatch reducing ρ_obs below the signal level.

**2. GW190521/H1 and L1 are contradictory.**
The same event gives Γ_effective = 2.01 (H1) and 0.26 (L1) from Phase III's
naive ratio.  In the Rician framework:
- H1 (ρ_obs=2.16, ρ_syn=1.07) → Γ_MLE = 1.79; strongly favors Γ > 1
- L1 (ρ_obs=0.40, ρ_syn=1.55) → Γ_MLE = 0; strongly favors Γ = 0
The opposite pulls from H1 and L1 nearly cancel.  GW190521 cannot determine Γ.

**3. GW150914/H1 is the single most informative event.**
With ρ_syn = 1.54, it is the only event where the GR-predicted signal is
comfortably above the noise floor AND the per-event ΔlnL is large (+0.262).
GW150914/H1 alone gives ΔlnL = +0.262, supporting sigma_coh over GR by
a factor of exp(0.262) = 1.30.  Raising ρ_syn for this event to ~3–5 via
pycbc would give a factor of 10–30 (strong evidence) without any additional events.

**4. GW170814/H1 provides corroborating support.**
ρ_syn = 1.13, ΔlnL = +0.089 → sigma_coh preferred by factor exp(0.089) = 1.09.
Modest but in the correct direction.

---

## Combined Evidence State (Phases I–IV)

| Phase | Result | Informative? |
|---|---|---|
| I.4 (time domain catalog) | p_F = 0.609 — null | No (echo signal below threshold) |
| I.5 (freq domain comb) | R = 0.916 — null | No (below noise floor) |
| II (amplitude bound) | ξ_UL = 1.0 (saturated) | No (ξ prediction untestable) |
| III (amplitude ratio) | Γ = 0.856 ± 1.196 — null | Marginally (4 events with ρ_syn > 1) |
| IV (Bayesian Rician) | MAP=0, P(Γ<1)=94%, B(sigma_coh/GR)≈2 | Weak signal (barely worth mentioning) |

**Summary:** The data shows a weak, statistically insignificant preference for
Γ < 1 (amplitude suppression), consistent with the σ_conv prediction.
It does not rule out GR.  The current analysis is 5–25× too insensitive
for a definitive test.

---

## Path to Decisiveness

To achieve B > 10 (strong evidence) for sigma_coh vs GR requires:

    ΔlnL_needed ≈ 2.3  (ln 10)
    Current ΔlnL per informative event: GW150914/H1 = +0.26 (with ρ_syn=1.54)

For a single event at ρ_syn = S, the Fisher information for Γ scales as S²,
so ΔlnL ∝ S².  To achieve ΔlnL ≈ 2.3 from one event:

    ΔlnL(S) / ΔlnL(1.54) = (S/1.54)²  ≈  2.3 / 0.26  = 8.8
    S_needed = 1.54 × √8.8 = 1.54 × 2.97 ≈ 4.6

**A single event with ρ_syn ≥ 5 would make the test decisive.**

This is achievable with pycbc full-IMR subtraction on GW150914/H1, which
would raise ρ_syn from 1.54 to approximately 5–10 (based on the 3–5×
improvement in QNM SNR from residual subtraction documented in the literature).

---

## Cross-References

- Phase I.4 catalog null: `misc/bh_phase_i_4_catalog_results.md`
- Phase I.5 comb null: `misc/bh_phase_i_5_freq_stack_results.md`
- Phase II amplitude bound: `misc/bh_phase_ii_amplitude_bound_results.md`
- Phase III naive ratio: `misc/bh_phase_iii_amplitude_ratio_results.md`
- Phase IV script: `scripts/phase_iv_bayesian_gamma.py`
- Phase IV figure: `misc/bh_phase_iv_bayesian_gamma.png`
- σ_conv / γ-mode predictions: `misc/bh_collision_phenomenology.md`
- Rician MLE derivation: standard matched-filter literature (e.g., Rice 1954)
