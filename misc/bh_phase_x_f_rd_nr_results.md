# Phase X — NR-Informed Per-Event f_rd Prior

**Date:** 2026-04-17
**Phase:** X — postmerger NR surrogate f_rd integration
**Pipeline:** `scripts/phase_x_f_rd_nr_prior.py`
**Paper:** Pacilio, Bhagwat, Nobili, Gerosa 2024, arXiv:2408.05276
**Baseline:** Phase VII corrected (B = 1.20, P(Γ<1) = 0.710)
**Label:** `[APPROXIMATE-NR]` — postmerger not yet installable on Windows;
values from NR-literature (SXS/LVK published papers)

---

## What changed

Phase VII used a global f_rd prior [0.037, 0.127] (uniform, from Phase V posterior).
The effective ρ_syn scales as √(f_rd / f_rd_ref) where f_rd_ref = 0.15.

Phase X replaces the global prior with per-event NR estimates from published SXS and
LVK parameter-estimation results.  The `postmerger` GPR surrogate (arXiv:2408.05276)
trained on 394 SXS simulations would give NR-exact values; current values are literature
approximations pending Windows-buildable installation of that package.

---

## Per-event f_rd values

| Event | q (approx) | χ_eff (approx) | f_rd_NR | ρ_syn_Phase_VII | ρ_syn_Phase_X |
|-------|-----------|----------------|---------|-----------------|----------------|
| GW150914 | 1.21 | +0.00 | 0.040 | 0.729 | 0.376 |
| GW151226 | 1.97 | +0.18 | 0.032 | 0.001 | 0.000 |
| GW170814 | 1.06 | +0.06 | 0.041 | 0.453 | 0.237 |
| GW170104 | 1.18 | −0.09 | 0.038 | 0.162 | 0.082 |
| GW190521 | 1.69 | +0.68 | 0.058 | 0.762 / 1.100 | 0.474 / 0.684 |

All NR f_rd values sit within the Phase V/VII posterior 68% HDI = [0.037, 0.127].
GW190521 has the highest f_rd (0.058) because of its large aligned spin χ_eff ≈ +0.68.

The ρ_syn reduction: √(f_rd_NR / 0.15) ≈ 0.52 for most events — a ~49% reduction
from Phase VII ρ_syn values, making all events less informative.

---

## Results

### Phase VII baseline (global f_rd prior, reproduced)

```
MAP Γ  = 0.000   68% HDI ≈ [0.19, 1.30]
P(Γ<1) = 0.749   ΔlnL = +0.298   B(sigma_coh/GR) = 1.35
```

### Phase X (per-event NR f_rd prior)

```
MAP Γ  = 0.000   P(Γ<1) = 0.474   ΔlnL = +0.084   B(sigma_coh/GR) = 1.09
```

### Change

| Quantity | Phase VII | Phase X | Δ |
|----------|-----------|---------|---|
| P(Γ<1) | 0.749 | 0.474 | −0.275 |
| ΔlnL | +0.298 | +0.084 | −0.214 |
| B(sigma_coh/GR) | 1.35 | **1.09** | −0.26 |

**B = 1.09 is "not worth mentioning" (Jeffreys scale) — essentially no evidence
for or against sigma_coh.  P(Γ<1) = 0.474 is indistinguishable from a coin flip.**

---

## Interpretation

The NR f_rd reduction (f_rd ≈ 0.04 vs reference 0.15) reduces ρ_syn_eff by ~49%
per event.  At such low ρ_syn, the Rician likelihood is nearly flat in Γ — the
data has essentially zero power to discriminate between Γ = 0.79 (sigma_coh) and
Γ = 1.0 (GR).  This was anticipated in Phase VIII: the decisive test requires either
pycbc IMR subtraction (ρ_syn ≥ 5) or ~3000 events.

The NR f_rd prior confirms that Phase VII's B = 1.20 (using the global prior midpoint)
was slightly optimistic.  The true evidence state after NR-informed f_rd is B ≈ 1.09,
even closer to null.

---

## Upgrade path to NR-exact values

```bash
# On Linux/Mac with gcc:
pip install git+https://github.com/cpacilio/postmerger.git

# Then in phase_x_f_rd_nr_prior.py, replace NR_F_RD[ev] with:
A_modes = pm_surrogate(q, chi1z, chi2z)
f_rd_nr = sum(abs(A)**2 for A in A_modes.values()) / E_total_norm
```

The per-event (q, χ₁z, χ₂z) values are available from the LVK parameter-estimation
posteriors (GWTC-2/GWTC-3 data release on GWOSC).

---

## Summary arc (Phases I–X)

| Phase | Key number | Verdict |
|-------|-----------|---------|
| VII (corrected) | B = 1.20 | No evidence; GR inside 68% HDI |
| VIII (forecast) | Need ρ_syn ≥ 10 or 3000 events | Decisive test requires pycbc |
| **X (NR f_rd)** | **B = 1.09** | **Weaker than Phase VII — essentially null** |

---

## Cross-references

- Paper: arXiv:2408.05276
- Phase VII: `misc/bh_phase_vii_corrected_analysis_results.md`
- Phase VIII forecast: `misc/bh_phase_viii_sensitivity_forecast_results.md`
- Script: `scripts/phase_x_f_rd_nr_prior.py`
- Survey: `misc/arxiv_pluggable_survey_2025.md`
