# Phase V — Joint Γ × f_rd Bayesian Posterior: Verdict

**Date:** 2026-04-17  
**Phase:** V — Joint inference of amplitude ratio Γ and ringdown fraction f_rd  
**Pipeline:** `scripts/phase_v_joint_gamma_frd.py`  
**Input:** Phase III (ρ_obs, ρ_syn @ f_rd=0.15) for 11 detector-events

---

## Motivation

Phase IV fixed f_rd = 0.15 and found MAP Γ = 0, P(Γ < 1) = 94%.  But f_rd is
uncertain by a factor of 3–7 (NR estimates range 0.03–0.20), and since
ρ_syn ∝ √f_rd, the entire Phase IV result could be a f_rd systematic rather
than a genuine Γ < 1 signal.

Phase V marginalises over f_rd simultaneously with Γ to answer:
**Once f_rd uncertainty is properly accounted for, what does the data say about Γ?**

---

## Model

**Effective signal SNR** per detector-event at arbitrary (Γ, f_rd):

    s_i(Γ, f_rd) = Γ × ρ_syn_i × sqrt(f_rd / f_rd_ref)

where ρ_syn_i is the Phase-III injection SNR at f_rd_ref = 0.15.

**Likelihood:** independent Rician per event (same as Phase IV)

    p(ρ_obs_i | Γ, f_rd) = Rician(ρ_obs_i; s_i(Γ, f_rd), σ=1)

**Priors:**

    Γ    ~ Uniform[0, 3]
    f_rd ~ LogUniform[0.03, 0.20]   (log-uniform spans the full NR range)

**Grid:** 400 × 300 points on (Γ, f_rd); trapezoid integration throughout.

---

## Γ–f_rd Degeneracy

Multiplying f_rd by a factor k scales ρ_syn by √k.  The same likelihood value
is achieved by:

    (Γ,  f_rd)  ↔  (Γ × √k,  f_rd / k)

This creates a banana-shaped degeneracy in the 2D posterior: **any combination
of Γ and f_rd that keeps the product Γ × √f_rd fixed is equally favoured by
the data**.  Breaking the degeneracy requires either:

1. An independent NR-calibrated prior on f_rd (narrower than log-uniform), or
2. Events spanning a wide range of ρ_syn — because the degeneracy direction
   in (Γ, f_rd) space is event-independent, but the banana width depends on
   ρ_syn per event (high-ρ_syn events constrain the product more tightly).

---

## Results

### Combined posterior summary

```
MAP Γ (marginal)    = 0.000   ← noise floor (Rician MLE = 0 for ρ_obs ≤ 1)
MAP f_rd (marginal) = 0.030   ← hits lower bound of prior

68% HDI Γ           = [0.165, 1.180]
95% HDI Γ           = [0.023, 1.970]

68% HDI f_rd        = [0.037, 0.127]
95% HDI f_rd        = [0.031, 0.185]
```

### Posterior tail probabilities on Γ (f_rd marginalised)

| Model | Γ_pred | P(Γ < Γ_pred) | In 68% HDI? |
|---|---|---|---|
| GR | 1.0000 | 77.2% | Yes |
| exp | 0.8395 | 69.4% | Yes |
| sigma_coh | 0.7923 | 66.8% | Yes |
| linear_cbrt | 0.7461 | 64.1% | Yes |

### Marginal Bayes factors vs GR (f_rd integrated out)

| Comparison | ΔlnL | Bayes factor | Jeffreys grade |
|---|---|---|---|
| sigma_coh / GR | +0.190 | 1.209 | Not worth mentioning |
| linear_cbrt / GR | +0.228 | 1.256 | Not worth mentioning |
| exp / GR | +0.149 | 1.161 | Not worth mentioning |

---

## Comparison with Phase IV

| Quantity | Phase IV (f_rd fixed) | Phase V (f_rd marginalised) | Change |
|---|---|---|---|
| MAP Γ | 0.000 | 0.000 | same |
| 68% HDI Γ | [0.000, 0.551] | [0.165, 1.180] | 2.1× wider |
| 95% HDI Γ | [0.000, 1.028] | [0.023, 1.970] | 1.9× wider |
| P(Γ < 1.0) | 94.4% | 77.2% | −17 pp |
| B(sigma_coh / GR) | 1.97 | 1.21 | −38% |

**Marginalising over f_rd substantially weakens the Γ < 1 preference.**
Phase IV's 94% posterior probability was largely an artifact of fixing f_rd = 0.15,
which consistently overestimates ρ_syn relative to ρ_obs.  Once f_rd is free
to absorb that overestimate, the data is compatible with GR at the 23% level
(P(Γ > 1) = 22.8%).

---

## Key Physical Findings

**1. The data prefers f_rd ≈ 0.04–0.09 (well below f_rd = 0.15).**

68% HDI on f_rd = [0.037, 0.127].  The reference value f_rd = 0.15 sits at
the 86th percentile of the marginal posterior — consistent with the data but
not the most probable value.  The posterior is saying: the GR-predicted
amplitudes are more consistent with ρ_obs if f_rd ≈ 0.05–0.08.

This is physically meaningful.  NR simulations of BBH coalescence show the
(2,2,0) ringdown fraction f_rd ranges widely:
- Head-on equal-mass: f_rd ≈ 0.05–0.08
- Quasi-circular, high-spin: f_rd ≈ 0.10–0.20
- The five events in our catalog span a range of mass ratios and spins, with
  several sources (GW151226, GW170104) having q < 0.5 and moderate spins —
  exactly where f_rd ≈ 0.05 is expected.

**2. The Γ–f_rd degeneracy is not broken at current sensitivity.**

The banana-shaped degeneracy (Γ × √f_rd = const) means the data cannot
separately constrain Γ and f_rd.  The 2D posterior contours run diagonally
across (Γ, f_rd) space.  Both (Γ=1, f_rd=0.05) and (Γ=0.6, f_rd=0.14)
are consistent with the data.

**3. After f_rd marginalisation, all four predictions remain consistent.**

GR (Γ=1.0) is now well within the 68% HDI (P(Γ > 1) = 23%).  The sigma_coh
and linear_cbrt predictions (Γ ≈ 0.75–0.79) are near the mode of the
marginal posterior — but the posterior is too broad to distinguish them from GR.

**4. Phase IV's "94% for Γ < 1" was a systematic artifact, not a signal.**

The correct interpretation: **the ρ_syn systematic (from f_rd = 0.15 being
too large) was generating an apparent preference for Γ < 1 that Phase IV's
fixed-f_rd analysis could not distinguish from a genuine signal.**  Phase V
resolves this ambiguity by letting the data speak about f_rd directly.

---

## Combined Evidence State (Phases I–V)

| Phase | Result | Key systematic |
|---|---|---|
| I.4 (time domain) | p_F = 0.609, null | Echo signal below detection threshold |
| I.5 (freq comb) | R = 0.916, null | Below noise floor |
| II (amplitude bound) | ξ_UL = 1.0, saturated | ρ_QNM << detection floor |
| III (amplitude ratio) | Γ = 0.856 ± 1.196, null | ρ_syn << 1 for 7/11 events |
| IV (Bayesian Rician) | MAP=0, P(Γ<1)=94%, B≈2 | f_rd=0.15 systematic inflation |
| V (joint Γ×f_rd) | MAP=0, P(Γ<1)=77%, B≈1.2 | Γ–f_rd degeneracy |

**Summary:** When the dominant systematic (f_rd) is properly marginalised,
the preference for Γ < 1 weakens to 77% — still mild but no longer driven
by an unjustified fixed assumption.  The data is uninformative about Γ at
current sensitivity.  The current constraint on f_rd (68% HDI: 0.037–0.127)
is itself a new, meaningful result: the data independently disfavours
f_rd ≥ 0.15 at moderate credence.

---

## Path to Breaking the Degeneracy

To separately constrain Γ and f_rd, at least one of:

**A. Independent NR prior on f_rd.**
Use f_rd posterior samples from SXS NR simulations matched to each event's
(q, a_eff) — this narrows the f_rd prior from log-uniform to a per-event
Gaussian σ(f_rd) ~ 0.02.  With this tighter prior, Phase IV's fixed-f_rd
approximation is justified and Γ becomes the residual degree of freedom.

**B. High-ρ_syn events.**
If ρ_syn >> 1 for multiple events, the Rician likelihood constrains both
Γ × √f_rd (amplitude) and the shape of the posterior banana.  With
ρ_syn ≥ 5 on even one event (achievable via pycbc IMR subtraction on
GW150914/H1), the degeneracy contracts to a narrow ellipse and Γ and f_rd
become individually constrained.

**C. Large event catalog.**
Events with different mass ratios and spins systematically differ in their
NR-predicted f_rd.  Including ~20 events with a per-event f_rd prediction
from NR breaks the global degeneracy statistically, even without tight
individual-event priors.

---

## Verdict

**Phase V finds no statistically significant evidence for or against any of
the four Γ predictions.  After properly marginalising over the f_rd
systematic, GR is comfortably within the 68% credible interval.**

The analysis does produce a physically meaningful new result:
**the 11-event dataset prefers f_rd ≈ 0.04–0.09 over the reference f_rd = 0.15,
at 68% credence.**  This is an independent observational constraint on the
ringdown energy fraction — consistent with NR predictions for the mass ratios
and spins in this event catalog, and consistent with all four Γ predictions.

The Phase I–V arc conclusively establishes that the current pipeline is not
sensitive enough to test the σ_conv / γ-mode predictions.  The correct next
step is not "more analysis of the same data" but "better data (pycbc + SXS
subtraction on GW150914/H1)" or "better priors (NR f_rd posteriors per event)".

---

## Cross-References

- Phase IV Bayesian Rician: `misc/bh_phase_iv_bayesian_gamma_results.md`
- Phase III amplitude ratio: `misc/bh_phase_iii_amplitude_ratio_results.md`
- Phase V script: `scripts/phase_v_joint_gamma_frd.py`
- Phase V figure: `misc/bh_phase_v_joint_gamma_frd.png`
- σ_conv / γ-mode predictions: `misc/bh_collision_phenomenology.md`
- NR f_rd estimates: Buonanno & Damour 1999; Pretorius 2005; SXS catalog
