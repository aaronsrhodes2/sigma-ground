# η Empirical Verdict — 2026-05-15

**Status:** ADOPTED — overrides Phase XI (2026-04-17) `bh_phase_xi_eta_candidates_results.md`.

## Decision

`ETA` is no longer claimed to be a derived quantity. It is now an **[EMPIRICAL-INPUT]** alongside `XI`, set equal to the DESI 2024 Union3 HDE c² fit:

```python
ETA = ETA_HDE_UNION3 = C_HDE_UNION3 ** 2 = 0.642**2 ≈ 0.412164
ETA_UNCERTAINTY_1SIGMA = 2 * 0.642 * 0.028 ≈ 0.036
```

Source: arXiv:2411.08639 (DESI 2024, Holographic Dark Energy fit, CMB+DESI+Union3 dataset).

The SSBM free-input count moves from **1 (XI)** to **2 (XI, ETA)**.

## What was rejected

Two earlier claims about η, both treated as derivations:

### Rejection 1 — `ETA = 0.4153` as "DERIVED from ρ_DE constraint"

The story was: "Matching ρ_DE(observed) = η × ρ_released gives η = 0.4153." On audit:

- The `ρ_released` side was not independently constructed in the codebase. The only structural argument supplied was a "convenient coherence ratio related to the golden spiral" — i.e. a number chosen *because* it involved φ — not a self-contained physical prediction.
- This makes the derivation **partly circular**: η was tuned against ρ_DE under a golden-spiral assumption that was itself unconstrained.
- A genuinely first-principles derivation would specify ρ_released without reference to η or to ad-hoc golden-ratio choices.

**Verdict:** the "derivation" label oversold the evidence. 0.4153 is empirically plausible (DESI 1σ contains it) but it is not derived. Removed.

### Rejection 2 — `ETA_FORMULA = exp(-φ/σ_conv) ≈ 0.4158` as a "first-principles candidate"

The Phase XI script `scripts/phase_xi_eta_candidates.py` ran a **formula search over {ξ, σ_conv, π, e, φ}** looking for an expression that hits 0.4153. The result `exp(-φ/σ_conv)` was reported with a 0.125% residual gap and tagged `[SPECULATIVE]` pending derivation.

On audit:

1. **Search space is large.** Five base constants under {+, −, ×, ÷, ^, log, exp} yields order 10³ simple expressions; finding a 0.1% match by chance is *expected*, not surprising. This is the classic numerology setup.
2. **Target was already φ-flavoured.** The original ETA = 0.4153 was selected via a "golden-spiral" heuristic, which by construction makes any φ-containing formula a likely match. Recovering φ in the formula recovers an input, not a prediction.
3. **0.125% is the signature of numerology, not derivation.** A true expression for η would be exact. A 1-part-in-800 residual gap is what you get from search-among-many-candidates, not from physics.
4. **No physical mechanism.** φ appears in real physics (quasicrystal symmetries, KAM theorem) with a structural argument. No such argument was offered here — only the formula match.

**Verdict:** ETA_FORMULA was numerology. The constant is retained in `constants.py` set to `None` with tag `[REJECTED 2026-05-15]` so:
- Any code that imports it still works (no ImportError surprises in downstream modules).
- Any code that *uses it as a number* fails loudly (TypeError on arithmetic with `None`).
- The historical entry is visible in PROVENANCE.md under the Rejected section.
- Re-adopting it requires deleting the rejection comment and providing a physical mechanism, both of which would show up in code review.

## What changes downstream

| Surface | Before | After |
|---|---|---|
| `ETA` value | 0.4153 | 0.412164 (= 0.642²) |
| `ETA_FORMULA` | 0.4158 (φ-formula) | None (rejected) |
| Free-input count | 1 (XI) | 2 (XI, ETA) |
| `eta_candidates()` | 4 entries incl. 'formula' | 3 entries (formula dropped) |
| `eta_coincidence_report()` | "triple coincidence" UP-001 framing | "single external corroboration" |
| `THETA_ENTANGLEMENT_PER_DIM = ETA**(1/3)` | 0.7463 (from 0.4153) | 0.7442 (from 0.412164) |
| `tests/test_eta_derivation.py` | pinned 0.125% formula gap | pins empirical-input identity |
| `test_cosmology.py` | tested formula agreement | tests Union3 identity, no formula |
| PROVENANCE.md count | 1 free-input, 0 rejected | 2 free-input, 1 rejected |

## What still needs to happen if η ever becomes derivable

A future derivation of η would need:
1. A physical mechanism — a calculation, not a formula match — that predicts η in terms of `XI`, fundamental constants, or other unambiguously measured inputs.
2. The prediction must be **numerical**, not just "in the DESI band" (the band is wide enough for many expressions).
3. The derivation must not invoke φ unless φ enters through a structural argument (e.g. an eigenvalue of a known dynamical system), not by formula-search.
4. When/if this happens, `ETA_FORMULA = None` would be replaced by the derived expression with the matching tag (`[DERIVED]`) and the rejection tombstone moved or annotated.

Until then, η is empirical. That's honest; numerology was not.

## References

- DESI 2024 HDE fit: arXiv:2411.08639 (the actual anchor)
- Superseded Phase XI write-up: `misc/bh_phase_xi_eta_candidates_results.md`
- Superseded Phase XI script: `scripts/phase_xi_eta_candidates.py`
- Constants: `sigma_ground/field/constants.py` (ETA, ETA_FORMULA, ETA_HDE_UNION3, ETA_UNCERTAINTY_1SIGMA)
- Provenance map: `sigma_ground/field/PROVENANCE.md` (Free Inputs section, Rejected section)
- Test pins: `tests/test_eta_derivation.py`, `sigma_ground/field/interface/test_cosmology.py`
