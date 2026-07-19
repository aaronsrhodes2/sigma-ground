# Curated-table miss-behavior audit — 2026-07-15

## Trigger

The Captain's first-ever direct question to the live `/chat` endpoint ("simulate
a zinc rod stuck in a layer of oxidizing, alkaline soil and see it corrode over
5 years") exposed a routing tiebreak bug and a missing zinc entry, fixed same-day
in `e471021` ("Fix corrosion routing tiebreak + add zinc"). That fix patched
*this one case*. This audit asks the general question: **when a curated lookup
table misses, does the system refuse honestly or silently substitute a wrong
value?** — across every place a user-named material/element reaches a curated
table, not just corrosion.

A prior pass at this audit (same day) was cut off mid-run by a usage-limit
interruption before producing any artifact. This is a full redo, scoped
narrower and finished.

## The two lanes

sigma-ground answers physics questions through two independent code paths:

1. **MCP `tools/` layer** (`sigma_ground/mcp/tools/*.py`) — discrete tool
   calls (`first_ionization_energy`, density/Young's-modulus lookups, body
   lookups, etc.), the layer the 150-question Qwen benchmark exercises.
2. **Materia front door** (`sigma_ground/materia/translator.py` +
   `manifest.py` → `scenarios.py`) — the natural-language dispatcher behind
   the live `/chat` endpoint. This is what actually served the Captain's zinc
   question.

**Finding 1 — the MCP tools/ layer already does this right.** Spot-checked
`atomic.py`, `astronomy.py`, `materials.py`, `playground.py`: every miss
returns a structured refusal — `"'{key}' not in lookup. Available: [...]"` —
confirmed live in benchmark logs (`results/sigma_ground_run.json` etc. show
this firing correctly for `Xe`, `mars`, `carbon_nanotube`, `liquid nitrogen`,
`sgr_a_star`). No fix needed here; this lane was already audited by the
150-question benchmark's own design.

**Finding 2 — the Materia front door was the actual gap.** The interface
functions it calls (e.g. `corrosion.py::pilling_bedworth_ratio`) already raise
`KeyError(f"Unknown material: {material_key!r}")` for a bad key — the honesty
safety net exists at that layer too. But `translator.py::_extract_material()`
**never produces a key the interface would reject**: if no material in the
question matches its ~25-entry synonym table, it returns `"iron"` — a
valid-but-wrong key — silently. The interface's honest-refusal code never
gets a chance to fire, because it never sees an invalid key.

Only **one** of the 22 verbs with a `material_key` slot (`acoustics`) opted
into the existing `material_required` gate that makes the router *decline*
rather than route with a silent default. The gate and its rationale were
already correctly designed and documented in `translator.py`
(`_named_material()`, lines ~200-214) — it just wasn't applied consistently.

## Fix applied

Added `"material_required": True` to the 10 verbs where **every** canonical
example/test question already names a real material or a recognized material
class word (verified empirically, not just by inspection — see Verification):

- `corrosion_attack` — the case that started this (also added missing
  `"corroding"`/`"corroded"` trigger forms: the manifest's own worked example,
  "...corroding over 5 years", didn't contain the literal substring "corrode"
  and so didn't route even after today's tiebreak fix — a second, narrower
  instance of the same incident, caught by re-running the routing harness)
- `elastic_solid`, `tribology`, `optical_dispersion`, `viscoelastic_material`,
  `condensed_matter` (all `_ROUTABLE_NEW` verbs)
- `material_profile`, `structural_response`, `thermal_response`,
  `material_full_profile` (base `VERB_MANIFEST` verbs)

Each got the same inline comment pattern already used for `acoustics`,
explaining why silent-default is wrong for that verb specifically.

## Verification

- `python misc/routing_corpus.py` (the manifest's own self-check — every
  verb's examples + a held-out generalization set + adversarial negatives):
  **131/132 → 132/132** positive, **30/30** held-out, **18/18** negative.
  (The 1 baseline miss was the "corroding" gap above; now fixed.)
- `pytest sigma_ground/materia/ sigma_ground/mcp/`: 288 passed, 2 xfailed, 3
  failed. Confirmed via `git stash` that **all 3 failures are pre-existing**
  and reproduce identically with `manifest.py` unmodified — unrelated to this
  change (`test_drag_heating_scenario_self_validates`,
  `test_qa_matches_wolfram_on_subset_wa_got_right`,
  `test_honest_100_percent_coverage`/`corrosion.environment_assessment`).

## Found, NOT fixed — needs judgment or more work

Gating these the same way would have broken their own worked examples or an
existing test, which means the fix isn't a mechanical flag-flip for them:

- **`piezoelectric_material`** — both manifest examples ("PZT under stress",
  "of quartz") name materials that **aren't in the `MATERIALS` synonym table
  at all**. This verb currently always answers for its hardcoded default
  (`PZT4`) regardless of what the user names — worse than a routing gate can
  fix; needs the synonym table extended (or a dedicated piezo-material table
  wired into `_material_synonyms()`) before it can be gated honestly.
- **`subsurface_scattering`** — both examples ("skin", "marble") reference
  materials entirely outside `MATERIALS` (25 engineering metals/composites).
  Same shape of problem as above, different table.
- **`magnetic_material`** — 1 of 2 examples ("dielectric breakdown field of
  quartz") hits the same missing-synonym problem.
- **`magnetic_hysteresis`** — 1 of 2 examples ("...of a transformer core")
  names no material or class word at all.
- **`thermoelectric`** — held-out case uses "bismuth telluride" (not in the
  synonym table) and an existing `test_routing.py` POSITIVE case
  ("thermoelectric figure of merit") expects a successful route with **no**
  material named at all.
- **`metallurgy`** — existing `test_routing.py` POSITIVE case ("Hall-Petch
  yield strength vs grain size") expects success with no material named.
  This may be intentional — Hall-Petch is a general σy = σ0 + k/√d
  *relationship*, arguably answerable generically — but that's a judgment
  call for the Captain, not something to silently decide either way.

## Not in scope for this pass

This audit targeted the Materia front door specifically, since that's the
layer that produced the live zinc incident and the layer the 150-question
benchmark doesn't exercise. The ~90 `field/interface/*.py` files with their
own module-level curated tables (corrosion, band structure, refractive
indices, ionization energies, etc.) were spot-checked, not swept exhaustively
— that was the original scope of the workflow that got cut off by the usage
limit. If a full file-by-file sweep of the interface layer is wanted, that's
still open.
