# Unification round 2 — the verified kill-test battery (2026-10-10)

One folder per item. The write-up is §15 of [misc/UNIFICATION_PROGRAM.md](../../../misc/UNIFICATION_PROGRAM.md); results,
figures and pre-registrations are in `misc/unification/round2/<item>/`; defects found are in `KNOWN_GAPS.md` and in the census
`misc/MENTAT_GAPS_UNIFICATION.md` (A9–A33).

| item | what | main script |
|---|---|---|
| U1 | σ-bridge vs five weak-field gates; PSR J1738+0333 on `nbody` gw_damping | `u1_kill_test_v2.py` |
| U2 | Loschmidt echo across N-body, inviscid and viscous SPH; entropy ledger | `run_nbody.py`, `run_sph*.py` |
| U3 | heat capacity as a dynamical response (kicked Plummer cluster) | `u3_worker.py`, `u3_branches.py`, `u3_analyze.py` |
| U4 | σ normalisation vs Gravity Probe A and Cassini | `u4_normalisation_table_v2.py` |
| U5 | `nbody`'s σ hook on Mercury's perihelion | `u5_run.py`, `u5_aggregate.py` |
| U6 | 1-D σ-wave solver (NEW UTILITY, scratch only) driven by a pulsating polytrope | `u6_*.py` |
| U7 | arrival-time exponents in the heat, SPH and N-body solvers | `u7_*.py` |
| U8 | range and extensivity: U ∝ N^5/3 vs pressure ∝ N | `u8_run.py` |
| U9 | Galilean boost; the 1PN sign defects | `u9_orbits.py`, `u9_dop853_v2.py`, `u9_report_v2.py` |
| U10 | Tolman potential, σ and steady T from one polytrope | `u10_run.py` |
| U11 | isothermal SPH column vs the barometric law | `u11_*.py` |
| U12 | e^σ through the material chain | `u12_material_chain*.py` |
| U13 | artificial viscosity's share of the SPH pressure virial | `u13_av_share*.py` |
| U14G | the buoyancy and conduction couplings (replaces the blocked U14) | `u14g_defect_check*.py` |
| T5 | torsion-balance window 1 cm–35 km | `t5_torsion_window_v2.py` |
| J6 | Jacobson 1995 in sympy (document: `misc/JACOBSON_DERIVATION.md`) | `jacobson_derivation_v2.py` |
| A2 | σ action options (document: `misc/SIGMA_ACTION_OPTIONS.md`) | `a2_sigma_action_v2.py` |
| N1 | nuclear/degenerate matter plan survey (document: `misc/NUCLEAR_MATTER_PLAN.md`) | `n1_survey_v2.py` |

**Provenance.** Each item was run by one agent, checked by three independent skeptics (reproduce; physics and literature;
could-it-fail and overclaim), repaired once and rechecked (workflow `wf_56966593-9d5`, Session 36). Files with a `_v2` suffix are
the repaired versions; the originals are kept for the audit trail. Every physical number comes from Mentat (`tools/mentat_client.py`
or the repo engines) with its source in the item's JSON.

**Running them.** Run from the repo root (`python -X utf8 scripts/unification/round2/<item>/<script>.py`). The scripts were written
in the session scratchpad and many write their outputs to absolute scratchpad paths; edit the output path at the top of a script
before re-running it. Large trace files (`*.npz`), logs and copies of literature were not committed.
