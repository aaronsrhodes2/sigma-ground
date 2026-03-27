# sigma-ground — Project Instructions

## Session Log
Maintain `misc/SESSION_LOG.md`.
Append a new session block at the end of each working session or when asked.

## Operatic Play — Scene Files
At the end of each session (or when asked), produce `misc/OPERATIC_PLAY_SCENE[N]_[TITLE].txt`
where [N] is the session number and [TITLE] is a short snake_case name for that session.
Each scene file covers that session only and is self-contained with its own
DRAMATIS PERSONAE header. Draw from SESSION_LOG.md.

## Project Context

**The Codebase — Pure Python, zero external dependencies**

- `sigma_ground/` — Unified physics library.
  Three sub-packages:
  - `sigma_ground.inventory` — Particle inventory & mass closure.
    Resolves materials → molecules → atoms → particles → quarks.
    CLI: `python -m sigma_ground.inventory`
  - `sigma_ground.field` — σ-field scalar physics. Constants, bounds,
    entanglement, spacetime geometry. □σ = −ξR.
    Core: constants, scale, nucleon, binding, bounds, entanglement.
    New physics: relativity, electrodynamics, decay, gr_basics.
    Interface layer: thermal, optics, statistical, plasma, orbital, fluid,
    quantum, mechanical, semiconductor_optics, crystal_field, and more.
  - `sigma_ground.dynamics` — N-body dynamics, SPH fluid, Barnes-Hut
    gravity, leapfrog integrator.

- `GOLDEN_RULES.md` — Eight rules governing all physics code in this project.
- `tests/` — Full test suite (~1260+ tests).
- `examples/` — Five standalone usage examples.
- `misc/` — Session log and operatic play scenes (13+ scenes).

**Rendering lives in matter-shaper (sibling project at ../matter-shaper/)**

**Testing**
- pytest, 1260+ tests passing (1198 original + new physics modules)
- Run: `pytest` or `pytest -v -s`

**Key physics concepts (don't panic)**
- σ (sigma) field — scalar field governing scale transitions
- Space cavitation — compressed spacetime pocket, electromagnetically
  incommensurable with surrounding universe
- r_s / R_H identity — Schwarzschild radius equals Hubble radius at junction
- Bond failure layers — 8 bond types fail in order during BH formation
