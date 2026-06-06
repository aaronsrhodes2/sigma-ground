# QuarkSum / SSBM — Project Instructions

**First time here?** Read [PLATINUM_RULES.md](../../PLATINUM_RULES.md) — this project inherits all universal rules there.
See also: [LOGGING_STANDARD.md](../../LOGGING_STANDARD.md) — format for session logs and operatic scene files.

## Session Persistence

Maintain `misc/SESSION_LOG.md` and create `misc/OPERATIC_PLAY_SCENE[N]_[TITLE].txt` files.
See [LOGGING_STANDARD.md](../../LOGGING_STANDARD.md) for templates and guidelines.

## Project Context

**The Big Picture — SSBM (Scale-Shifted Baryonic Matter)**
A unified cosmological framework proposing that extreme spacetime compression
within black hole accretion disks induces a localized scale transition in
infalling baryonic matter. Dark matter, the Big Bang, and cosmic expansion
are unified under this framework. Theory authored by Captain Aaron Rhodes.

**The Codebase — Pure Python, zero external dependencies**

- `quarksum/` — Particle inventory & mass closure tool.
  Resolves materials → molecules → atoms → particles → quarks.
  Proves the books balance. CLI: `python -m quarksum`

- `Materia/` — Spacetime geometry, σ (sigma) field computation, the full
  SSBM physics engine. Orbital mechanics, fluid dynamics, nucleosynthesis,
  gravitational waves, etc.

- `MatterShaper/` — Pure-Python 3D ray-tracer. Renders scenes from physics.
  Materials carry atomic composition; σ-values affect mass properties (not color).

- `local_library/` — Lightweight proof-of-concept for □σ = −ξR

**Testing**
- pytest, currently ~2047 tests, Baseline C: 89.8% pass rate (1838/2047)
- 2 permanently red files: test_jpl_ephemeris (network-gated) and
  test_position_precision (simulation-gated) — both marked EXTREME, expected
- 149 xfails: 30 science_gaps + 118 model_conformance + 1 chaotic_nbody
- Run: `pytest` or `pytest -v -s` for full physics reports

**Key physics concepts (don't panic)**
- σ (sigma) field — the scalar field governing scale transitions
- Space cavitation — a compressed spacetime pocket where matter is
  electromagnetically incommensurable with the surrounding universe
- r_s / R_H identity — Schwarzschild radius equals Hubble radius at junction
- Bond failure layers — 8 bond types fail in order during BH formation
