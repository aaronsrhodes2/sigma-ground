# Golden Rules of sigma-ground Physics

These rules govern how all physics code is written in this project.
They encode lessons learned building the library and ensure every module
stays consistent, correct, and genuinely connected to the σ-field framework.

---

## Rule 1 — One Source of Truth for Constants

All measured values live in `sigma_ground/field/constants.py`.
Every module imports from there. **No magic numbers anywhere else.**

```python
# Correct
from sigma_ground.field.constants import C, HBAR, G

# Wrong — never do this
C = 3e8
```

Cosmological constants (G, C, ℏ), nuclear data (proton mass, quark masses),
electrostatics (e, ε₀), and SSBM parameters (ξ, η, σ_conv) all live there.

---

## Rule 2 — Derive, Don't Assume

Every non-measured value shows its derivation from measured constants
inline. Comment the formula.

```python
# Derived: α = e² / (4πε₀ℏc)
ALPHA = E_CHARGE**2 / (4 * math.pi * EPS_0 * HBAR * C)  # ≈ 1/137.036
```

If a derivation requires more than a line, put it in the docstring.

---

## Rule 3 — Explicit Domain Bounds

Every model defines where it is valid. Document the breakdown condition.
Use SAFE / EDGE / WALL / BEYOND classification from `sigma_ground/field/bounds.py`
when integrating with other field modules.

```python
def lorentz_factor(v):
    """Valid for 0 ≤ v < c. Diverges at v = c."""
    if v >= C:
        raise ValueError(f"v={v} ≥ c={C}: Lorentz factor undefined")
```

---

## Rule 4 — Wire to σ

Every physics module must include at least one function showing how the
σ-field modifies its standard result. This is what makes sigma-ground
different from a generic physics utilities package.

The pattern:
```python
def sigma_X(sigma, standard_input):
    """How X changes under σ-field compression."""
    return standard_result * scale_factor(sigma)
```

The σ-field value in everyday matter is negligible (Earth surface: ~7×10⁻¹⁰).
The σ-functions become meaningful approaching black hole accretion disks and
at the Big Bang (σ → σ_conv ≈ 1.849).

---

## Rule 5 — Physics Never Imports Rendering

The ARCHITECTURE.md rule. Inviolable.

```
sigma_ground.field     ✓ may import sigma_ground.dynamics
sigma_ground.field     ✗ may NOT import matter_shaper or any renderer
sigma_ground.dynamics  ✓ may import sigma_ground.field
```

---

## Rule 6 — Nature Already Has the Best Answer

Prefer exact analytic solutions over approximations. Use numerical methods
only when no closed form exists. Cite the source (textbook, PDG, NIST).

```python
# Hawking temperature — exact analytic result
# Hawking 1974, Comm. Math. Phys. 43, 199
def hawking_temperature(M):
    return HBAR * C**3 / (8 * math.pi * G * M * K_B)
```

---

## Rule 7 — No Averages, No Assumptions, No Gaps

Track all terms. Don't silently drop small contributions. When approximating,
state what you are dropping and why it is safe to drop it.

```python
# OK: explicit approximation
def binding_energy_approx(A, Z):
    """Bethe-Weizsäcker formula. Drops pairing term for simplicity."""
    ...

# Not OK: silent drop
def binding_energy(A, Z):
    volume = ...  # pairing term silently missing — don't do this
```

---

## Rule 8 — Tests Prove the Physics

Every new function gets at least one test checking known values against
standard references (NIST, PDG, textbooks). Physics tests are not unit
tests — they are verification against reality.

```python
def test_electron_rest_energy():
    # NIST CODATA: electron rest energy = 0.51099895 MeV
    E_mev = rest_energy(M_ELECTRON_KG) / (E_CHARGE * 1e6)  # J → MeV
    assert abs(E_mev - 0.511) < 0.001  # within 0.2%
```

---

## Rule 9 — If One, Then All

When adding a property for one material, element, or entity, add it for
**every applicable entry** in that database. Incomplete databases are silent
gaps — they compile, they run, and they give wrong answers when simulated.

```python
# Wrong — cherry-picked data
SUPERCONDUCTORS = {
    'aluminum': {'T_c_K': 1.175, 'kappa': 0.01, ...},
    'niobium':  {'T_c_K': 9.25,  'kappa': 1.05, ...},
    # 45 other superconductors silently missing — simulation sees only 2
}

# Correct — every known superconductor, every field populated
SUPERCONDUCTORS = {
    'aluminum': {'T_c_K': 1.175, 'kappa': 0.01, 'kappa_source': 'measured', ...},
    'niobium':  {'T_c_K': 9.25,  'kappa': 1.05, 'kappa_source': 'measured', ...},
    'titanium': {'T_c_K': 0.40,  'kappa': 0.09, 'kappa_source': 'derived',  ...},
    # ... all 53 elements + compounds, no gaps
}
```

If a value cannot be measured, derive it and flag the provenance. If it
truly cannot be determined, flag it explicitly — never silently omit the entry.

---

*These rules exist because the σ-field framework lives or dies on the
precision of its foundations. Every magic number is a crack in the theory.
Every undocumented approximation is a place where the framework secretly fails.
Every missing database entry is a thing the simulation cannot see.*

*"Nature already has the best answer — we just need to find it."*

---

## Vision — The Matter Information Cascade

This library is built on a discovery: **you can derive all material properties
from a small number of measured atomic inputs through physics equations.**

For metals: ~15 measured numbers per element → hundreds of derived properties.
For organics: ~42 measured atomic properties (7 atoms × 6 each) plus 7
homonuclear bond energies → all covalent bonds → all molecular geometries →
all intermolecular forces → all bulk properties.

The equations *are* the compression. You never need to store what you can derive.

This is the same principle that lets the inventory system load the entire
observable universe and query any particle in it without a massive thinking
machine. A lookup table of every molecule's properties would be infinite.
The derivation chain that produces them from atoms fits in a few kilobytes.

**The goal:** virtual matter that behaves exactly like real matter in every way,
constructed from first principles plus a minimal set of measured seeds. No
lookup tables of bulk properties. No fitting parameters disguised as physics.
No magic numbers that haven't been traced to a measurement or a derivation.

The cascade flows downward:
```
measured atomic properties (χ, r_cov, IE₁, mass, D(A-A))
  → bond energies (Pauling)
    → bond lengths (Schomaker-Stevenson)
      → molecular geometry (VSEPR)
        → intermolecular forces (H-bond, London, dipole-dipole)
          → bulk properties (density, viscosity, surface tension, ...)
```

Each step is a physics equation, not a lookup. Each equation cites its source.
Each measured input is labeled MEASURED. Each derivation is labeled
FIRST_PRINCIPLES. The provenance is the proof.

*"We proved we could load the entire universe and query any particle in it.
The matter information cascade is the same insight applied to chemistry:
the compression is the physics itself."*
