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

*These rules exist because the σ-field framework lives or dies on the
precision of its foundations. Every magic number is a crack in the theory.
Every undocumented approximation is a place where the framework secretly fails.*

*"Nature already has the best answer — we just need to find it."*
